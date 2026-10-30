//! A prompt chunk's pieced pass and exchanges, isolated so their diagnostic spans stay beside each stream operation.
const relay = @import("cuda_relay.zig");
const taps = @import("cuda_taps.zig");
const fw = @import("cuda_forward.zig");
const prof = @import("cuda_prof.zig");
const latent = @import("cuda_latent.zig");
const shared = @import("cuda_shared.zig");
const fronts = @import("cuda_forward_fronts.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const Site = fw.Forward.Site;
const room = fw.Forward.room;
const share = fw.Forward.share;

/// forward.pieced_layers: a prompt chunk's layers with exchanges in row pieces; same kernels and bits as unpieced.
pub fn pieced(f: *const Forward, p: *const Pieces, R: usize, cut: []const [2]usize) !void {
    const layers = f.ranks[0].w.layers;
    const ahead = fronts.wanted(f, p, cut); // TF_GLM_OVN_FRONTS: each block's front runs in the glue before it
    for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
    var li: usize = 0;
    var di: usize = 0;
    for (layers, 0..) |_, j| {
        prof.context(f.prof, j, null, .mixer);
        for (f.ranks) |r| if (r.segs.len > 0) try shared.mixer(f, r, r.w.layers[j], li, di, R, null) else try f.mixerFrom(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null, ahead and j > 0);
        try exchangePieces(f, p, j, .mixer, cut, false);
        try gluePieces(f, p, j, .mixer, cut, R, ahead);
        prof.context(f.prof, j, null, .ffn);
        for (f.ranks) |r| if (ahead) try fronts.ffnRest(f, r, r.w.layers[j], R) else try f.ffn(r, r.w.layers[j], R, null);
        try exchangePieces(f, p, j, .ffn, cut, ahead);
        try gluePieces(f, p, j, .ffn, cut, R, ahead and j + 1 < layers.len);
        if (f.taps) |t| try tapPieces(f, t, j, cut);
        switch (layers[j].mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    if (f.taps) |t| try shareTaps(f, t, cut);
}

/// overlap.Overlap.partials: pieces filled on the main stream, exchanged on the second; `fronted`: see fronts.fill.
pub fn exchangePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize, fronted: bool) !void {
    defer prof.context(f.prof, j, null, if (site == .mixer) .mixer else .ffn);
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    var later: usize = 0; // TF_GLM_OVN_PEER_FIRST: pieces whose own rows fill once every piece's exchange is posted
    for (cut, 0..) |piece, q| {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const lo = piece[0];
        const n = piece[1] - lo;
        const split = f.splits(n);
        const first = split and fronts.peerFirst(f, p, n);
        if (first) {
            for (f.ranks) |r| { // the peers' rows: those before this rank's share and those after it
                const mine = share(n, world, r.w.share.rank);
                for ([_][2]usize{ .{ lo, lo + mine[0] }, .{ lo + mine[1], piece[1] } }) |span| if (span[1] > span[0]) {
                    try fronts.fill(f, r, r.w.layers[j], site, span[0], span[1], r.b.part + span[0] * D * 4, fronted);
                };
            }
            later |= @as(usize, 1) << @intCast(q);
        } else for (f.ranks) |r| try fronts.fill(f, r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4, fronted);
        try p.filled[q].record(f.stream);
        var stamp = prof.span(f.prof, p.side, .wait_filled);
        try p.side.wait(p.filled[q]);
        stamp.end();
        stamp = prof.span(f.prof, p.side, .partials);
        try partials(f, p, lo, n, split, first);
        stamp.end();
        try p.gathered[q].record(p.side);
    }
    for (cut, 0..) |piece, q| if (later >> @intCast(q) & 1 == 1) {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const lo = piece[0];
        const n = piece[1] - lo;
        for (f.ranks) |r| { // straight into the rank's own slot of the room, where partials would have copied them
            const mine = share(n, world, r.w.share.rank);
            try fronts.fill(f, r, r.w.layers[j], site, lo + mine[0], lo + mine[1], room(r, lo) + r.w.share.rank * (mine[1] - mine[0]) * D * 4, fronted);
        }
    };
}

/// overlap.Overlap.glue: each piece's glue once its exchange is in, its rows then shared; `front`: see fronts.ahead.
pub fn gluePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize, R: usize, front: bool) !void {
    defer prof.context(f.prof, j, null, if (site == .mixer) .mixer else .ffn);
    const world = f.ranks[0].w.share.world;
    const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
    var sent: usize = 0;
    for (cut, 0..) |piece, q| {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const lo = piece[0];
        const n = piece[1] - lo;
        var stamp = prof.span(f.prof, f.stream, .wait_gathered);
        try f.stream.wait(p.gathered[q]);
        stamp.end();
        if (!f.splits(n)) {
            for (f.ranks) |r| try f.glueRows(r, j, site, lo, piece[1], room(r, lo));
            if (front) for (f.ranks) |r| try fronts.ahead(f, r, j, site, R, lo, piece[1]);
            continue;
        }
        for (f.ranks) |r| {
            const mine = share(n, world, r.w.share.rank);
            try f.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
        }
        try p.glues[q].record(f.stream);
        stamp = prof.span(f.prof, p.side, .wait_glues);
        try p.side.wait(p.glues[q]);
        stamp.end();
        stamp = prof.span(f.prof, p.side, .glued);
        try shareGlued(f, p, lo, n, last);
        stamp.end();
        try p.shared[q].record(p.side);
        sent |= @as(usize, 1) << @intCast(q);
        if (front) for (f.ranks) |r| {
            const mine = share(n, world, r.w.share.rank);
            try fronts.ahead(f, r, j, site, R, lo + mine[0], lo + mine[1]);
        };
    }
    for (0..cut.len) |q| if (sent >> @intCast(q) & 1 == 1) {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const stamp = prof.span(f.prof, f.stream, .wait_shared);
        try f.stream.wait(p.shared[q]);
        stamp.end();
        if (front) for (f.ranks) |r| try fronts.peers(f, r, j, site, R, cut[q]);
    };
}

/// A piece's partials on the side stream, gathered or split; `own_later`: own rows fill into their slot later, no copy.
pub fn partials(f: *const Forward, p: *const Pieces, lo: usize, n: usize, split: bool, own_later: bool) !void {
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    const bytes = n * D * 4;
    if (!split) switch (f.exchange) {
        .local => for (f.ranks) |to| for (f.ranks) |from| {
            if (to.w.share.rank == from.w.share.rank) continue;
            const off = from.w.share.rank * bytes;
            try f.copy(p.side, room(to, lo) + off, room(from, lo) + off, bytes);
        },
        .nccl => |x| {
            const r = f.ranks[0];
            try x.lib.check(x.lib.api.ncclAllGather(room(r, lo) + r.w.share.rank * bytes, room(r, lo), n * D, .f32, x.comm, p.side.handle), "ncclAllGather");
        },
    } else switch (f.exchange) {
        .local => for (f.ranks) |to| {
            const mine = share(n, world, to.w.share.rank);
            const own = mine[1] - mine[0];
            for (f.ranks) |from| if (!own_later or from.w.share.rank != to.w.share.rank) {
                try f.copy(p.side, room(to, lo) + from.w.share.rank * own * D * 4, from.b.part + (lo + mine[0]) * D * 4, own * D * 4);
            };
        },
        .nccl => |x| {
            const r = f.ranks[0];
            const me = r.w.share.rank;
            const mine = share(n, world, me);
            const own = mine[1] - mine[0];
            const plan: relay.Relay = .{ .world = world, .rank = me, .n = n };
            if (f.ring) try relay.partials(x.lib, x.comm, p.side, plan, room(r, lo), r.b.part, lo, D) else try relay.splitPartials(x.lib, x.comm, p.side, plan, room(r, lo), r.b.part, lo, D);
            if (!own_later) try f.copy(p.side, room(r, lo) + me * own * D * 4, r.b.part + (lo + mine[0]) * D * 4, own * D * 4);
        },
    }
}

/// A split piece's glued rows (normed, or hidden after the last layer) to every rank on the side stream.
pub fn shareGlued(f: *const Forward, p: *const Pieces, lo: usize, n: usize, last: bool) !void {
    const D: usize = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    switch (f.exchange) {
        .local => for (f.ranks) |from| {
            const mine = share(n, world, from.w.share.rank);
            const bytes = (mine[1] - mine[0]) * D * 2;
            for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                const at = (lo + mine[0]) * D * 2;
                try f.copy(p.side, (if (last) to.b.hidden else to.b.normed) + at, (if (last) from.b.hidden else from.b.normed) + at, bytes);
            };
        },
        .nccl => |x| {
            const r = f.ranks[0];
            const me = r.w.share.rank;
            const rows_of = if (last) r.b.hidden else r.b.normed;
            const plan: relay.Relay = .{ .world = world, .rank = me, .n = n };
            if (f.ring) try relay.glued(x.lib, x.comm, p.side, plan, rows_of, lo, D) else try relay.splitGlued(x.lib, x.comm, p.side, plan, rows_of, lo, D);
        },
    }
}

/// DFlash2's taps after layer j: split glue posts a rank's own rows of x only, so it taps those (shareTaps sends them).
pub fn tapPieces(f: *const Forward, t: taps.Taps, j: usize, cut: []const [2]usize) !void {
    if (taps.slot(j) == null) return;
    const r = f.ranks[0];
    const world = r.w.share.world;
    const row = r.w.config.hc * r.w.config.hidden * 2;
    for (cut) |piece| {
        const lo = piece[0];
        const n = piece[1] - lo;
        if (f.splits(n) and f.exchange == .nccl) {
            const mine = share(n, world, r.w.share.rank);
            try t.at(f.stream, j, r.b.x, lo + mine[0], mine[1] - mine[0]);
            continue;
        }
        if (f.splits(n)) for (f.ranks) |from| if (from.w.share.rank != r.w.share.rank) {
            const mine = share(n, world, from.w.share.rank);
            try f.copy(f.stream, r.b.x + (lo + mine[0]) * row, from.b.x + (lo + mine[0]) * row, (mine[1] - mine[0]) * row);
        };
        try t.at(f.stream, j, r.b.x, lo, n);
    }
}

/// After a pieced pass: every split piece's own tap rows (tapPieces) to every rank, so each rank's taps hold every row.
pub fn shareTaps(f: *const Forward, t: taps.Taps, cut: []const [2]usize) !void {
    const x = switch (f.exchange) {
        .local => return, // tapPieces copied the owners' residual rows in
        .nccl => |x| x,
    };
    const r = f.ranks[0];
    const dims = taps.layers.len * taps.hidden;
    for (cut) |piece| {
        const n = piece[1] - piece[0];
        if (!f.splits(n)) continue;
        const plan: relay.Relay = .{ .world = r.w.share.world, .rank = r.w.share.rank, .n = n };
        if (f.ring) try relay.glued(x.lib, x.comm, f.stream, plan, t.out, piece[0], dims) else try relay.splitGlued(x.lib, x.comm, f.stream, plan, t.out, piece[0], dims);
    }
}
