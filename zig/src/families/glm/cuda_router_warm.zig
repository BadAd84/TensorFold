//! TF_GLM_OVN_ROUTER_WARM: warm the router kernels (at 2, the prompt-path mixer too) at boot, off the first request.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const moe = @import("cuda_moe.zig");
const latent = @import("cuda_latent.zig");
const weights = @import("cuda_weights.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// The throwaway state's tokens: enough for every row class's position-0 dense window (cuda_cost_time uses the same).
const warm_capacity = 1024;

/// P1ShortTTFT-1's router_warm_rows: {2} plus each router block's first row and the block itself (patch lines 24-28).
pub const row_classes = [_]u32{ 1, 2, 16, 17, 32, 33, 64, 65, 128 };

/// The prefix of row_classes that fits `rows` buffer rows (production breaks when R > b.normed.shape[0]).
pub fn classes(rows: usize) []const u32 {
    var n: usize = 0;
    while (n < row_classes.len and row_classes[n] <= rows) n += 1;
    return row_classes[0..n];
}

/// TF_GLM_OVN_ROUTER_WARM: 0 off (default), 1 the router warm, 2 the router warm plus the prompt-path mixer warm.
pub const Level = enum(u8) { off = 0, router = 1, all = 2 };

/// The knob's value; any other integer or non-integer is refused (as the engine refuses other OVN typos).
pub fn level() error{BadOvnKnob}!Level {
    return switch (try cuda.ovn.int("ROUTER_WARM", 0)) {
        0 => .off,
        1 => .router,
        2 => .all,
        else => error.BadOvnKnob,
    };
}

/// The warm's layers, each the first of its kind (as production's warm_routers takes the first MoE and DSA layer).
pub const Picked = struct {
    moe: ?moe.Layer = null,
    index: ?dsa_block.Index = null,
    kda_layer: ?weights.Layer = null,
    mla_layer: ?weights.Layer = null,
};

pub fn pickLayers(layers: []const weights.Layer) Picked {
    var p: Picked = .{};
    for (layers) |l| {
        switch (l.ffn) {
            .moe => |m| {
                if (p.moe == null) p.moe = m;
            },
            .dense => {},
        }
        switch (l.mixer) {
            .kda => {
                if (p.kda_layer == null) p.kda_layer = l;
            },
            .mla => |a| {
                if (p.mla_layer == null) {
                    p.mla_layer = l;
                    p.index = a.index;
                }
            },
        }
    }
    return p;
}

/// The indexer gate as a router call (production's index_gate = glue.router(normed, ix.gate, igr)); ne is the d=128.
pub const IndexGate = struct { gate: u64, igr: u64, router_part: u64, ne: usize };

fn indexGate(index: ?dsa_block.Index, dsa: dsa_block.Scratch) ?IndexGate {
    const ix = index orelse return null;
    return .{ .gate = ix.gate, .igr = dsa.igr, .router_part = dsa.router_part, .ne = ix.d };
}

/// P1ShortTTFT-1: the MoE route and the indexer gate, once at each fitting row class, on the MoE and DSA scratch.
pub fn routerWarm(mk: moe.Kernels, set: glue.Set, stream: cuda.Stream, moe_layer: ?moe.Layer, moe_scratch: moe.Scratch, gate: ?IndexGate, normed: u64, hidden: usize, rows: usize) !usize {
    var n: usize = 0;
    for (classes(rows)) |r| {
        const R: usize = r;
        if (moe_layer) |l| {
            try moe.route(mk, stream, l, moe_scratch, normed, hidden, R);
            n += 1;
        }
        if (gate) |g| {
            try glue.router(set, stream, normed, hidden, g.gate, g.igr, g.router_part, R, hidden, g.ne);
            n += 1;
        }
    }
    return n;
}

/// WarmAll-1: the hyper-connection glue and the first KDA and DSA mixer blocks, at each row class, on a fresh state.
fn mixerWarm(e: *Engine, f: *fw.Forward, tst: *fw.State, p: Picked, hidden: usize) !usize {
    const r = f.ranks[0];
    const rank = e.w.share.rank;
    var n: usize = 0;
    for (classes(e.bp.rows)) |row| {
        const R: usize = row;
        try tst.reset(e.stream); // a fresh sequence at position 0
        try tst.ready(e.stream, R);
        const nch = @min(latent.chunksFor(R), e.bp.nch);
        const out = e.bp.gathered + rank * R * hidden * 4;
        if (p.kda_layer) |l| {
            try f.pre(r, l.attn_hc, l.in_norm, R);
            try f.mixer(r, l, 0, 0, R, nch, out);
            n += 2;
        }
        if (p.mla_layer) |l| {
            try f.pre(r, l.attn_hc, l.in_norm, R);
            try f.mixer(r, l, 0, 0, R, nch, out);
            n += 2;
        }
        try f.post(r, e.bp.x, R, 0);
        n += 1;
    }
    return n;
}

/// The warm's single-rank in-process Forward on the prompt buffers: no collective, taps, prefetch, prof or pairing.
fn promptForward(e: *Engine, one: *[1]fw.Rank) fw.Forward {
    one.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = &e.bp }};
    return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = one, .exchange = .local, .split = e.settings.split, .ring = e.settings.ring, .split_min = e.settings.split_min, .kda_wide = e.settings.kda_wide };
}

/// The boot entry (every rank, after the weights load, before the server accepts requests). Returns the launches.
pub fn warm(e: *Engine) !usize {
    const lvl = try level();
    if (lvl == .off) return 0;
    const t0 = std.Io.Clock.awake.now(e.io).toNanoseconds();
    const hidden = e.c.hidden;
    const cap = e.bp.rows;
    const picked = pickLayers(e.w.layers);
    var one: [1]fw.Rank = undefined;
    var f = promptForward(e, &one);
    var n = try routerWarm(f.moeKernels(), f.set(), e.stream, picked.moe, e.bp.moe, indexGate(picked.index, e.bp.dsa), e.bp.normed, hidden, cap);
    switch (lvl) {
        .off => unreachable,
        .router => try e.stream.synchronize(),
        .all => {
            var tst = try fw.State.initFormat(e.gpa, e.d, &e.w, warm_capacity, e.settings.kv8);
            defer tst.deinit();
            errdefer e.stream.synchronize() catch {}; // drain before the throwaway state's buffer is freed
            one[0] = .{ .w = &e.w, .st = &tst, .b = &e.bp };
            n += try mixerWarm(e, &f, &tst, picked, hidden);
            try e.stream.synchronize();
        },
    }
    if (e.share.rank == 0) {
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(e.io).toNanoseconds() - t0)) / 1e6;
        std.log.info("glm: TF_GLM_OVN_ROUTER_WARM={d}: {d} warm launches, {d:.1} ms (first-use loads off the first request)", .{ @backingInt(lvl), n, ms });
    }
    return n;
}
