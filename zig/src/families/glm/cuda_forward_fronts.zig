//! TF_GLM_OVN_PEER_FIRST and TF_GLM_OVN_FRONTS: a pieced prompt pass's fills and next-block fronts by row ranges.
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const prof = @import("cuda_prof.zig");
const b16 = @import("cuda_b16.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const weights = @import("cuda_weights.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const Rank = fw.Rank;
const Site = fw.Forward.Site;
const share = fw.Forward.share;

/// The fewest rows a reordered call takes: b16 (rows_from) and the FP8 lane (fused_rows) pick kernels by rows below it.
fn least(knob: usize) usize {
    return @max(knob, b16.rows_from, cuda.qmmf.fused_rows);
}

/// PEER_FIRST for a split piece of `n` rows: on, and every rank's share of it at least the knob's rows.
pub fn peerFirst(f: *const Forward, p: *const Pieces, n: usize) bool {
    return p.peer_first != 0 and n / f.ranks[0].w.share.world >= least(p.peer_first);
}

/// FRONTS for a pieced pass: on, prompt rows, no shared window, and every front call of at least the knob's rows.
pub fn wanted(f: *const Forward, p: *const Pieces, cut: []const [2]usize) bool {
    if (p.fronts == 0) return false;
    for (f.ranks) |r| if (!r.b.prefill or r.segs.len > 0) return false;
    const world = f.ranks[0].w.share.world;
    for (cut) |piece| {
        const n = piece[1] - piece[0];
        if ((if (f.splits(n)) n / world else n) < least(p.fronts)) return false;
    }
    return true;
}

/// After `site` of layer j, the next block's front on rank r's rows lo..hi: the FFN's, or layer j + 1's mixer's.
pub fn ahead(f: *const Forward, r: Rank, j: usize, site: Site, R: usize, lo: usize, hi: usize) !void {
    const b = r.b;
    const D = r.w.config.hidden;
    if (site == .mixer) switch (r.w.layers[j].ffn) {
        .dense => |m| {
            const stamp = prof.span(f.prof, f.stream, .ffn);
            defer stamp.end();
            try mlp.bodyRows(f.mlpKernels(), f.stream, m, b.dense, b.normed, D, lo, hi, b.prefill);
        },
        .moe => |m| {
            const stamp = prof.span(f.prof, f.stream, .shared_expert);
            defer stamp.end();
            try moe.sharedFront(f.moeKernels(), f.stream, m, b.moe, b.normed, D, lo, hi);
        },
    } else switch (r.w.layers[j + 1].mixer) {
        .kda => |k| {
            const stamp = prof.span(f.prof, f.stream, .projections);
            defer stamp.end();
            var s = b.kda;
            s.p = b.kproj; // a prompt chunk's projection rows, as Forward.mixerFrom sets them
            try kda_block.projectRows(f.set(), f.stream, k, s, b.normed, D, lo, hi);
        },
        .mla => |a| try dsa_block.prepareRows(f.set(), f.stream, a, b.dsa, b.normed, D, lo, hi, dsa_block.sparseRows(a, r.st.pos, R), f.prof),
    }
}

/// ahead on every other rank's rows of `piece` once they are shared in: those before r's share and those after it.
pub fn peers(f: *const Forward, r: Rank, j: usize, site: Site, R: usize, piece: [2]usize) !void {
    const mine = share(piece[1] - piece[0], r.w.share.world, r.w.share.rank);
    if (mine[0] > 0) try ahead(f, r, j, site, R, piece[0], piece[0] + mine[0]);
    if (piece[0] + mine[1] < piece[1]) try ahead(f, r, j, site, R, piece[0] + mine[1], piece[1]);
}

/// Forward.ffn after ahead ran on every row: the MoE's router and routed experts (a dense layer's front is its body).
pub fn ffnRest(f: *const Forward, r: Rank, l: weights.Layer, R: usize) !void {
    const stamp = prof.span(f.prof, f.stream, .ffn);
    defer stamp.end();
    switch (l.ffn) {
        .dense => {},
        .moe => |m| try moe.body(f.moeKernels(), f.stream, m, r.b.moe, r.b.normed, r.w.config.hidden, R, r.b.prefill),
    }
}

/// Forward.fillRows; `fronted` at an MoE FFN: its shared expert already ran (ahead), so only moe.fillTail is left.
pub fn fill(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize, out: u64, fronted: bool) !void {
    if (!fronted or site != .ffn or l.ffn != .moe) return f.fillRows(r, l, site, lo, hi, out);
    const stamp = prof.span(f.prof, f.stream, .output_projection);
    defer stamp.end();
    const inner = prof.span(f.prof, f.stream, .shared_expert);
    defer inner.end();
    try moe.fillTail(f.moeKernels(), f.stream, l.ffn.moe, r.b.moe, lo, hi, r.b.prefill, out);
}
