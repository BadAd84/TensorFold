//! TF_GLM_OVN_DECODE_SHARD: a decode window's DSA indexer pools in contiguous shares across the ranks (production's
//! decode pool shard, ots0107 + ots0112): each rank scores and picks its share, one gather, every rank the merge.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const prof = @import("cuda_prof.zig");

/// Pools a share is made of: a page of pooled keys (glm_pages::at<128>), so a share's keys start on a page.
pub const unit = 128;
/// glm_glue_shard.cuh's blocks: a pick's threads (a row a block), a merge's.
pub const pick_threads = 1024;
pub const merge_threads = 256;

/// A decode window's exchange: this rank of `world` over NCCL (the group's gathers when the forward has one).
pub const Shard = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    group: ?*const cuda.tp_group.Group = null,
    rank: usize,
    world: usize,
    from: usize, // TF_GLM_OVN_DECODE_SHARD: the position windows shard from
};

/// TF_GLM_OVN_DECODE_SHARD: the context position decode windows shard from (unset or 0: never); a typo is refused.
pub fn from() !usize {
    return @intCast(try cuda.ovn.int("DECODE_SHARD", 0));
}

/// The forward's shard for a window on this rank: null with the knob off, for prompt rows and for in-process ranks.
pub fn of(at: usize, prefill: bool, exchange: anytype, rank: usize, world: usize) ?Shard {
    if (at == 0 or prefill) return null;
    return switch (exchange) {
        .local => null,
        .nccl => |x| .{ .lib = x.lib, .comm = x.comm, .group = x.group, .rank = rank, .world = world, .from = at },
    };
}

/// Whether a window of `rows` rows at `pos` shards: from `x.from` on, over two ranks or more (the same on every rank).
pub fn takes(x: Shard, pos: usize, rows: usize) bool {
    return x.from > 0 and x.world > 1 and rows > 0 and pos >= x.from;
}

/// A rank's share of a window's `cols` scored pools: [off, off + ext), whole units, contiguous in rank order.
pub const Part = struct { off: usize, ext: usize };

pub fn part(cols: usize, rank: usize, world: usize) Part {
    const each = cdiv(cdiv(cols, unit), world) * unit;
    const off = @min(cols, rank * each);
    return .{ .off = off, .ext = @min(each, cols - off) };
}

/// The gather's bytes a rank: `rows` rows of K order keys then K pools, int32.
pub fn pairBytes(rows: usize) usize {
    return rows * sparse.topk_pools * 2 * 4;
}

/// Scratch.ls bytes the gather needs on a window's buffers of `rows` rows (0 with the knob off or on prompt buffers).
pub fn listBytes(rows: usize, prefill: bool, world: usize) !usize {
    if (prefill or world < 2 or (try from()) == 0) return 0;
    return world * pairBytes(rows);
}

/// sparse.selectTokens over the ranks: this rank's share, the pairs gathered in place at s.ls, then the merge's tokens
/// and counts, the bytes selectTokens writes.
pub fn select(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, x: Shard, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, np_max_cap: usize, s: sparse.Scratch, tokens: u64, counts: u64, profile: ?*prof.Prof) !void {
    const bytes = pairBytes(rows);
    const mine = s.ls + x.rank * bytes;
    try share(set, ops, stream, x.rank, x.world, qi, wts, w_stride, pk, t, pos, host_pos, rows, heads, d, np_max_cap, s, mine, profile);
    const stamp = prof.span(profile, stream, .index_select);
    defer stamp.end();
    if (x.group) |g| try g.allGather(stream, mine, s.ls, bytes) else try x.lib.check(x.lib.api.ncclAllGather(mine, s.ls, bytes, .u8, x.comm, stream.handle), "ncclAllGather");
    try merge(set, stream, s.ls, rows, x.world, pos, tokens, counts, t);
}

/// What rank `rank` runs before the gather: its share's scores (s.scores at their columns), then its pairs at `out`.
pub fn share(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, rank: usize, world: usize, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, np_max_cap: usize, s: sparse.Scratch, out: u64, profile: ?*prof.Prof) !void {
    if (d != 128 or heads > 32) return error.IndexShape;
    const np = sparse.poolBucket(host_pos, rows, np_max_cap);
    const p = part(sparse.scoreColumns(host_pos, rows, np), rank, world);
    var stamp = prof.span(profile, stream, .index_score);
    if (p.ext > 0) try score(set, ops, stream, qi, wts, w_stride, pk, t, pos, rows, heads, d, np, p, s);
    stamp.end();
    stamp = prof.span(profile, stream, .index_select);
    defer stamp.end();
    try pick(set, stream, s.scores, np, p, pos, rows, out);
}

/// The share scored as a window of its own: its pool 0 is pool off (position 4 off rows earlier, keys from page
/// off / unit on), scores stored from column off: each score is the unsharded one (its row's query, its pool's key).
fn score(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, rows: usize, heads: usize, d: usize, np: usize, p: Part, s: sparse.Scratch) !void {
    var at = pos; // the share's position: off pools earlier (rank 0's is the window's)
    if (p.off > 0) {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(-@as(c_int, @intCast(p.off * sparse.pool)));
        a.add(s.at);
        try cuda.launch.launch(try ops.function("tf_glm_add_i32"), .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &a);
        at = s.at;
    }
    const keys = if (t.pt != 0) pk else pk + p.off * d * 2;
    const pt = if (t.pt != 0) t.pt + p.off / unit * 4 else 0;
    const out = s.scores + p.off * 4;
    const nb = cdiv(p.ext, 64);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d), -0.5));
    // the scorer selectTokens takes for this window (both write the same bytes, glm-index-check)
    const quick = if (set.index) |k| if (k.takes(rows, np, pk, t.ps)) k else null else null;
    if (quick) |k| return k.scores(stream, qi, wts, w_stride, keys, out, at, rows, np, nb, scale, sparse.headScale(heads), heads, pt, t.ps);
    const loop: usize = if (rows >= sparse.score_loop_from) sparse.score_loop else 1;
    var sa: cuda.Args = .{};
    inline for (.{ qi, wts, @as(i64, @intCast(w_stride)), keys, out, at, @as(i64, @intCast(np)), @as(i64, @intCast(nb)), scale, sparse.headScale(heads), @as(i64, @intCast(heads)), @as(c_int, @intCast(loop)), pt, @as(i64, @intCast(t.ps)) }) |v| sa.add(v);
    try cuda.launch.launch(set.glue.scores, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(cdiv(nb, loop)) }, .block = .{ .x = 128 } }, stream, &sa);
}

/// tf_glm_index_shard_pick: each row's K best pools of share `p` of scores [rows][np] as pairs at `out` [rows][2 K].
pub fn pick(set: glue.Set, stream: cuda.Stream, scores: u64, np: usize, p: Part, pos: u64, rows: usize, out: u64) !void {
    var a: cuda.Args = .{};
    inline for (.{ scores, @as(i64, @intCast(np)), pos, @as(i64, @intCast(p.off)), @as(i64, @intCast(p.ext)), @as(c_int, sparse.topk_pools), out }) |v| a.add(v);
    try cuda.launch.launch(set.glue.index_shard_pick, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = pick_threads } }, stream, &a);
}

/// tf_glm_index_shard_merge: `world` ranks' pairs at `recv` (rank blocks of `rows` rows) to tokens and counts.
pub fn merge(set: glue.Set, stream: cuda.Stream, recv: u64, rows: usize, world: usize, pos: u64, tokens: u64, counts: u64, t: latent.Table) !void {
    var a: cuda.Args = .{};
    inline for (.{ recv, @as(i64, @intCast(rows)), @as(c_int, @intCast(world)), pos, @as(c_int, sparse.topk_pools), tokens, counts, t.pt, @as(i64, @intCast(t.page_rows)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.index_shard_merge, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = merge_threads } }, stream, &a);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "TF_GLM_OVN_DECODE_SHARD: contiguous whole-page shares cover a window's pools, the last one short" {
    // ~596k tokens: 149,000 pools in 1,165 units, 389 a rank
    try std.testing.expectEqual(Part{ .off = 0, .ext = 49792 }, part(149000, 0, 3));
    try std.testing.expectEqual(Part{ .off = 49792, .ext = 49792 }, part(149000, 1, 3));
    try std.testing.expectEqual(Part{ .off = 99584, .ext = 49416 }, part(149000, 2, 3));
    // 512 pools over three ranks: the third holds none, and its share is empty at the end
    try std.testing.expectEqual(Part{ .off = 256, .ext = 256 }, part(512, 1, 3));
    try std.testing.expectEqual(Part{ .off = 512, .ext = 0 }, part(512, 2, 3));
    try std.testing.expectEqual(Part{ .off = 0, .ext = 1000 }, part(1000, 0, 1));
    var covered: usize = 0;
    for (0..4) |r| {
        const p = part(70001, r, 4);
        try std.testing.expectEqual(covered, p.off);
        try std.testing.expect(p.off % unit == 0 or p.ext == 0);
        covered += p.ext;
    }
    try std.testing.expectEqual(@as(usize, 70001), covered);
    try std.testing.expectEqual(@as(usize, 2 * 4096), pairBytes(2));
}

test "TF_GLM_OVN_DECODE_SHARD: every rank shards the same windows, from the knob's position on, two ranks or more" {
    var x: Shard = .{ .lib = undefined, .comm = null, .rank = 2, .world = 3, .from = 131072 };
    try std.testing.expect(!takes(x, 131071, 2));
    try std.testing.expect(takes(x, 131072, 2));
    try std.testing.expect(!takes(x, 600000, 0));
    x.rank = 0; // the rank plays no part
    try std.testing.expect(takes(x, 131072, 2));
    x.world = 1;
    try std.testing.expect(!takes(x, 600000, 2));
    const Ex = union(enum) { local, nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*const cuda.tp_group.Group = null } };
    try std.testing.expect(of(131072, false, @as(Ex, .local), 0, 3) == null);
    const n: Ex = .{ .nccl = .{ .lib = undefined, .comm = null } };
    try std.testing.expect(of(0, false, n, 0, 3) == null); // the knob off
    try std.testing.expect(of(131072, true, n, 0, 3) == null); // a prompt chunk
    try std.testing.expectEqual(@as(usize, 131072), of(131072, false, n, 1, 3).?.from);
}
