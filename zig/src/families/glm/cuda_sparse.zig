//! GLM-5.3-Flash's DSA indexer on CUDA (the Python family's sparse): each row keeps its best 512 4-token pools.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");

/// sparse.POOL, TOPK_POOLS, SELECT_LIST, SCORE_LOOP_FROM, SCORE_LOOP; geometry.MLA_SELECT_ROWS.
pub const pool = 4;
pub const topk_pools = 512;
pub const select_list = 1024;
pub const score_loop_from = 64;
pub const score_loop = 32;
pub const select_rows = 512;
/// A row's token list: the 512 pools' tokens, then the incomplete pool's up to 3.
pub const width = topk_pools * pool + pool - 1;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// TF_GLM_OVN_PK8 (TRADE), set once at engine init (as cuda_state's drop_mtp): pooled keys as kv8 rows (codes, scale).
pub var pk8 = false;

/// A pooled key's bytes: bf16 values, or with pk8 its `d` E4M3 codes and FP32 scale (glm_pk8_row.cuh's ROW).
pub fn pooledRow(d: usize) usize {
    return if (pk8) d + 4 else d * 2;
}

/// tf_glm_scores' function for the pooled keys' format.
pub fn scoresFn(set: glue.Set) cuda.Function {
    return if (pk8) set.glue.scores_pk8 else set.glue.scores;
}

/// At most wide_rows rows (one wave on 48 SMs) over wide_pools pools or more: 1,024 threads a row, the same bytes.
const wide_rows = 48;
const wide_pools = 4096;

pub fn wideSelect(rows: usize, np: usize) bool {
    return rows <= wide_rows and np >= wide_pools;
}

/// sparse.index_update: the window's LayerNormed index keys and gates into ik and ig at pos, completed pools into pk.
pub fn indexUpdate(set: glue.Set, stream: cuda.Stream, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, t: latent.Table, pos: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ k_raw, @as(i64, @intCast(k_stride)), gate, ln_w, ln_b, ik, ig, pos, @as(f32, 1e-6) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.index_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
    var b: cuda.Args = .{};
    inline for (.{ ik, ig, ape, pk, pos, @as(i64, @intCast(rows)), t.pt, @as(i64, @intCast(t.ps)) }) |v| b.add(v);
    try cuda.launch.launch(if (pk8) set.glue.pool_keys_pk8 else set.glue.pool_keys, .{ .grid = .{ .x = @intCast(rows / 4 + 2) }, .block = .{ .x = 32 } }, stream, &b);
}

/// sparse.pool_bucket: visible pools for rows pos .. pos + rows - 1, a power of two (at least 1024), at most `np_max`.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    return @min(np_max, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
}

/// sparse.score_columns: the pool columns selection reads for those rows (the last row's complete pools, at least 512).
pub fn scoreColumns(pos: usize, rows: usize, np_max: usize) usize {
    return @min(np_max, @max(topk_pools, (pos + rows) / pool));
}

/// select_tokens' 1/sqrt(heads) head-weight scale (32 heads: 1 / 5.656854249492381, as the Python family writes it).
pub fn headScale(heads: usize) f32 {
    if (heads == 32) return @floatCast(1.0 / 5.656854249492381);
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(heads))));
}

/// The scratch of one selectTokens: scores fp32 [min(rows, 512)][np], and with np past select_list the list rows
/// (ls, li; ls at least ovnListBytes: TF_GLM_OVN_DECODE_INDEX's split tables, TF_GLM_OVN_POOLS32's int32 pools,
/// TF_GLM_OVN_DECODE_SHARD's gathered pairs).
pub const Scratch = struct { scores: u64, ls: u64, li: u64, pools: u64, at: u64 };

/// The bytes the TF_GLM_OVN_ knobs need at Scratch.ls (0 with them off): `seg` the most rows one selectTokens
/// takes, np_max the most pools a row scores; a prefill buffer's `rows` gather world - 1 rows more (Split.int32).
pub fn ovnListBytes(rows: usize, seg: usize, prefill: bool, world: usize, np_max: usize) !usize {
    const index = @import("cuda_index.zig");
    const sr = try index.splitRows();
    const tables = if (sr > 0) index.splitBytes(@min(seg, sr - 1), np_max) else 0;
    const pools32 = if (prefill and cuda.ovn.on("POOLS32")) (rows + world) * topk_pools * 4 else 0;
    return @max(tables, pools32, try @import("cuda_index_shard.zig").listBytes(rows, prefill, world));
}

/// A prompt chunk's indexer rows split across the ranks (TF_GLM_INDEX_SPLIT): a block each, the picks all-gathered;
/// `int32` (TF_GLM_OVN_POOLS32, read at load) gathers them as int32 at Scratch.ls.
pub const Split = struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, int32: bool = false };

/// Whether rank 0 splits a prompt chunk's indexer rows at `start`: from `from` on (0: never), over two ranks or more.
pub fn splitAt(start: usize, from: usize, world: usize) bool {
    return from > 0 and world > 1 and start >= from;
}

/// sparse.select_tokens: rows' tokens (i32 [rows][width], ascending, -1 padded; on pages their rows); count 0 if dense.
pub fn selectTokens(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, np_max_cap: usize, s: Scratch, tokens: u64, counts: u64) !void {
    return selectTokensProfile(set, ops, stream, qi, wts, w_stride, pk, t, pos, host_pos, rows, heads, d, np_max_cap, s, tokens, counts, null, null);
}

/// With `split`, this rank picks for its block of rows only (as one pass would, block by block) and gathers the rest.
pub fn selectTokensProfile(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, np_max_cap: usize, s: Scratch, tokens: u64, counts: u64, split: ?Split, profile: ?*@import("cuda_prof.zig").Prof) !void {
    const prof = @import("cuda_prof.zig");
    const np_max = poolBucket(host_pos, rows, np_max_cap);
    const b = @min(rows, select_rows);
    const loop: usize = if (rows >= score_loop_from) score_loop else 1;
    const add = try ops.function("tf_glm_add_i32");
    const per = if (split) |x| cdiv(rows, x.world) else rows;
    const lo = if (split) |x| @min(rows, x.rank * per) else 0;
    const hi = @min(rows, lo + per);
    var a: usize = lo;
    while (a < hi) : (a += b) {
        const n = @min(b, hi - a);
        var at = pos;
        if (a > 0) {
            var args: cuda.Args = .{};
            args.add(pos);
            args.add(@as(c_int, @intCast(a)));
            args.add(s.at);
            try cuda.launch.launch(add, .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &args);
            at = s.at;
        }
        const nb = cdiv(scoreColumns(host_pos + a, n, np_max), 64);
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d), -0.5));
        var sa: cuda.Args = .{};
        inline for (.{ qi + a * heads * d * 2, wts + a * w_stride * 2, @as(i64, @intCast(w_stride)), pk, s.scores, at, @as(i64, @intCast(np_max)), @as(i64, @intCast(nb)), scale, headScale(heads), @as(i64, @intCast(heads)), @as(c_int, @intCast(loop)), t.pt, @as(i64, @intCast(t.ps)) }) |v| sa.add(v);
        if (d != 128 or heads > 32) return error.IndexShape;
        const quick = if (set.index) |k| if (k.takes(rows, np_max, pk, t.ps)) k else null else null;
        var stamp = prof.span(profile, stream, .index_score);
        if (quick) |k| try k.scores(stream, qi + a * heads * d * 2, wts + a * w_stride * 2, w_stride, pk, s.scores, at, n, np_max, nb, scale, headScale(heads), heads, t.pt, t.ps) else try cuda.launch.launch(scoresFn(set), .{ .grid = .{ .x = @intCast(n), .y = @intCast(cdiv(nb, loop)) }, .block = .{ .x = 128 } }, stream, &sa);
        stamp.end();
        stamp = prof.span(profile, stream, .index_select);
        if (quick) |k| {
            if (split == null and k.splits(rows, np_max)) { // TF_GLM_OVN_DECODE_INDEX: the window's tokens directly
                try k.selectSplit(stream, s.scores, at, n, np_max, s.ls, tokens, counts, t.pt, t.page_rows);
                stamp.end();
                return;
            }
            if (k.select or k.decodeSelect(rows)) {
                try k.pools(stream, s.scores, s.pools + a * topk_pools * 8, at, n, np_max, 0);
                stamp.end();
                continue;
            }
        }
        var sel: cuda.Args = .{};
        inline for (.{ s.scores, s.pools + a * topk_pools * 8, @as(i64, @intCast(np_max)), at, @as(c_int, topk_pools) }) |v| sel.add(v);
        const wide = wideSelect(n, np_max);
        try cuda.launch.launch(if (wide) set.glue.select_rows_wide else set.glue.select_rows, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = if (wide) 1024 else 128 } }, stream, &sel);
        stamp.end();
    }
    const stamp = prof.span(profile, stream, .index_select);
    defer stamp.end();
    var picks = s.pools; // the pools the tokens are read from: int64, or the int32 ones gathered at s.ls
    var assemble = try ops.function("tf_glm_dsa_tokens");
    if (split) |x| { // in place: each rank's equal block (the prefill buffers' pools hold world rows more), rank order
        if (x.int32) { // this rank's block packed to int32 at its place in s.ls, gathered there: half the bytes
            const n32 = per * topk_pools;
            var pa: cuda.Args = .{};
            inline for (.{ s.pools + x.rank * n32 * 8, s.ls + x.rank * n32 * 4, @as(i64, @intCast(n32)) }) |v| pa.add(v);
            try cuda.launch.launch(try ops.function("tf_glm_pools_i32"), .{ .grid = .{ .x = @intCast(cdiv(n32, 256)) }, .block = .{ .x = 256 } }, stream, &pa);
            try x.lib.check(x.lib.api.ncclAllGather(s.ls + x.rank * n32 * 4, s.ls, n32 * 4, .u8, x.comm, stream.handle), "ncclAllGather");
            picks = s.ls;
            assemble = try ops.function("tf_glm_dsa_tokens_i32");
        } else {
            const bytes = per * topk_pools * 8;
            try x.lib.check(x.lib.api.ncclAllGather(s.pools + x.rank * bytes, s.pools, bytes, .u8, x.comm, stream.handle), "ncclAllGather");
        }
    }
    var args: cuda.Args = .{};
    args.add(picks);
    args.add(pos);
    args.add(@as(c_int, topk_pools));
    args.add(tokens);
    args.add(counts);
    args.add(t.pt);
    args.add(@as(i64, @intCast(t.page_rows)));
    try cuda.launch.launch(assemble, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
}

test "sparse.py's sizing rules" {
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(100, 1, 1 << 20));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(5000, 1, 1 << 20)); // 1251 visible pools
    try std.testing.expectEqual(@as(usize, 512), scoreColumns(100, 1, 1024));
    try std.testing.expectEqual(@as(usize, 1250), scoreColumns(5000, 1, 2048));
    try std.testing.expectEqual(@as(usize, 2051), width);
    try std.testing.expectEqual(@as(f32, @floatCast(1.0 / 5.656854249492381)), headScale(32));
}

test "rank 0 splits a prompt chunk's indexer rows from TF_GLM_INDEX_SPLIT's position on, over two ranks or more" {
    try std.testing.expect(!splitAt(65536, 0, 3)); // 0: never
    try std.testing.expect(splitAt(65536, 65536, 3));
    try std.testing.expect(!splitAt(63488, 65536, 3));
    try std.testing.expect(!splitAt(65536, 65536, 1)); // one rank holds every row
}
