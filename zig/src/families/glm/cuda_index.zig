//! A DSA indexer's scores and pools in CUDA (glm_index.cu): the bytes of tf_glm_scores and tf_glm_select_rows.

const std = @import("std");
const cuda = @import("cuda");

const sparse = @import("cuda_sparse.zig");

pub const topk = sparse.topk_pools;
/// glm_index_scores: rows a CTA (a warp each), 64-pool tiles a CTA, threads a CTA.
pub const score_rows = 8;
pub const score_tiles = 16;
pub const threads = score_rows * 32;
/// glm_index_pools' block: glm_index::T.
pub const pool_threads = 256;
/// The rows a launch must have to take it outside Mode.all: where tf_glm_scores' `loop` tiling starts.
pub const rows_from = sparse.score_loop_from;
pub const cap_max = 4096;

/// TF_GLM_INDEX_CUDA: unset or 0 (today's launches), 1 (both kernels, prompt chunks), scores (only the scores, to
/// bisect), all (both kernels, decode windows too).
pub const Mode = enum {
    off,
    both,
    scores,
    all,

    pub fn parse(text: ?[]const u8) !Mode {
        const t = text orelse return .off;
        if (t.len == 0 or std.mem.eql(u8, t, "0")) return .off;
        if (std.mem.eql(u8, t, "1")) return .both;
        if (std.mem.eql(u8, t, "scores")) return .scores;
        if (std.mem.eql(u8, t, "all")) return .all;
        return error.BadIndexMode;
    }

    pub fn read() !Mode {
        const v = std.c.getenv("TF_GLM_INDEX_CUDA") orelse return .off;
        return parse(std.mem.span(v));
    }
};

/// The 64-pool tiles a CTA scores: a run of score_tiles for prompt chunks, one for a decode window (its rows share
/// each tile, and one tile a CTA keeps every SM busy at short contexts).
pub fn tileLoop(rows: usize) usize {
    return if (rows >= rows_from) score_tiles else 1;
}

/// glm_index_scores' grid: rows in CTAs of 8, the `nb` pool blocks in runs of tileLoop(rows).
pub fn scoresGrid(rows: usize, nb: usize) [2]u32 {
    return .{ @intCast(cdiv(rows, score_rows)), @intCast(@max(1, cdiv(nb, tileLoop(rows)))) };
}

/// The candidates a row of the pools kernel keeps and the floor's target (speed only; the pools are the same).
pub const Tile = struct { cap: usize, target: usize };

pub fn tile(np: usize) Tile {
    if (np <= 12288) return .{ .cap = 1024, .target = 724 };
    if (np <= 32768) return .{ .cap = 2048, .target = 1024 };
    return .{ .cap = cap_max, .target = 2048 };
}

/// TF_GLM_OVN_DECODE_INDEX's split select (production's select_split): its block, and the buckets it starts at.
pub const split_threads = 128;
pub const split_from = 8192; // sparse.SPLIT_FROM: fewer pools select a row a CTA
pub const split_floor_rows = 8; // sparse.SEG_FLOOR_ROWS: the knob's 1, windows of more rows take the one-pass floor

/// sparse.split_chunk: the scores a chunk CTA takes (speed only).
pub fn splitChunk(np: usize) usize {
    return if (np <= 16384) 2048 else 4096;
}

/// sparse.split_chunks: chunk CTAs a row of np scores takes, a power of two (the chunk tables' row stride).
pub fn splitChunks(np: usize) usize {
    return std.math.ceilPowerOfTwoAssert(usize, @max(1, cdiv(np, splitChunk(np))));
}

/// sparse.split_scratch's bytes for `rows` rows of np scores: totals [rows][4][256], chunk histograms, chunk counts.
pub fn splitBytes(rows: usize, np: usize) usize {
    return rows * (1024 + splitChunks(np) * 257) * 4;
}

/// TF_GLM_OVN_DECODE_INDEX as the rows a window must stay under to take the split (unset or 0: off; 1: 8).
pub fn splitRows() !usize {
    return splitRowsOf(cuda.ovn.int("DECODE_INDEX", 0) catch return error.BadIndexMode);
}

fn splitRowsOf(v: u64) usize {
    return if (v == 1) split_floor_rows else @intCast(@min(v, rows_from));
}

pub const Kernels = struct {
    module: cuda.Module,
    scores_fn: cuda.Function,
    pools_fn: cuda.Function,
    /// Whether the pools kernel replaces tf_glm_select_rows (Mode.both, Mode.all).
    select: bool,
    /// Whether decode windows (fewer than rows_from rows) take the kernels too (Mode.all, TF_GLM_OVN_DECODE_INDEX).
    decode: bool,
    /// The pools a decode window needs before it takes them (TF_GLM_INDEX_DECODE_FROM, read at load; 0: any depth).
    decode_from: usize = 0,
    /// TF_GLM_OVN_DECODE_INDEX (read at load; 0: off): decode windows of fewer rows select by tf_glm_index_split.
    split_rows: usize = 0,
    split_fn: ?cuda.Function = null,
    /// TF_GLM_OVN_INDEX_SCHED (read at load): prompt-sized launches score with tf_glm_index_scores_sched.
    sched_fn: ?cuda.Function = null,

    /// `image`: cuda.kernels.glm_index.
    pub fn load(d: *const cuda.Driver, image: []const u8, mode: Mode) !Kernels {
        if (image.len == 0) return error.NoIndexKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const sr = try splitRows();
        return .{
            .module = m,
            .scores_fn = try m.function("tf_glm_index_scores"),
            .pools_fn = try m.function("tf_glm_index_pools"),
            .select = mode == .both or mode == .all,
            .decode = mode == .all or sr > 0,
            .decode_from = try decodeFrom(if (std.c.getenv("TF_GLM_INDEX_DECODE_FROM")) |v| std.mem.span(v) else null),
            .split_rows = sr,
            .split_fn = if (sr > 0) try m.function("tf_glm_index_split") else null,
            .sched_fn = if (cuda.ovn.on("INDEX_SCHED")) try m.function("tf_glm_index_scores_sched") else null,
        };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }

    /// Whether `rows` rows over `np` pools with this key cache layout go through the kernels.
    pub fn takes(k: *const Kernels, rows: usize, np: usize, pk: u64, ps: u64) bool {
        return (rows >= rows_from or (k.decode and np >= k.decode_from)) and np >= topk and pk % 16 == 0 and ps % 16 == 0;
    }

    /// Whether a decode window of `rows` rows the kernels take selects its pools with them (TF_GLM_OVN_DECODE_INDEX).
    pub fn decodeSelect(k: *const Kernels, rows: usize) bool {
        return k.split_rows > 0 and rows < rows_from;
    }

    /// Whether `rows` rows over `np` pools take tf_glm_index_split (fewer than split_rows rows, np past split_from).
    pub fn splits(k: *const Kernels, rows: usize, np: usize) bool {
        return k.split_fn != null and rows < k.split_rows and rows < rows_from and np >= split_from;
    }

    /// tf_glm_scores' launch for `rows` rows at `at` into out [rows][np]; the first `nb` 64-pool blocks are written.
    pub fn scores(k: *const Kernels, s: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, out: u64, at: u64, rows: usize, np: usize, nb: usize, scale: f32, wscale: f32, heads: usize, pt: u64, ps: u64) !void {
        const g = scoresGrid(rows, nb);
        var a: cuda.Args = .{};
        inline for (.{ qi, wts, @as(i64, @intCast(w_stride)), pk, out, at, @as(i64, @intCast(rows)), @as(i64, @intCast(np)), @as(i64, @intCast(nb)), scale, wscale, @as(i64, @intCast(heads)), @as(c_int, @intCast(tileLoop(rows))), pt, @as(i64, @intCast(ps)) }) |v| a.add(v);
        const f = if (k.sched_fn) |x| (if (rows >= rows_from) x else k.scores_fn) else k.scores_fn;
        try cuda.launch.launch(f, .{ .grid = .{ .x = g[0], .y = g[1] }, .block = .{ .x = threads } }, s, &a);
    }

    /// tf_glm_select_rows' launch: each row's 512 best pools of scores [rows][np] ascending into out [rows][512] int64.
    pub fn pools(k: *const Kernels, s: cuda.Stream, scores_ptr: u64, out: u64, at: u64, rows: usize, np: usize, path: u64) !void {
        const t = tile(np);
        var a: cuda.Args = .{};
        inline for (.{ scores_ptr, out, @as(i64, @intCast(np)), at, @as(c_int, topk), @as(c_int, @intCast(t.cap)), @as(c_int, @intCast(t.target)), path }) |v| a.add(v);
        try cuda.launch.launch(k.pools_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = pool_threads } }, s, &a);
    }

    /// tf_glm_index_split's five steps over scores [rows][np] at `at`: tokens [rows][sparse.width] and counts [rows]
    /// as tf_glm_select_rows then tf_glm_dsa_tokens write them; `scratch` holds splitBytes(rows, np).
    pub fn selectSplit(k: *const Kernels, s: cuda.Stream, scores_ptr: u64, at: u64, rows: usize, np: usize, scratch: u64, tokens: u64, counts: u64, pt: u64, page_rows: u64) !void {
        const f = k.split_fn orelse return error.NoIndexKernels;
        const chs = splitChunk(np);
        const cp = splitChunks(np);
        const hist = scratch + rows * 1024 * 4;
        const gt = hist + rows * cp * 256 * 4;
        try s.d.check(s.d.api.cuMemsetD32Async(scratch, 0, rows * 1024, s.handle), "cuMemsetD32Async");
        for (0..5) |step| {
            var a: cuda.Args = .{};
            inline for (.{ scores_ptr, @as(i64, @intCast(np)), at, @as(c_int, topk), @as(c_int, @intCast(chs)), @as(c_int, @intCast(cp)), @as(c_int, @intCast(step)), scratch, hist, gt, tokens, counts, pt, @as(i64, @intCast(page_rows)) }) |v| a.add(v);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(cp) }, .block = .{ .x = split_threads } }, s, &a);
        }
    }
};

/// TF_GLM_INDEX_DECODE_FROM, a context position, as the pools a decode window must reach (unset or 0: any depth).
pub fn decodeFrom(text: ?[]const u8) !usize {
    const t = text orelse return 0;
    if (t.len == 0) return 0;
    return (std.fmt.parseInt(usize, t, 10) catch return error.BadIndexMode) / sparse.pool;
}

test "GLM index: TF_GLM_INDEX_DECODE_FROM is a position, counted in pools; a typo is refused" {
    try std.testing.expectEqual(@as(usize, 0), try decodeFrom(null));
    try std.testing.expectEqual(@as(usize, 16384), try decodeFrom("65536"));
    try std.testing.expectError(error.BadIndexMode, decodeFrom("64k"));
}

test "GLM index: with a decode depth gate, shallow decode windows keep today's path and deep ones take the kernels" {
    var k: Kernels = undefined;
    k.decode = true;
    k.decode_from = 16384; // 65,536 positions
    try std.testing.expect(!k.takes(4, 16383, 0x1000, 32768));
    try std.testing.expect(k.takes(4, 16384, 0x1000, 32768));
    try std.testing.expect(k.takes(rows_from, 1024, 0x1000, 32768)); // prompt chunks are not gated
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "GLM index: TF_GLM_INDEX_CUDA is off unless 1, scores or all, and a typo is refused" {
    try std.testing.expectEqual(Mode.off, try Mode.parse(null));
    try std.testing.expectEqual(Mode.off, try Mode.parse(""));
    try std.testing.expectEqual(Mode.off, try Mode.parse("0"));
    try std.testing.expectEqual(Mode.both, try Mode.parse("1"));
    try std.testing.expectEqual(Mode.scores, try Mode.parse("scores"));
    try std.testing.expectEqual(Mode.all, try Mode.parse("all"));
    try std.testing.expectError(error.BadIndexMode, Mode.parse("on"));
}

test "GLM index: the scores grid covers 8 rows a CTA, 16 pool blocks a prompt CTA and one a decode CTA" {
    // a 512-row chunk over 147,456 pools: 2,304 blocks
    try std.testing.expectEqual([2]u32{ 64, 144 }, scoresGrid(512, 2304));
    try std.testing.expectEqual([2]u32{ 9, 1 }, scoresGrid(65, 16));
    try std.testing.expectEqual([2]u32{ 8, 2 }, scoresGrid(64, 17));
    // a 4-row decode window over 596k tokens' 149,000 pools: one CTA a 64-pool block
    try std.testing.expectEqual([2]u32{ 1, 2329 }, scoresGrid(4, 2329));
    try std.testing.expectEqual([2]u32{ 2, 32 }, scoresGrid(9, 32));
    try std.testing.expectEqual([2]u32{ 1, 1 }, scoresGrid(1, 0));
}

test "GLM index: the candidate cap follows the pool count and stays in shared memory" {
    try std.testing.expectEqual(Tile{ .cap = 1024, .target = 724 }, tile(12288));
    try std.testing.expectEqual(Tile{ .cap = 2048, .target = 1024 }, tile(12289));
    try std.testing.expectEqual(Tile{ .cap = 2048, .target = 1024 }, tile(32768));
    try std.testing.expectEqual(Tile{ .cap = 4096, .target = 2048 }, tile(32769));
    try std.testing.expect(tile(1 << 30).cap <= cap_max);
}

test "GLM index: TF_GLM_OVN_DECODE_INDEX's split takes production's chunks and its 1 means 8 rows" {
    try std.testing.expectEqual(@as(usize, 2048), splitChunk(16384));
    try std.testing.expectEqual(@as(usize, 4096), splitChunk(16385));
    try std.testing.expectEqual(@as(usize, 4), splitChunks(8192));
    try std.testing.expectEqual(@as(usize, 8), splitChunks(16385)); // 5 chunks of 4,096, rounded up
    try std.testing.expectEqual(@as(usize, 64), splitChunks(262144)); // 1M tokens' pools
    try std.testing.expectEqual(@as(usize, 3 * (1024 + 64 * 257) * 4), splitBytes(3, 262144));
    try std.testing.expectEqual(@as(usize, 0), splitRowsOf(0));
    try std.testing.expectEqual(@as(usize, split_floor_rows), splitRowsOf(1));
    try std.testing.expectEqual(@as(usize, 4), splitRowsOf(4));
    try std.testing.expectEqual(@as(usize, rows_from), splitRowsOf(1 << 20));
    var k: Kernels = undefined;
    k.split_rows = 8;
    k.split_fn = null;
    try std.testing.expect(!k.splits(4, 1 << 16)); // no kernel loaded
    k.split_fn = @as(cuda.Function, undefined);
    try std.testing.expect(k.splits(7, split_from));
    try std.testing.expect(!k.splits(8, split_from));
    try std.testing.expect(!k.splits(7, split_from - 1));
    try std.testing.expect(k.decodeSelect(rows_from - 1) and !k.decodeSelect(rows_from));
}

test "GLM index: prompt-sized launches on aligned key caches take the kernels, decode windows only with Mode.all" {
    var k: Kernels = undefined;
    k.decode = false;
    k.decode_from = 0;
    try std.testing.expect(k.takes(rows_from, 1024, 0x1000, 32768));
    try std.testing.expect(!k.takes(rows_from - 1, 1024, 0x1000, 32768));
    try std.testing.expect(!k.takes(4, 1 << 18, 0x1000, 32768));
    try std.testing.expect(!k.takes(512, 1024, 0x1008, 32768));
    try std.testing.expect(!k.takes(512, 1024, 0x1000, 32760));
    try std.testing.expect(!k.takes(512, topk - 1, 0x1000, 32768));
    k.decode = true;
    try std.testing.expect(k.takes(1, 1024, 0x1000, 32768));
    try std.testing.expect(k.takes(rows_from - 1, 1 << 18, 0x1000, 32768));
    try std.testing.expect(!k.takes(4, topk - 1, 0x1000, 32768));
    try std.testing.expect(!k.takes(4, 1024, 0x1008, 32768));
}
