//! GLM-5.3-Flash's MLA on the 512-wide latent cache (DSA layers): the Python family's latent on BF16, row by row.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

/// latent.CHUNK, KT, HB, HB_WIDE, PROMPT_ROWS, PROMPT_RB.
pub const chunk = 512;
pub const key_tile = 32;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
pub const prompt_rows = 64;
pub const prompt_rb = 64;

/// latent.head_block: 32 heads a program in prompt chunks (each key read once) and, from five rows, when a rank
/// carries 17-24 heads; 16 otherwise. The second rule is patch 0104: 0.3-0.5 ms a round at three Sparks' 22/21/21.
pub fn headBlock(rows: usize, heads: usize) usize {
    if (rows >= prompt_rows) return heads_tile_wide;
    return if (rows >= 5 and heads >= 17 and heads <= 24) heads_tile_wide else heads_tile;
}

/// latent.chunks_for: 512-key chunks to cover `length` keys.
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// latent.absorb_q: out [rows][heads][lw] bf16 = q [rows][heads][d] . wk [heads][d][lw], fp32 sums; sk: E4M3 kv_b's.
pub fn absorbQ(set: glue.Set, stream: cuda.Stream, q: u64, wk: u64, sk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
    if (d != 256 or lw != 512) return error.LatentShape;
    if (sk != 0) { // TF_GLM_OVN_KVB8: wk E4M3 [heads][d][lw], sk fp32 [heads][d][lw / 128]
        if (wk % 8 != 0) return error.LatentShape;
        return rowBlocks(set.glue.absorb_q8, stream, .{ q, wk, sk, out }, rows, heads, lw / 32);
    }
    const f = if (set.mla.vec and wk % 16 == 0) set.glue.absorb_q_vec else set.glue.absorb_q; // TF_GLM_OVN_ABSORB_VEC
    try rowBlocks(f, stream, .{ q, wk, out }, rows, heads, lw / 32);
}

/// The absorb and expand launches (pointers, then rows, heads, rb): rows in one block in decode, prompt_rb in prompts.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, ptrs: anytype, rows: usize, heads: usize, cols: usize) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (ptrs) |v| a.add(@as(u64, v));
    inline for (.{ @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v: out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw]; sv: E4M3 kv_b's.
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, sv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (dv != 256 or lw != 512) return error.LatentShape;
    if (sv != 0) { // TF_GLM_OVN_KVB8: wv E4M3 [heads][dv][lw], sv fp32 [heads][dv][lw / 128]
        if (wv % 8 != 0) return error.LatentShape;
        return rowBlocks(set.glue.expand_v8, stream, .{ ol, wv, sv, out }, rows, heads, dv / 16);
    }
    const vec = set.mla.vec and wv % 16 == 0 and ol % 16 == 0; // TF_GLM_OVN_ABSORB_VEC: kv_b and o by 16-byte loads
    try rowBlocks(if (vec) set.glue.expand_v_vec else set.glue.expand_v, stream, .{ ol, wv, out }, rows, heads, dv / 16);
}

/// A latent cache holds BF16 rows, or E4M3 rows followed by one FP32 scale per row.
pub fn cacheBytes(capacity: usize, lw: usize, kv8: bool) !usize {
    if (kv8 and lw != 512) return error.LatentShape;
    const row = if (kv8) try std.math.add(usize, lw, 4) else try std.math.mul(usize, lw, 2);
    return std.math.mul(usize, capacity, row);
}

/// On pages: the stream's table (i32 page ids) and the step from page to page in bytes and latent rows; 0: in order.
pub const Table = struct { pt: u64 = 0, ps: u64 = 0, page_rows: u64 = 0 };

/// BF16 rows through `t`, or E4M3 rows and their scales (scales nonzero): on pages each page part's codes, then scales.
pub fn latentWriteFormat(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, scales: u64, t: Table, pos: u64, rows: usize, lw: usize) !void {
    if (scales == 0) return latentWrite(set, stream, lat, lat_stride, cache, t, pos, rows, lw);
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, scales, pos, t.pt, @as(i64, @intCast(t.ps)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.lat_write_fp8, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, stream, &a);
}

/// latent.latent_write: rows of `lat` (stride `lat_stride`) into slots pos .. pos + rows - 1 (pos an i32 on device).
pub fn latentWrite(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, t: Table, pos: u64, rows: usize, lw: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, pos, @as(i64, @intCast(lw)), t.pt, @as(i64, @intCast(t.ps)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.lat_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The chunk partials of one attention call: po fp32 [nch][rows][heads][lw], pm and pl fp32 [nch][rows][heads].
pub const Partials = struct { po: u64, pm: u64, pl: u64 };

/// latent.attention: causal qa over the cache in `nch` 512-key chunks into out bf16; `dummy`: the merge's count i32.
pub fn attention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, t: Table, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    return attentionFormat(set, stream, qa, cache, 0, t, pos, p, out, dummy, rows, heads, lw, nch, scale, hb);
}

pub fn attentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    if (fuses(set.mla, rows, hb)) return fused(set, stream, qa, cache, scales, t, pos, 0, 0, out, rows, heads, lw, nch, 0, scale, false);
    try chunks(set, stream, qa, cache, scales, t, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (sparse false: keys through pos + r) and _sparse_chunks (row r's `counts` tokens of `tokens`).
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    const k = set.glue;
    const s = set.mla.skip; // TF_GLM_OVN_MLA_FUSEMERGE: the same partials from the tiles up to each chunk's last key
    const f = switch (hb) {
        heads_tile => if (scales == 0) (if (s) k.chunks_16_skip else k.chunks_16) else if (s) k.chunks_16_fp8_skip else k.chunks_16_fp8,
        heads_tile_wide => if (scales == 0) (if (s) k.chunks_32_skip else k.chunks_32) else if (s) k.chunks_32_fp8_skip else k.chunks_32_fp8,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (scales != 0) a.add(scales);
    inline for (.{ pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)), t.pt, @as(i64, @intCast(t.ps)) }) |v| a.add(v);
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

/// TF_GLM_OVN_MLA_FUSEMERGE: whether a launch of `rows` rows on `hb`-head blocks runs chunks and merge as one kernel.
fn fuses(m: glue.Mla, rows: usize, hb: usize) bool {
    return m.fuse_rows != 0 and rows >= m.fuse_rows and hb == heads_tile_wide;
}

/// chunks then merge as one launch (glm_glue_attn_ovn.cuh): a block a row and 32 heads walks the chunks, no partials.
fn fused(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, tokens: u64, counts: u64, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    const f = if (scales == 0) set.glue.chunks_32_fused else set.glue.chunks_32_fp8_fused;
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (scales != 0) a.add(scales);
    inline for (.{ pos, tokens, counts, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)), t.pt, @as(i64, @intCast(t.ps)), @as(c_int, @intCast(nch)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, heads_tile_wide) }, .block = .{ .x = 256 }, .shared = glue.chunksSmem(heads_tile_wide) }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, t: Table, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    return sparseAttentionFormat(set, stream, qa, cache, 0, t, tokens, counts, w, p, out, rows, heads, lw, scale);
}

pub fn sparseAttentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = chunksFor(w);
    const hb = headBlock(rows, heads);
    if (fuses(set.mla, rows, hb)) return fused(set, stream, qa, cache, scales, t, 0, tokens, counts, out, rows, heads, lw, nch, w, scale, true);
    try chunks(set, stream, qa, cache, scales, t, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, hb, true);
    try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
}

fn merge(set: glue.Set, stream: cuda.Stream, p: Partials, out: u64, counts: u64, rows: usize, heads: usize, lw: usize, nch: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ p.po, p.pm, p.pl, out, counts, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &a);
}

test "latent.py's tiling rules" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(8, 32)); // unchanged away from 17-24 heads
    try std.testing.expectEqual(@as(usize, 32), headBlock(64, 22)); // prompt chunks
    try std.testing.expectEqual(@as(usize, 32), headBlock(5, 22)); // TP=3 decode window from five rows
    try std.testing.expectEqual(@as(usize, 16), headBlock(4, 22)); // below five rows it stays narrow
    try std.testing.expectEqual(@as(usize, 3), chunksFor(1025));
    try std.testing.expectEqual(@as(usize, 1), chunksFor(512));
}
