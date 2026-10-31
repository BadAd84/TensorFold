//! TF_GLM_OVN_SPARSE_ONEPASS: prompt rows' sparse latent attention in one pass a row (glm_onepass.cu), production's.

const std = @import("std");
const cuda = @import("cuda");
const latent = @import("cuda_latent.zig");
const glue = @import("cuda_glue.zig");

/// glm_onepass.cu's block (12 warps) and its Smem: three bf16 32-key tiles, P [24][32] bf16, scales, alphas, sums.
pub const threads = 384;
pub const smem: u32 = 3 * 32 * 512 * 2 + 24 * 32 * 2 + 3 * 32 * 4 + 24 * 4 + 24 * 4;
/// The heads a rank carries for the kernel: production's 16 < H <= 24 (TP=3's 22 / 21).
pub const min_heads = 17;
pub const max_heads = 24;

pub const Kernels = struct {
    module: cuda.Module,
    f: cuda.Function,

    /// `image`: cuda.kernels.glm_onepass; `major`: the device's compute capability (the kernel needs sm_90 or later).
    pub fn load(d: *const cuda.Driver, image: []const u8, major: u32) !Kernels {
        if (image.len == 0) return error.NoOnepassKernel;
        if (major < 9) return error.OnepassNeedsSm90;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const f = try m.function("tf_glm_sparse_onepass");
        try f.allowDynamicShared(smem);
        return .{ .module = m, .f = f };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }
};

/// The knob, read at load: unset, empty or 0 keeps today's chunk programs and merge for every row.
pub fn wanted() bool {
    return cuda.ovn.on("SPARSE_ONEPASS");
}

/// TF_GLM_OVN_ONEPASS_EXACT, read at load: the bit-identical one-pass path (tf_glm_onepass_exact), default off.
pub fn wantedExact() bool {
    return cuda.ovn.on("ONEPASS_EXACT");
}

/// The kernels when the knob asks for them, else null; one that will not load is logged and left out.
pub fn open(d: *const cuda.Driver, image: []const u8, major: u32) ?Kernels {
    if (!wanted()) return null;
    return Kernels.load(d, image, major) catch |err| {
        std.log.warn("TF_GLM_OVN_SPARSE_ONEPASS: the onepass kernel is not used ({t}); prompt rows keep the chunk programs", .{err});
        return null;
    };
}

/// Whether a sparse call takes the kernel: prompt buffers (production's b.prefill), an FP8 cache, 17-24 heads.
pub fn takes(prompt: bool, scales: u64, heads: usize, lw: usize) bool {
    return prompt and scales != 0 and lw == 512 and heads >= min_heads and heads <= max_heads;
}

/// `takes`'s shapes, only where the default uses 32-head blocks (headBlock == heads_tile_wide), else chunk programs.
pub fn takesExact(prompt: bool, scales: u64, heads: usize, lw: usize, rows: usize) bool {
    return takes(prompt, scales, heads, lw) and latent.headBlock(rows, heads) == latent.heads_tile_wide;
}

/// latent.sparse_onepass: rows with counts > 0 over their `w`-wide token lists (as `t` maps them) into out bf16.
pub fn run(f: cuda.Function, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: latent.Table, tokens: u64, counts: u64, w: usize, out: u64, rows: usize, heads: usize, scale: f32) !void {
    if (rows == 0) return;
    if (heads < min_heads or heads > max_heads) return error.OnepassHeads;
    const page_rows: u32 = if (t.pt != 0) @intCast(t.page_rows) else 0; // 0: a slab, scales[k]
    var a: cuda.Args = .{};
    inline for (.{ qa, cache, scales, page_rows, tokens, @as(c_int, @intCast(w)), counts, out, @as(c_int, @intCast(heads)), scale }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads }, .shared = smem }, stream, &a);
}

/// The exact one-pass path (tf_glm_onepass_exact): glm_attn_ovn::chunks<32,true,true> in registers, one block a row.
pub fn runExact(f: cuda.Function, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: latent.Table, tokens: u64, counts: u64, w: usize, out: u64, rows: usize, heads: usize, scale: f32) !void {
    if (rows == 0) return;
    if (heads < min_heads or heads > max_heads) return error.OnepassHeads;
    const nch: c_int = @intCast(latent.chunksFor(w));
    var a: cuda.Args = .{};
    inline for (.{ qa, cache, scales, tokens, counts, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, t.pt, @as(i64, @intCast(t.ps)), nch }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1 }, .block = .{ .x = 256 }, .shared = glue.chunksSmem(32) }, stream, &a);
}

test "onepass: prompt rows on an FP8 cache at TP=3's head counts only" {
    try std.testing.expect(takes(true, 0x1000, 22, 512));
    try std.testing.expect(takes(true, 0x1000, 21, 512));
    try std.testing.expect(!takes(false, 0x1000, 22, 512)); // decode windows keep the chunk programs
    try std.testing.expect(!takes(true, 0, 22, 512)); // a BF16 cache keeps them too
    try std.testing.expect(!takes(true, 0x1000, 32, 512)); // TP=2's 32 heads
    try std.testing.expect(!takes(true, 0x1000, 16, 512));
    try std.testing.expectEqual(@as(u32, 100416), smem); // glm_onepass.cu's static_assert
}

test "onepass exact: the knob gates the path and the 32-head-block shapes take it" {
    cuda.ovn.fake("ONEPASS_EXACT", null);
    try std.testing.expect(!wantedExact());
    cuda.ovn.fake("ONEPASS_EXACT", "0");
    try std.testing.expect(!wantedExact());
    cuda.ovn.fake("ONEPASS_EXACT", "1");
    try std.testing.expect(wantedExact());
    // the exact path takes the onepass shapes only where the default uses 32-head blocks (rows >= 5 at 17-24 heads)
    try std.testing.expect(takesExact(true, 0x1000, 22, 512, 64)); // a prompt chunk
    try std.testing.expect(takesExact(true, 0x1000, 21, 512, 5)); // the five-row wide-block floor
    try std.testing.expect(!takesExact(true, 0x1000, 22, 512, 4)); // four rows: the default uses 16-head blocks
    try std.testing.expect(!takesExact(true, 0x1000, 22, 512, 1)); // one row (a chunk's last row) too
    try std.testing.expect(!takesExact(true, 0, 22, 512, 64)); // a BF16 cache keeps the chunk programs
    try std.testing.expect(!takesExact(false, 0x1000, 22, 512, 64)); // decode windows too
    try std.testing.expect(!takesExact(true, 0x1000, 32, 512, 64)); // TP=2's 32 heads
    cuda.ovn.fake("ONEPASS_EXACT", null);
}
