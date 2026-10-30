//! GLM-5.3-Flash's glue on CUDA: the Python family's cuda/glue.py on glm_glue.cu, whose kernels keep Triton's bytes.

const std = @import("std");
const cuda = @import("cuda");
const mm_mod = @import("cuda_mm.zig");
const index_mod = @import("cuda_index.zig");

/// What the forward's kernels launch from: our glue's, glm_mm's when tiled, qlinear's lane for FP8 kinds, copy_rows.
pub const Set = struct { glue: *const Kernels, index: ?*const index_mod.Kernels = null, mm: ?*const mm_mod.Kernels = null, packs: ?*const mm_mod.Packed = null, prompt: bool = false, linear: cuda.qlinear.Linear = .{}, copy_rows: ?cuda.Function = null, pdl_early: bool = false, mla: Mla = .{}, onepass: ?cuda.Function = null };

/// The MLA launch choices (TF_GLM_OVN_MLA_FUSEMERGE, _ABSORB_VEC, _LATENT_MMA, _SEG_PASS); the defaults launch today's.
pub const Mla = struct {
    fuse_rows: usize = 0, // attention launches of this many rows or more, 32-head blocks: chunks and merge in one
    skip: bool = false, // chunk launches stop at the chunk's last live key tile
    vec: bool = false, // absorb_q and expand_v read bf16 kv_b by 16-byte loads
    mma: bool = false, // a prompt buffer's absorb and expand on the tensor cores (latent.py's PROMPT_MMA, a trade)
    seg: bool = false, // decode chunk launches on glm_glue_attn_seg.cuh's warp-specialised pass (same partials)

    /// MLA_FUSEMERGE: unset or 0 off; 1 fuses from latent.prompt_rows rows (prompt chunks); N > 1 from N rows.
    pub fn read() error{BadOvnKnob}!Mla {
        const fuse: usize = @intCast(try cuda.ovn.int("MLA_FUSEMERGE", 0));
        return .{ .fuse_rows = if (fuse == 1) @import("cuda_latent.zig").prompt_rows else fuse, .skip = fuse != 0, .vec = cuda.ovn.on("ABSORB_VEC"), .mma = cuda.ovn.on("LATENT_MMA"), .seg = cuda.ovn.on("SEG_PASS") };
    }

    /// Whether a TF_GLM_OVN knob whose kernels sit in glm_glue is set (MLA_FUSEMERGE, ABSORB_VEC, KVB8, DECODE_SHARD,
    /// LATENT_MMA, PK8, SEG_PASS).
    pub fn wanted() bool {
        return cuda.ovn.on("MLA_FUSEMERGE") or cuda.ovn.on("ABSORB_VEC") or cuda.ovn.on("KVB8") or cuda.ovn.on("DECODE_SHARD") or cuda.ovn.on("LATENT_MMA") or cuda.ovn.on("PK8") or cuda.ovn.on("SEG_PASS");
    }
};

/// A hidden-size norm's weight: bf16, or float32 (FP4-Local's) read as float32 by the _f32w kernels.
pub const Norm = union(enum) { bf16: u64, f32: u64 };

/// Our glm_glue.cu's functions.
pub const Kernels = struct {
    rmsnorm_4096: cuda.Function,
    rmsnorm_1536: cuda.Function,
    rmsnorm_512: cuda.Function,
    rmsnorm_4096_f32w: cuda.Function,
    residual_add: cuda.Function,
    stream_mean: cuda.Function,
    rank_sum: cuda.Function,
    hc_post: cuda.Function,
    combine: cuda.Function,
    combine_sy: cuda.Function,
    slice_sum: cuda.Function,
    top1: cuda.Function,
    top1_sums: cuda.Function,
    put_words: cuda.Function,
    snap_copy: cuda.Function, // TF_GLM_OVN_SNAP_GATHER's copy (launched only with it on)
    embed_b16: cuda.Function,
    conv_shift: cuda.Function,
    swiglu_512: cuda.Function,
    swiglu_64: cuda.Function,
    group_sums: cuda.Function,
    topk: cuda.Function,
    topk_sum: cuda.Function,
    router_part: cuda.Function,
    router_tile: cuda.Function,
    hc_partial: cuda.Function,
    hc_partial8: cuda.Function,
    hc_finish: cuda.Function,
    hc_finish_f32w: cuda.Function,
    absorb_q: cuda.Function,
    expand_v: cuda.Function,
    lat_write: cuda.Function,
    lat_write_fp8: cuda.Function,
    index_write: cuda.Function,
    pool_keys: cuda.Function,
    select_rows: cuda.Function,
    select_rows_wide: cuda.Function,
    bmm_decode: cuda.Function,
    bmm_rows: cuda.Function,
    bmm_tile: cuda.Function,
    bmm_tile1: cuda.Function,
    bmm_d32s1: cuda.Function,
    bmm_d32s2: cuda.Function,
    bmm_d32s3: cuda.Function,
    bmm_d64s2: cuda.Function,
    chunks_16: cuda.Function,
    chunks_32: cuda.Function,
    chunks_16_fp8: cuda.Function,
    chunks_32_fp8: cuda.Function,
    merge: cuda.Function,
    scores: cuda.Function,
    router16: ?cuda.Function, // TF_GLM_OVN_ROUTER16=<stages, 3 or 4>: 16-row router tiles for up to 16 rows (else null)
    // TF_GLM_OVN_HC_ROWS=16|32 (prompt rows) and TF_GLM_OVN_HC_DEC (rows under 64): glm_glue_hc_rows.cuh (hcLoad)
    hc_partial16: ?cuda.Function,
    hc_partial32: ?cuda.Function,
    hc_row16: ?cuda.Function,
    hc_row16_f32w: ?cuda.Function,
    hc_row32: ?cuda.Function,
    hc_row32_f32w: ?cuda.Function,
    hc_partial48: ?cuda.Function,
    hc_partial48_pdl: ?cuda.Function,
    // TF_GLM_OVN_GLUE_FUSE (hcGlue), TF_GLM_OVN_GLUE_ROW (hcRow) and TF_GLM_OVN_PDL_EARLY's variants (Set.pdl_early)
    hc_glue: cuda.Function,
    hc_glue_f32w: cuda.Function,
    hc_glue_pdl: cuda.Function,
    hc_glue_f32w_pdl: cuda.Function,
    hc_row: cuda.Function,
    hc_row_f32w: cuda.Function,
    hc_post_pdl: cuda.Function,
    hc_partial8_pdl: cuda.Function,
    hc_finish_pdl: cuda.Function,
    hc_finish_f32w_pdl: cuda.Function,
    router_tile_pdl: cuda.Function,
    swiglu_512_pdl: cuda.Function,
    swiglu_64_pdl: cuda.Function,
    rmsnorm_4096_pdl: cuda.Function,
    rmsnorm_1536_pdl: cuda.Function,
    rmsnorm_512_pdl: cuda.Function,
    rmsnorm_4096_f32w_pdl: cuda.Function,
    // TF_GLM_OVN_* forms (glm_glue_attn_ovn.cuh, glm_glue_latent_ovn.cuh), launched only with their knob on
    chunks_16_skip: cuda.Function,
    chunks_32_skip: cuda.Function,
    chunks_16_fp8_skip: cuda.Function,
    chunks_32_fp8_skip: cuda.Function,
    chunks_32_fused: cuda.Function,
    chunks_32_fp8_fused: cuda.Function,
    absorb_q_vec: cuda.Function,
    expand_v_vec: cuda.Function,
    absorb_q8: cuda.Function,
    expand_v8: cuda.Function,
    index_shard_pick: cuda.Function, // TF_GLM_OVN_DECODE_SHARD (glm_glue_shard.cuh): a rank's share, then the merge
    index_shard_merge: cuda.Function,
    absorb_mma: cuda.Function, // TF_GLM_OVN_LATENT_MMA (glm_glue_latent_mma.cuh): prompt rows on the tensor cores
    expand_mma: cuda.Function,
    absorb_mma8: cuda.Function, // and on TF_GLM_OVN_KVB8's E4M3 kv_b
    expand_mma8: cuda.Function,
    pool_keys_pk8: cuda.Function, // TF_GLM_OVN_PK8 (glm_glue_pk8.cuh): pooled keys as kv8 rows, and their scores
    scores_pk8: cuda.Function,
    chunks_seg16: cuda.Function, // TF_GLM_OVN_SEG_PASS (glm_glue_attn_seg.cuh): 16-head blocks, and 24 for 17-24 heads
    chunks_seg24: cuda.Function,
    chunks_seg16_fp8: cuda.Function,
    chunks_seg24_fp8: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        return loadOvn(m, Mla.wanted());
    }

    /// As load; the TF_GLM_OVN_* forms (chunks_16_skip on) looked up only with `ovn`, else null handles (none loads).
    pub fn loadOvn(m: cuda.Module, ovn: bool) !Kernels {
        const from = comptime std.meta.fieldIndex(Kernels, "chunks_16_skip").?;
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names, 0..) |name, i| {
            if (@FieldType(Kernels, name) == cuda.Function) @field(k, name) = if (i < from or ovn) try m.function("tf_glm_" ++ name) else .{ .d = m.d, .handle = null };
        }
        const router16 = try cuda.ovn.int("ROUTER16", 0);
        k.router16 = if (router16 == 0) null else try m.function(if (router16 == 4) "tf_glm_router16_s4" else "tf_glm_router16_s3");
        try hcLoad(&k, m, try hcRows(), cuda.ovn.on("HC_DEC"));
        try k.bmm_tile.allowDynamicShared(bmm_tile_smem);
        try k.bmm_tile1.allowDynamicShared(bmm_tile1_smem[1]);
        try k.chunks_16.allowDynamicShared(chunksSmem(16));
        try k.chunks_32.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8.allowDynamicShared(chunksSmem(32));
        if (!ovn) return k;
        try k.chunks_16_skip.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_skip.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8_skip.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8_skip.allowDynamicShared(chunksSmem(32));
        try k.chunks_32_fused.allowDynamicShared(chunksSmem(32));
        try k.chunks_32_fp8_fused.allowDynamicShared(chunksSmem(32));
        inline for (.{ k.absorb_mma, k.expand_mma, k.absorb_mma8, k.expand_mma8 }) |f| try f.allowDynamicShared(latent_mma_smem);
        inline for (.{ k.chunks_seg16, k.chunks_seg16_fp8 }, .{ k.chunks_seg24, k.chunks_seg24_fp8 }) |a, b| {
            try a.allowDynamicShared(seg_smem[0]);
            try b.allowDynamicShared(seg_smem[1]);
        }
        return k;
    }
};

/// An element-wise kernel's launch: a block of 256 columns, `rows` x the blocks to cover `d`.
fn rowsLaunch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 256) }, .block = .{ .x = 256 } }, stream, a);
}

/// As above, 8 columns a thread (16-byte loads), 1024 a block of 128 (Triton's BLOCK); d % 8 == 0, rows 16B-aligned.
fn rows8Launch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize, ptrs: anytype) !void {
    if (d % 8 != 0) return error.WidthNotEights;
    inline for (ptrs) |p| if (p % 16 != 0) return error.RowsUnaligned;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 1024) }, .block = .{ .x = 128 } }, stream, a);
}

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn pow2(d: usize) i64 {
    return @intCast(std.math.ceilPowerOfTwoAssert(usize, d));
}

/// glue.rmsnorm: bf16(w * bf16(x / sqrt(mean(x^2) + eps))) into `out`, with `xs` each 64 outputs' fp32 sum.
pub fn rmsnorm(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const k = set.glue;
    const e = set.pdl_early;
    const f, const warps: u32 = switch (d) {
        4096 => .{ if (e) k.rmsnorm_4096_pdl else k.rmsnorm_4096, 8 },
        1536 => .{ if (e) k.rmsnorm_1536_pdl else k.rmsnorm_1536, 4 },
        512 => .{ if (e) k.rmsnorm_512_pdl else k.rmsnorm_512, 4 },
        else => return error.NoRmsnormWidth,
    };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(x_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(o_stride)));
    a.add(xs orelse @as(u64, 0));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(xs != null)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
}

/// rmsnorm by a Norm: a float32 weight (the hidden size) through rmsnorm_4096_f32w, rmsnorm_4096's arithmetic.
pub fn rmsnormBy(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, n: Norm, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const g = switch (n) {
        .bf16 => |w| return rmsnorm(set, stream, x, x_stride, w, out, o_stride, xs, eps, rows, d),
        .f32 => |g| g,
    };
    if (d != 4096) return error.NoRmsnormWidth;
    var a: cuda.Args = .{};
    inline for (.{ x, i64of(x_stride), g, out, i64of(o_stride), xs orelse @as(u64, 0), eps, @as(c_int, @intFromBool(xs != null)) }) |v| a.add(v);
    const f = if (set.pdl_early) set.glue.rmsnorm_4096_f32w_pdl else set.glue.rmsnorm_4096_f32w;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// swiglu's column block: 512 where it divides the width, else 64 (TP=3's 704 and 640, the latter 128 in glue.py).
fn swigluBlock(w: usize) !usize {
    if (w % 512 == 0) return 512;
    if (w % 64 == 0) return 64;
    return error.NoSwigluBlock;
}

/// glue.swiglu: [gate | up] -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)), 64-sums in `xs`.
pub fn swiglu(set: Set, stream: cuda.Stream, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
    const block = try swigluBlock(w);
    const k = set.glue;
    const e = set.pdl_early;
    const f, const threads: u32 = if (block == 512) .{ if (e) k.swiglu_512_pdl else k.swiglu_512, 128 } else .{ if (e) k.swiglu_64_pdl else k.swiglu_64, 64 };
    var a: cuda.Args = .{};
    inline for (.{ gu, out, xs, limit, i64of(w) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(w / block) }, .block = .{ .x = threads } }, stream, &a);
}

/// glm_glue_bmm.cuh's Tile::smem(f32) of tf_glm_bmm_tile: 3 stages of 128 x and 64 w rows, 64 bf16 a row.
pub const bmm_tile_smem: u32 = 3 * (128 + 64) * 64 * 2;
/// tf_glm_bmm_tile1's [bf16, fp32 out]: 3 stages of 128 x and 128 w rows, 32 bf16 a row, or the fp32 tile if larger.
pub const bmm_tile1_smem = [2]u32{ 3 * (128 + 128) * 32 * 2, 128 * (128 + 4) * 4 };
/// glm_glue_attn.cuh's wide_smem<hb>(): q (hb x 512), two 32-key tiles, p (hb x 32) in bf16, then max/sum scratch.
pub fn chunksSmem(comptime hb: u32) u32 {
    return (hb * 512 + 2 * 32 * 512 + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// glm_glue_latent_mma.cuh's SMEM: 3 stages of a 64 x 64 q / o tile and a 128 x 64 kv_b tile, bf16.
pub const latent_mma_smem: u32 = 3 * (64 * 64 + 128 * 64) * 2;

/// glm_glue_attn_seg.cuh's Smem<2> and Smem<3>: three 32-key bf16 tiles, P (8 NQ heads x 32) bf16, 8 NQ alphas.
pub const seg_smem = [2]u32{ 3 * 32 * 1024 + 16 * 32 * 2 + 16 * 4, 3 * 32 * 1024 + 24 * 32 * 2 + 24 * 4 };
/// Its threads: 5 PV, NQ QK and 4 producer warps (NQ 2 for 16-head blocks, 3 for 24).
pub const seg_threads = [2]u32{ (5 + 2 + 4) * 32, (5 + 3 + 4) * 32 };

/// K slices of the router's dot (glue.ROUTER_KS): fixed by the shape, as the sliced and the single pass differ in bits.
pub const router_ks = 8;

/// glue.router: out [m][ne] fp32 = x . w rows; d a multiple of router_ks x 64: K slices in `part`, added in order.
pub fn router(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    try routerSlices(set, stream, x, x_stride, w, part, m, d, ne);
    try sliceSum(set, stream, part, out, m * ne, router_ks, true);
}

/// glue._router_part alone: the router's K slices into `part` [router_ks][m][ne] fp32, not yet added.
pub fn routerSlices(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    if (d % (router_ks * 64) != 0) return error.RouterWidth; // the Python family's single-pass _router: no GLM width
    var a: cuda.Args = .{};
    inline for (.{ x, w, part, i64of(m), i64of(x_stride), i64of(d), i64of(ne), @as(c_int, router_ks) }) |v| a.add(v);
    if (x % 16 == 0 and x_stride % 8 == 0) { // rows on 16 bytes: cp.async tiles (decode's rows past m load as zeros)
        if (set.glue.router16) |f| if (m <= 16) { // TF_GLM_OVN_ROUTER16: 16-row tiles, each warp 8 experts
            return cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a);
        };
        const tile = if (set.pdl_early) set.glue.router_tile_pdl else set.glue.router_tile; // _pdl: launched dependent
        try cuda.launch.launch(tile, .{ .grid = .{ .x = cdiv(m, 64), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 }, .pdl = set.pdl_early }, stream, &a);
    } else try cuda.launch.launch(set.glue.router_part, .{ .grid = .{ .x = cdiv(m, 16), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a);
}

/// glue.select: top_k by sigmoid + bias (lower id on ties), normed if `norm`, times `scale`, then the shared expert.
pub fn select(set: Set, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    var a: cuda.Args = .{};
    inline for (.{ logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
}

/// select on routerSlices' `part`: topk adds the K slices in sliceSum's order itself (the same logits).
pub fn selectSum(set: Set, stream: cuda.Stream, part: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    var a: cuda.Args = .{};
    inline for (.{ part, i64of(rows * experts), @as(c_int, router_ks), bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.topk_sum, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
}

/// glue.combine: out [rows][d] fp32 = sum over slots in order of wts[r][k] y[r][k] (y fp32 in decode, bf16 in prefill).
pub fn combine(set: Set, stream: cuda.Stream, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ y, @as(c_int, @intFromBool(y_f32)), wts, out, i64of(d), cint(slots) }) |v| a.add(v);
    try rows8Launch(set.glue.combine, stream, &a, rows, d, .{ y, out });
}

/// glue.combine_shared: combine with the last slot read in place from `sy` [rows][d] fp32 (rounded for bf16 slots).
pub fn combineSy(set: Set, stream: cuda.Stream, y: u64, y_f32: bool, sy: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ y, @as(c_int, @intFromBool(y_f32)), sy, wts, out, i64of(d), cint(slots) }) |v| a.add(v);
    try rows8Launch(set.glue.combine_sy, stream, &a, rows, d, .{ y, sy, out });
}

/// K blocks of the hyper-connection mixing dots (glue.HC_BLOCKS): fixed by the shape, like the router's slices.
pub const hc_blocks = 16;

/// The mixing kernels' scratch: rows x hc_blocks x 32 fp32 (24 dots and the sum of squares of each K block).
pub fn hcPartBytes(rows: usize) usize {
    return rows * hc_blocks * 32 * 4;
}

/// TF_GLM_OVN_HC_ROWS: unset or 0 off, else 16 or 32 prompt rows a block; any other value is refused.
pub fn hcRows() error{BadOvnKnob}!u32 {
    const n = try cuda.ovn.int("HC_ROWS", 0);
    return switch (n) {
        0, 16, 32 => @intCast(n),
        else => error.BadOvnKnob,
    };
}

/// HC_ROWS' row kernels' dynamic shared memory: a block's partials, rows x hc_blocks x 32 fp32.
pub fn hcRowSmem(rows: u32) u32 {
    return rows * hc_blocks * 32 * 4;
}

/// HC_ROWS' kernels for `rows` (16 or 32, 0 none) and HC_DEC's when `dec` into `k`, every other of them null.
pub fn hcLoad(k: *Kernels, m: cuda.Module, rows: u32, dec: bool) !void {
    inline for (.{ 16, 32 }) |n| {
        const p = std.fmt.comptimePrint("{d}", .{n});
        @field(k, "hc_partial" ++ p) = if (rows == n) try m.function("tf_glm_hc_partial" ++ p) else null;
        inline for (.{ "", "_f32w" }) |w| {
            const f = if (rows == n) try m.function("tf_glm_hc_row" ++ p ++ w) else null;
            if (f) |x| try x.allowDynamicShared(hcRowSmem(n));
            @field(k, "hc_row" ++ p ++ w) = f;
        }
    }
    k.hc_partial48 = if (dec) try m.function("tf_glm_hc_partial48") else null;
    k.hc_partial48_pdl = if (dec) try m.function("tf_glm_hc_partial48_pdl") else null;
}

/// HC_ROWS' kernel for prompt rows (`prompt`: the forward's b.prefill, as production's prompt=), and its rows a block.
const RowsForm = struct { f: cuda.Function, n: u32 };
fn rowsForm(prompt: bool, by16: ?cuda.Function, by32: ?cuda.Function) ?RowsForm {
    if (!prompt) return null;
    if (by32) |f| return .{ .f = f, .n = 32 };
    if (by16) |f| return .{ .f = f, .n = 16 };
    return null;
}

/// glue.hc_pre: streams [rows][s d] -> the collapsed RMS-normed row and its 64-sums `xs`; `post`, `comb` (Sinkhorn).
pub fn hcPre(set: Set, stream: cuda.Stream, x: u64, fn_w: u64, base: u64, scale: u64, norm: Norm, out: u64, xs: u64, post: u64, comb: u64, part: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    const k = set.glue;
    var a: cuda.Args = .{};
    inline for (.{ x, fn_w, part, i64of(s * d), @as(c_int, hc_blocks) }) |v| a.add(v);
    // Prompt rows: HC_ROWS' R-row blocks; under 64 rows: HC_DEC's hc.cu layout or 16 blocks a row; else 4 rows a block.
    if (rowsForm(set.prompt, k.hc_partial16, k.hc_partial32)) |r| {
        a.add(i64of(rows));
        try cuda.launch.launch(r.f, .{ .grid = .{ .x = cdiv(rows, r.n), .y = hc_blocks }, .block = .{ .x = 128 } }, stream, &a);
    } else if (rows < 64) {
        if (k.hc_partial48) |plain| {
            const f = if (set.pdl_early) (k.hc_partial48_pdl orelse return error.HcDecUnloaded) else plain;
            var c: cuda.Args = .{};
            inline for (.{ x, fn_w, part, i64of(rows) }) |v| c.add(v);
            try cuda.launch.launch(f, .{ .grid = .{ .x = hc_blocks, .y = 3, .z = cdiv(rows, 4) }, .block = .{ .x = 288 }, .pdl = set.pdl_early }, stream, &c);
        } else {
            const p8 = if (set.pdl_early) k.hc_partial8_pdl else k.hc_partial8; // _pdl: launched dependent
            try cuda.launch.launch(p8, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks }, .block = .{ .x = 1024 }, .pdl = set.pdl_early }, stream, &a);
        }
    } else {
        a.add(i64of(rows));
        try cuda.launch.launch(k.hc_partial, .{ .grid = .{ .x = cdiv(rows, 4), .y = hc_blocks }, .block = .{ .x = 128 } }, stream, &a);
    }
    const e = set.pdl_early;
    const f, const w = switch (norm) {
        .bf16 => |p| .{ if (e) set.glue.hc_finish_pdl else set.glue.hc_finish, p },
        .f32 => |p| .{ if (e) set.glue.hc_finish_f32w_pdl else set.glue.hc_finish_f32w, p },
    };
    var b: cuda.Args = .{};
    inline for (.{ x, part, base, scale, w, out, xs, post, comb, eps, hc_eps, @as(c_int, hc_blocks), cint(iters) }) |v| b.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &b);
}

/// glue.hc_post: branch = bf16(`g`, rank 0 first); xout[s] = bf16(post[s] branch + sum_t comb[t][s] x[t]); x may alias.
/// `world` here and in hcGlue, hcRow: the ranks, or rotlayer.word's (a rotated MoE layer's partials from its owner).
pub fn hcPost(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, post: u64, comb: u64, rows: usize, d: usize, s: usize, world: usize) !void {
    if (s != 4) return error.StreamsNotFour;
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, post, comb, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(if (set.pdl_early) set.glue.hc_post_pdl else set.glue.hc_post, stream, &a, rows, d, .{ x, xout, g });
}

/// TF_GLM_OVN_GLUE_FUSE: hcPost from x into xout (never in place), hcPre on xout, one kernel; `count` u32 zeros a row.
pub fn hcGlue(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, fn_w: u64, base: u64, scale: u64, norm: Norm, out: u64, xs: u64, post: u64, comb: u64, part: u64, count: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize, world: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    if (rows == 0 or rows >= 64 or x == xout) return error.HcGlueWindow;
    inline for (.{ x, xout, g }) |p| if (p % 16 != 0) return error.RowsUnaligned;
    const k = set.glue;
    const e = set.pdl_early;
    const f, const w = switch (norm) {
        .bf16 => |p| .{ if (e) k.hc_glue_pdl else k.hc_glue, p },
        .f32 => |p| .{ if (e) k.hc_glue_f32w_pdl else k.hc_glue_f32w, p },
    };
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, i64of(rows * d), cint(world), fn_w, part, count, base, scale, w, out, xs, post, comb, eps, hc_eps, cint(iters) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks }, .block = .{ .x = 256 } }, stream, &a);
}

/// TF_GLM_OVN_GLUE_ROW: hcPost in place then hcPre, one kernel of 4 rows a block (prompt rows: HC_ROWS' 16 or 32).
pub fn hcRow(set: Set, stream: cuda.Stream, x: u64, g: u64, fn_w: u64, base: u64, scale: u64, norm: Norm, out: u64, xs: u64, post: u64, comb: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize, world: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    inline for (.{ x, g }) |p| if (p % 16 != 0) return error.RowsUnaligned;
    const k = set.glue;
    const f, const w, const r = switch (norm) {
        .bf16 => |p| .{ k.hc_row, p, rowsForm(set.prompt, k.hc_row16, k.hc_row32) },
        .f32 => |p| .{ k.hc_row_f32w, p, rowsForm(set.prompt, k.hc_row16_f32w, k.hc_row32_f32w) },
    };
    var a: cuda.Args = .{};
    inline for (.{ x, g, i64of(rows * d), cint(world), fn_w, base, scale, w, out, xs, post, comb, eps, hc_eps, cint(iters), i64of(rows) }) |v| a.add(v);
    if (r) |h| return cuda.launch.launch(h.f, .{ .grid = .{ .x = cdiv(rows, h.n) }, .block = .{ .x = 256 }, .shared = hcRowSmem(h.n) }, stream, &a);
    try cuda.launch.launch(f, .{ .grid = .{ .x = cdiv(rows, 4) }, .block = .{ .x = 256 } }, stream, &a);
}

/// glue.rank_sum: out bf16 = partials summed rank 0 first, `me`'s from `own`, others' from `peers` in rank order.
pub fn rankSum(set: Set, stream: cuda.Stream, own: u64, peers: u64, peer_stride: usize, out: u64, me: usize, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ own, peers, out, i64of(peer_stride), i64of(d), cint(world), cint(me) }) |v| a.add(v);
    try rows8Launch(set.glue.rank_sum, stream, &a, rows, d, .{ own, peers, out });
}

/// glue.residual_add (the MTP block's plain residual): xout = bf16(x + bf16(the partials `g` summed rank 0 first)).
pub fn residualAdd(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.residual_add, stream, &a, rows, d, .{ x, xout, g });
}

/// glue.stream_mean: out [rows][d] = bf16 of the s streams' mean (the final collapse).
pub fn streamMean(set: Set, stream: cuda.Stream, x: u64, out: u64, rows: usize, d: usize, s: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, out, i64of(d), cint(s) }) |v| a.add(v);
    try rows8Launch(set.glue.stream_mean, stream, &a, rows, d, .{ x, out });
}

/// glue.embed of a BF16 table: each row's token (ids i32) as `copies` streams of d values, out [rows][copies d].
pub fn embedB16(set: Set, stream: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ ids, table, out, i64of(d), cint(copies) }) |v| a.add(v);
    try rowsLaunch(set.glue.embed_b16, stream, &a, rows, d);
}

/// forward._shift_conv: each layer's 3-row conv window becomes rows keep .. keep + 2 of [window; projection rows].
pub fn convShift(set: Set, stream: cuda.Stream, conv: u64, proj: u64, keep: usize, proj_layer: usize, proj_row: usize, layers: usize, c: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ conv, proj, i64of(keep), i64of(3 * c), i64of(proj_layer), i64of(proj_row), i64of(c) }) |v| a.add(v);
    try rowsLaunch(set.glue.conv_shift, stream, &a, layers, c);
}

/// glue._router_sum and qmm._reduce: `total` values summed over `ks` slices in slice order, fp32 or bf16 out.
pub fn sliceSum(set: Set, stream: cuda.Stream, part: u64, out: u64, total: usize, ks: usize, wide: bool) !void {
    var a: cuda.Args = .{};
    inline for (.{ part, out, i64of(total), cint(ks), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.slice_sum, .{ .grid = .{ .x = cdiv(total, 1024) }, .block = .{ .x = 128 } }, stream, &a);
}

test "the norm's block is the next power of two of the width" {
    try std.testing.expectEqual(@as(i64, 4096), pow2(4096));
    try std.testing.expectEqual(@as(i64, 8192), pow2(6144));
}

test "the wrappers' shape rules as glue.py's" {
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(6144)); // swiglu's block at the dense TP=2 width
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(1024)); // a routed expert's rows at TP=2
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(704)); // TP=3's first two ranks
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(640)); // TP=3's last rank
    try std.testing.expectError(error.NoSwigluBlock, swigluBlock(96));
    try std.testing.expectEqual(@as(i64, 512), pow2(288 + 1)); // _topk's BLOCK for 288 experts
    try std.testing.expectEqual(@as(i64, 16), pow2(8 + 1)); // its SLOTP for top-8 and the shared slot
}
