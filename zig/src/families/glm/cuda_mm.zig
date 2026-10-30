//! GLM's bf16 linears on tensor cores (glm_mm): weights tiled once at load, any row count, row-stable K slices.
const std = @import("std");
const cuda = @import("cuda");

/// A linear's tiles: bf16 pairs (glm_mm_tile16), or `bits`-bit codes (glm_mm_tile4/6/8) with fp16 scales at `s`.
pub const Tiled = struct { w: u64, n: usize, k: usize, s: u64 = 0, bits: u8 = 16 };

/// The widths glm_mm packs to, by kernel set: bf16 first.
const widths = [_]u8{ 16, 4, 6, 8 };

fn widthIndex(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &widths, bits) orelse error.BadBits;
}

/// Bytes a packed linear's codes take (its scales follow, 256-aligned): what fits in place of its bf16 tiles.
pub fn packedBytes(n: usize, k: usize, bits: u8) [2]usize {
    return .{ std.mem.alignForward(usize, n * k * bits / 8, 256), n * (k / 64) * 2 };
}

/// The linears packed at load: each one's tiles pointer (sorted) and its bits; a linear not here is bf16.
pub const Packed = struct {
    ptrs: []const u64 = &.{},
    bits: []const u8 = &.{},

    /// The tiled weight `w` [n][k] as the matmul reads it: packed codes with their scales after them, or bf16.
    pub fn tiled(p: *const Packed, w: u64, n: usize, k: usize) Tiled {
        var lo: usize = 0;
        var hi: usize = p.ptrs.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (p.ptrs[mid] < w) lo = mid + 1 else hi = mid;
        }
        if (lo == p.ptrs.len or p.ptrs[lo] != w) return .{ .w = w, .n = n, .k = k };
        const b = p.bits[lo];
        return .{ .w = w, .n = n, .k = k, .s = w + packedBytes(n, k, b)[0], .bits = b };
    }
};

/// glm_mm's bf16 kernels by row bucket, the K-slice reduce and the tiling.
pub const Kernels = struct {
    by_rows: [4]cuda.Function, // bf16: 16, 32, 64, 128 rows a block
    packed_rows: [3][4]cuda.Function, // 4, 6 and 8 bits, as by_rows
    prompt: cuda.Function, // 128-row blocks, row blocks fastest: a prompt's blocks share each weight tile in L2
    reduce: cuda.Function,
    tile: cuda.Function,
    tile_packed: [3]cuda.Function, // glm_mm_tile4/6/8: packed rows and their scales into the tiles
    prompt_fixed_k: bool = false, // opt-in prompt experiment; never selected from the number of batched rows
    pdl: bool, // dependent launches: a matmul's weights stream in while the kernel before it finishes
    ovn: Ovn = .{}, // the speed-hunt variants (TF_GLM_OVN_*): each null while its knob is unset

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{ .by_rows = undefined, .packed_rows = undefined, .tile_packed = undefined, .prompt = try m.function("glm_mm16_rows"), .reduce = try m.function("glm_mm_reduce"), .tile = try m.function("glm_mm_tile16"), .pdl = true };
        inline for (.{ "glm_mm16_16", "glm_mm16_32", "glm_mm16_64", "glm_mm16_128" }, 0..) |name, i| {
            k.by_rows[i] = try m.function(name);
            try k.by_rows[i].allowDynamicShared(smem(bucket(i)));
        }
        inline for (widths[1..], 0..) |b, wi| {
            inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
                k.packed_rows[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ b, bm }));
                try k.packed_rows[wi][i].allowDynamicShared(sharedBytes(b, bm));
            }
            k.tile_packed[wi] = try m.function(std.fmt.comptimePrint("glm_mm_tile{d}", .{b}));
        }
        try k.prompt.allowDynamicShared(smem(128));
        if (std.c.getenv("TF_GLM_PDL")) |v| k.pdl = !std.mem.eql(u8, std.mem.span(v), "0");
        k.ovn = try Ovn.load(m, k.reduce);
        return k;
    }
};

/// The speed-hunt variants of glm_mm, each loaded only while its knob is set; every one keeps today's bits.
pub const Ovn = struct {
    wide: ?[4]cuda.Function = null, // TF_GLM_OVN_MM_WIDE: 128 x 128 prompt blocks in L2 row bands, by width
    wide_threads: u32 = 256, // 256: 64 x 32 warps (any value but 2), 512: 32 x 32 warps (2)
    cluster: ?[4][3]cuda.Function = null, // TF_GLM_OVN_MM_CLUSTER=<Mi>: buckets 16, 32, 64 summing slices in a cluster
    cluster_from: usize = 0, // ... for weights of at least this many values (32 Mi: production's CLUSTER_VALUES)
    dectile: ?[4]cuda.Function = null, // TF_GLM_OVN_MM_DECTILE=<blocks>: 16-row windows on 32-column tiles
    dectile_below: usize = 0, // ... where today's launch has fewer blocks than this
    swiglu: ?cuda.Function = null, // TF_GLM_OVN_SWIGLU_REDUCE: [gate | up]'s slices summed and SwiGLU'd in one kernel

    /// What to load: the knobs' values (`cluster_from` null: no clusters; 0 would cluster every sliced window).
    pub const Choice = struct { wide: u64 = 0, cluster_from: ?usize = null, dectile_below: usize = 0, swiglu: bool = false };

    /// The knobs as the environment sets them; `probe` (any glm_mm function) tells whether the image has clusters.
    pub fn load(m: cuda.Module, probe: cuda.Function) !Ovn {
        const mi = try cuda.ovn.int("MM_CLUSTER", 0);
        return choose(m, probe, .{
            .wide = try cuda.ovn.int("MM_WIDE", 0),
            .cluster_from = if (mi == 0) null else @intCast(mi << 20),
            .dectile_below = @intCast(try cuda.ovn.int("MM_DECTILE", 0)),
            .swiglu = cuda.ovn.on("SWIGLU_REDUCE"),
        });
    }

    /// The variants `c` names, loaded from glm_mm's module `m` (tf-cuda-test glm-mm-ovn drives every one directly).
    pub fn choose(m: cuda.Module, probe: cuda.Function, c: Choice) !Ovn {
        var o: Ovn = .{};
        if (c.wide != 0) {
            var f: [4]cuda.Function = undefined;
            inline for (widths, 0..) |b, wi| {
                const pre = std.fmt.comptimePrint("glm_mm{d}_", .{b});
                f[wi] = try m.function(if (c.wide == 2) pre ++ "w4x4" else pre ++ "w2x4");
                try f[wi].allowDynamicShared(tileShared(b, 128, 128, 3));
            }
            o.wide = f;
            o.wide_threads = if (c.wide == 2) 512 else 256;
        }
        if (c.cluster_from) |from| {
            // clusters need sm_90; every rank runs the same GPU and image, so every rank decides alike
            if (try probe.attribute(.binary_version) < 90) return error.OvnClustersNeedSm90;
            var f: [4][3]cuda.Function = undefined;
            inline for (widths, 0..) |b, wi| inline for (.{ 16, 32, 64 }, 0..) |bm, i| {
                f[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}c", .{ b, bm }));
                try f[wi][i].allowDynamicShared(sharedBytes(b, bm));
            };
            o.cluster = f;
            o.cluster_from = from;
        }
        if (c.dectile_below != 0) {
            var f: [4]cuda.Function = undefined;
            inline for (widths, 0..) |b, wi| {
                f[wi] = try m.function(std.fmt.comptimePrint("glm_mm{d}_16n32", .{b}));
                try f[wi].allowDynamicShared(tileShared(b, 16, 32, 4));
            }
            o.dectile = f;
            o.dectile_below = c.dectile_below;
        }
        if (c.swiglu) o.swiglu = try m.function("glm_mm_reduce_swiglu");
        return o;
    }

    /// Whether a decode window of `w` with `sk` slices sums them in a cluster (glm_mmB_<bm>c) instead of glm_mm_reduce.
    fn clusters(o: *const Ovn, w: Tiled, sk: usize) bool {
        return o.cluster != null and sk > 1 and w.n * w.k >= o.cluster_from;
    }
};

/// OTS 0083's prompt_tile rule (forward.py:465-475): 128 x 128 blocks where K is long and they fill the GPU twice over.
fn wideFits(rows: usize, w: Tiled) bool {
    return w.k >= 1024 and ((rows + 127) / 128) * ((w.n + 127) / 128) >= 96;
}

/// The weight `w` [n][k] as glm_mm reads it: packed where `packs` holds it, else bf16 tiles (b16.matmul's lookup).
pub fn tiledBy(packs: ?*const Packed, w: u64, n: usize, k: usize) Tiled {
    return if (packs) |p| p.tiled(w, n, k) else .{ .w = w, .n = n, .k = k };
}

/// swigluReduce for a bf16 [gate | up] as b16.matmul runs it: false, launching nothing, without glm_mm's kernels.
pub fn swigluReduceB16(mk: ?*const Kernels, packs: ?*const Packed, one: bool, s: cuda.Stream, x: u64, ldx: usize, w: u64, n: usize, kk: usize, act: u64, part: u64, m: usize, limit: f32) !bool {
    const k = mk orelse return false;
    return swigluReduce(k, s, x, ldx, tiledBy(packs, w, n, kk), act, part, m, limit, one);
}

const split_target = 192; // column tiles times K slices a launch aims at (the GB10's SMs, a few blocks each)

fn bucket(i: usize) usize {
    return @as(usize, 16) << @intCast(i);
}

fn smem(bm: usize) u32 {
    return sharedBytes(16, bm);
}

/// A block's shared memory: three stages of `bm` rows of x and a 64-column tile of weights (`bits / 2` words a lane).
pub fn sharedBytes(bits: u8, bm: usize) u32 {
    return tileShared(bits, bm, 64, 3);
}

/// glm_mm.cu's Tile::SMEM: `stages` stages of `bm` rows of x and `bn` columns of chunks with their fp16 scales.
pub fn tileShared(bits: u8, bm: usize, bn: usize, stages: usize) u32 {
    const wpc: usize = if (bits == 16) 8 else bits / 2;
    return @intCast(stages * (bm * 128 + bn / 8 * 32 * wpc * 4 + bn / 8 * 16));
}

/// K slices fixed by the shape: column tiles times slices reach the target, each slice at least 4 groups of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Bytes of fp32 partials a call of `m` rows needs (0: one slice; windows past 64 rows go 64 at a time).
pub fn partBytes(m: usize, w: Tiled) usize {
    const sk = splitK(w.n, w.k);
    return if (sk > 1) sk * @min(m, 64) * w.n * 4 else 0;
}

/// matmul's operands: K in groups of 64, columns in tiles of 8, rows on 16 bytes.
fn checkShape(w: Tiled, x: u64, ldx: usize) error{BadShape}!void {
    if (w.k % 64 != 0 or w.n % 8 != 0 or x % 16 != 0 or ldx % 8 != 0) return error.BadShape;
}

/// x [m][k] (`ldx` apart) by `w` -> out [m][n] (`wide`: fp32); `one`: prompt blocks at any rows, sums per row alike.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool, one: bool) !void {
    try checkShape(w, x, ldx);
    var done: usize = 0;
    const wi = try widthIndex(w.bits);
    while (done < m) { // windows to 64 rows a launch with K slices; prompts in one launch of 128-row blocks, one slice
        const big = if (one) !k.prompt_fixed_k else m - done >= 128;
        const rows = if (big) m - done else @min(m - done, 64);
        const i: usize = if (big) 3 else if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const sk = if (big) 1 else splitK(w.n, w.k); // a prompt's blocks share each weight tile in L2: no slices
        const o = out + done * w.n * @as(usize, if (wide) 4 else 2);
        const summed = try window(k, s, x + done * ldx * 2, ldx, w, wi, o, part, rows, i, sk, big, wide);
        if (sk > 1 and !summed) {
            const total = rows * w.n;
            var r: cuda.Args = .{};
            r.add(part);
            r.add(o);
            r.add(@as(c_longlong, @intCast(total)));
            r.add(@as(c_int, @intCast(sk)));
            r.add(@as(c_int, @intFromBool(wide)));
            try cuda.launch.launch(k.reduce, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &r);
        }
        done += rows;
    }
}

/// One window's matmul launch (bucket `i`, `sk` slices into `part`); true when its slices met in a cluster into `o`.
fn window(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, wi: usize, o: u64, part: u64, rows: usize, i: usize, sk: usize, big: bool, wide: bool) !bool {
    const bm = bucket(i);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_longlong, @intCast(ldx)));
    a.add(w.w);
    a.add(w.s);
    a.add(o);
    a.add(part);
    a.add(@as(c_int, @intFromBool(wide)));
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(w.n)));
    a.add(@as(c_int, @intCast(w.k)));
    a.add(@as(c_int, @intCast(sk)));
    const cols: u32 = @intCast((w.n + 63) / 64);
    const row_blocks: u32 = @intCast((rows + bm - 1) / bm);
    const z: u32 = @intCast(sk);
    if (big) {
        if (k.ovn.wide) |f| if (wideFits(rows, w)) { // one launch of 128 x 128 blocks in tile_of's row bands
            const blocks: u32 = @intCast(((rows + 127) / 128) * ((w.n + 127) / 128));
            try cuda.launch.launch(f[wi], .{ .grid = .{ .x = blocks }, .block = .{ .x = k.ovn.wide_threads }, .shared = tileShared(w.bits, 128, 128, 3), .pdl = k.pdl }, s, &a);
            return false;
        };
        if (wi == 0) {
            try cuda.launch.launch(k.prompt, .{ .grid = .{ .x = row_blocks, .y = cols }, .block = .{ .x = 256 }, .shared = smem(128), .pdl = k.pdl }, s, &a);
            return false;
        }
    } else if (k.ovn.clusters(w, sk)) { // a tile's slices are one cluster along z: slice 0 writes the sum
        const f = k.ovn.cluster.?[wi][i];
        try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = row_blocks, .z = z }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = sharedBytes(w.bits, bm), .cluster = .{ .x = 1, .y = 1, .z = z }, .pdl = k.pdl }, s, &a);
        return true;
    } else if (k.ovn.dectile) |f| if (i == 0 and cols * z < k.ovn.dectile_below) { // twice the blocks, same slices
        const narrow: u32 = @intCast((w.n + 31) / 32);
        try cuda.launch.launch(f[wi], .{ .grid = .{ .x = narrow, .y = row_blocks, .z = z }, .block = .{ .x = 128 }, .shared = tileShared(w.bits, 16, 32, 4), .pdl = k.pdl }, s, &a);
        return false;
    };
    // packed prompts take the 128-row bucket's blocks, one slice
    const f = if (wi == 0) k.by_rows[i] else k.packed_rows[wi - 1][i];
    try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = row_blocks, .z = z }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = sharedBytes(w.bits, bm), .pdl = k.pdl }, s, &a);
    return false;
}

/// TF_GLM_OVN_SWIGLU_REDUCE: x by [gate | up] `w` (n = 2 width) summed over its slices, rounded and SwiGLU'd into act.
/// false (nothing launched) where the call takes no slices or sums them in a cluster: the caller runs matmul + swiglu.
pub fn swigluReduce(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, act: u64, part: u64, m: usize, limit: f32, one: bool) !bool {
    const f = k.ovn.swiglu orelse return false;
    const sk = splitK(w.n, w.k);
    if (one or m >= 128 or sk == 1 or k.ovn.clusters(w, sk)) return false;
    try checkShape(w, x, ldx);
    const wi = try widthIndex(w.bits);
    const width = w.n / 2;
    var done: usize = 0;
    while (done < m) { // matmul's windows: up to 64 rows, every one with `sk` slices
        const rows = @min(m - done, 64);
        const i: usize = if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        _ = try window(k, s, x + done * ldx * 2, ldx, w, wi, 0, part, rows, i, sk, false, false);
        var r: cuda.Args = .{};
        r.add(part);
        r.add(act + done * width * 2);
        r.add(@as(c_longlong, @intCast(rows)));
        r.add(@as(c_longlong, @intCast(width)));
        r.add(@as(c_int, @intCast(sk)));
        r.add(limit);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * width + 255) / 256) }, .block = .{ .x = 256 } }, s, &r);
        done += rows;
    }
    return true;
}

/// Packed rows (cuda_qpack's) and fp16 scales [n][k / 64] at `src`, `ss` tiled into `dst`, `ds` (`bits` 4/6/8).
pub fn tilePacked(k: *const Kernels, s: cuda.Stream, bits: u8, src: u64, ss: u64, dst: u64, ds: u64, n: usize, kk: usize) !void {
    const wi = try widthIndex(bits);
    if (wi == 0) return error.BadBits;
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(ss);
    a.add(dst);
    a.add(ds);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile_packed[wi - 1], .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// A row-major bf16 [n][k] linear at `src` tiled into `dst` (as many bytes).
pub fn tile(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize, kk: usize) !void {
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

test "K slices as the shapes take them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 4096)); // 192 column tiles already
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 2048)); // 64 tiles, 32 groups: 4 slices of 8
}
