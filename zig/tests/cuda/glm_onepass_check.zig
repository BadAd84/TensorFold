//! glm-onepass-check (GPU): TF_GLM_OVN_SPARSE_ONEPASS's tf_glm_sparse_onepass against float64, on a slab and on pages.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const op = glm.onepass;
const kv8 = glm.kv8_ref;

const lw = 512;
const w = glm.sparse.width; // a token list's width (2,051)
const page_keys = 512; // positions a page holds
const page_rows = 600; // a page part's stride in 512-byte rows: its codes, its scales, then other caches' bytes
const sentinel: u16 = 0xA5A5;
const scale: f32 = 0.0625;

const Case = struct { rows: usize, heads: usize, cap: usize };
const cases = [_]Case{
    .{ .rows = 24, .heads = 22, .cap = 6000 }, // TP=3's first rank
    .{ .rows = 24, .heads = 21, .cap = 6000 }, // and the other two
    .{ .rows = 16, .heads = 17, .cap = 3000 }, // the kernel's head range, low end
    .{ .rows = 16, .heads = 24, .cap = 3000 }, // and high end
};

/// Row r's count is counts_by_row[r % 8]: a count-0 row (left alone), one key, part and whole tiles, a full list.
const counts_by_row = [_]usize{ 0, 1, 5, 31, 32, 33, 700, w };

const Bufs = std.ArrayList(cuda.DeviceBuffer);

fn put(gpu: check.Gpu, bufs: *Bufs, bytes: []const u8) !u64 {
    var b = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
    errdefer b.free();
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

pub fn run(gpu: check.Gpu) !void {
    const major: u32 = @intCast(try gpu.ctx.attribute(.compute_capability_major));
    try check.expect(major >= 9, "glm-onepass-check: tf_glm_sparse_onepass needs sm_90 or later (this GPU: sm_{d}x)", .{major});
    var k = try op.Kernels.load(gpu.d, cuda.kernels.glm_onepass, major);
    defer k.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var rng = std.Random.DefaultPrng.init(0x0851);
    const rand = rng.random();
    var fails: usize = 0;
    for (cases) |c| fails += try one(gpu, k.f, s, rand, c);
    try check.expect(fails == 0, "glm-onepass-check: {d} checks failed", .{fails});
    check.pass("glm-onepass-check: {d} cases within float64's bound; slab == pages == split launches == a second run", .{cases.len});
}

fn one(gpu: check.Gpu, f: cuda.Function, s: cuda.Stream, rand: std.Random, c: Case) !usize {
    const gpa = gpu.gpa;
    std.debug.print("  [rows {d} heads {d} cap {d}] build the cache, the token lists and q\n", .{ c.rows, c.heads, c.cap });
    // each position's latent row as the FP8 writer packs it: 512 codes and a scale
    const codes = try gpa.alloc(u8, c.cap * lw);
    defer gpa.free(codes);
    const scales = try gpa.alloc(f32, c.cap);
    defer gpa.free(scales);
    var row: [lw]u16 = undefined;
    for (0..c.cap) |p| {
        const mag: f32 = @floatFromInt(1 + p % 7);
        for (&row) |*x| x.* = kv8.bfBits(rand.floatNorm(f32) * mag);
        scales[p] = try kv8.pack(&row, codes[p * lw ..][0..lw]);
    }
    // the same rows on shuffled pages: a part's 512 rows of codes, their scales, then bytes no cache here owns
    const ids = try gpa.alloc(i32, (c.cap + page_keys - 1) / page_keys + 3);
    defer gpa.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    rand.shuffle(i32, ids);
    const part = page_rows * lw;
    const arena = try gpa.alloc(u8, ids.len * part);
    defer gpa.free(arena);
    @memset(arena, 0);
    for (0..c.cap) |p| {
        const at = @as(usize, @intCast(ids[p / page_keys])) * part;
        @memcpy(arena[at + (p % page_keys) * lw ..][0..lw], codes[p * lw ..][0..lw]);
        @memcpy(arena[at + page_keys * lw + (p % page_keys) * 4 ..][0..4], std.mem.asBytes(&scales[p]));
    }
    // token lists as tf_glm_dsa_tokens writes them: positions on a slab, page rows on pages
    const pos = try gpa.alloc(i32, c.rows * w);
    defer gpa.free(pos);
    const paged = try gpa.alloc(i32, c.rows * w);
    defer gpa.free(paged);
    const counts = try gpa.alloc(i32, c.rows);
    defer gpa.free(counts);
    for (0..c.rows) |r| {
        const n = counts_by_row[r % counts_by_row.len];
        counts[r] = @intCast(n);
        for (0..w) |i| {
            const p: usize = rand.uintLessThan(usize, c.cap);
            pos[r * w + i] = if (i < n) @intCast(p) else -1;
            paged[r * w + i] = if (i < n) ids[p / page_keys] * page_rows + @as(i32, @intCast(p % page_keys)) else -1;
        }
    }
    const q = try gpa.alloc(u16, c.rows * c.heads * lw);
    defer gpa.free(q);
    for (q) |*x| x.* = kv8.bfBits(rand.floatNorm(f32) * 0.5); // scores' spread 0.7 to 5 by the rows' magnitudes
    const blank = try gpa.alloc(u16, q.len);
    defer gpa.free(blank);
    @memset(blank, sentinel);

    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpa);
    }
    const dq = try put(gpu, &bufs, std.mem.sliceAsBytes(q));
    const dcodes = try put(gpu, &bufs, codes);
    const dscales = try put(gpu, &bufs, std.mem.sliceAsBytes(scales));
    const darena = try put(gpu, &bufs, arena);
    const dpt = try put(gpu, &bufs, std.mem.sliceAsBytes(ids));
    const dpos = try put(gpu, &bufs, std.mem.sliceAsBytes(pos));
    const dpaged = try put(gpu, &bufs, std.mem.sliceAsBytes(paged));
    const dcnt = try put(gpu, &bufs, std.mem.sliceAsBytes(counts));
    var outs: [4]u64 = undefined;
    for (&outs) |*o| o.* = try put(gpu, &bufs, std.mem.sliceAsBytes(blank));
    try gpu.d.synchronize();

    std.debug.print("  [rows {d} heads {d}] slab twice, slab in two launches, pages\n", .{ c.rows, c.heads });
    const slab: glm.latent.Table = .{};
    const pages: glm.latent.Table = .{ .pt = dpt, .ps = part, .page_rows = page_rows };
    try op.run(f, s, dq, dcodes, dscales, slab, dpos, dcnt, w, outs[0], c.rows, c.heads, scale);
    try op.run(f, s, dq, dcodes, dscales, slab, dpos, dcnt, w, outs[1], c.rows, c.heads, scale);
    const half = c.rows / 2 + 1; // off the counts' 8-row pattern
    const qrow = c.heads * lw * 2;
    try op.run(f, s, dq, dcodes, dscales, slab, dpos, dcnt, w, outs[2], half, c.heads, scale);
    try op.run(f, s, dq + half * qrow, dcodes, dscales, slab, dpos + half * w * 4, dcnt + half * 4, w, outs[2] + half * qrow, c.rows - half, c.heads, scale);
    try op.run(f, s, dq, darena, darena + page_keys * lw, pages, dpaged, dcnt, w, outs[3], c.rows, c.heads, scale);
    try s.synchronize();

    var got: [4][]u16 = undefined;
    var n_got: usize = 0;
    defer for (got[0..n_got]) |g| gpa.free(g);
    for (&got, outs) |*g, o| {
        g.* = try gpa.alloc(u16, q.len);
        n_got += 1;
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = o, .len = q.len * 2, .borrowed = true }).download(0, std.mem.sliceAsBytes(g.*));
    }
    var fails: usize = 0;
    const names = [_][]const u8{ "a second run", "two launches", "pages" };
    for (got[1..], names) |g, name| {
        check.sameBytes(name, std.mem.sliceAsBytes(g), std.mem.sliceAsBytes(got[0])) catch {
            fails += 1;
        };
    }
    fails += try reference(c, got[0], q, codes, scales, pos);
    return fails;
}

/// Every row against float64 softmax(q . k ks scale) . (k ks) over its tokens; count-0 rows must keep the sentinel.
fn reference(c: Case, got: []const u16, q: []const u16, codes: []const u8, scales: []const f32, pos: []const i32) !usize {
    var value: [256]f64 = undefined; // an e4m3 code's value (exact in bf16 and in f64)
    for (&value, 0..) |*v, b| v.* = kv8.bf(kv8.decoded(@intCast(b), 1.0));
    var sc: [w]f64 = undefined;
    var acc: [lw]f64 = undefined;
    var bad: usize = 0;
    var untouched: usize = 0;
    var worst: f64 = 0;
    for (0..c.rows) |r| {
        const n = counts_by_row[r % counts_by_row.len];
        const out = got[r * c.heads * lw ..][0 .. c.heads * lw];
        if (n == 0) {
            for (out) |x| untouched += @intFromBool(x != sentinel);
            continue;
        }
        for (0..c.heads) |h| {
            const qh = q[(r * c.heads + h) * lw ..][0..lw];
            var m = -std.math.inf(f64);
            for (0..n) |i| {
                const p: usize = @intCast(pos[r * w + i]);
                var dot: f64 = 0;
                for (qh, codes[p * lw ..][0..lw]) |x, b| dot += @as(f64, kv8.bf(x)) * value[b];
                sc[i] = dot * @as(f64, scales[p]) * @as(f64, scale);
                m = @max(m, sc[i]);
            }
            @memset(&acc, 0);
            var l: f64 = 0;
            for (0..n) |i| {
                const p: usize = @intCast(pos[r * w + i]);
                const e = @exp(sc[i] - m);
                l += e;
                const ek = e * @as(f64, scales[p]);
                for (&acc, codes[p * lw ..][0..lw]) |*a, b| a.* += ek * value[b];
            }
            var amax: f64 = 0;
            for (acc) |a| amax = @max(amax, @abs(a / l));
            for (acc, out[h * lw ..][0..lw]) |a, x| {
                const want = a / l;
                const err = @abs(@as(f64, kv8.bf(x)) - want);
                // bf16 P (2^-9 a key), the bf16 output (2^-9) and fp32 sums: within 2^-7 of the row's largest output
                if (!(err <= 0x1p-7 * (amax + @abs(want)))) bad += 1;
                if (amax > 0) worst = @max(worst, err / amax);
            }
        }
    }
    std.debug.print("  [rows {d} heads {d}] float64: worst error {e:.3} of a head's largest output, {d} outside, {d} count-0 values written\n", .{ c.rows, c.heads, worst, bad, untouched });
    if (bad != 0 or untouched != 0) {
        std.debug.print("FAIL glm-onepass-check rows {d} heads {d}: {d} outputs outside float64's bound, {d} count-0 values written\n", .{ c.rows, c.heads, bad, untouched });
        return 1;
    }
    return 0;
}
