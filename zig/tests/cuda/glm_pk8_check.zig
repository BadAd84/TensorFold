//! glm-pk8-check (GPU): TF_GLM_OVN_PK8's kv8 pooled keys. The writer's rows equal kv8.quantize_rows of the bf16
//! writer's rows byte for byte; both scorers on kv8 rows equal today's on the rows dequantized to bf16 (a power-of-two
//! scale on a dot is exact), slab and pages.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const base = @import("glm_index_check.zig");
const sp = glm.sparse;

const d = base.d;
const row8 = d + 4; // glm_pk8_row.cuh's ROW
const Case = base.Case;
const Bufs = base.Bufs;

const cases = [_]Case{
    .{ .cap = 16384, .pos = 4 * 16384 - 40, .rows = 2 },
    .{ .cap = 65536, .pos = 4 * 65536 - 9, .rows = 4, .paged = true },
    .{ .cap = 16384, .pos = 4 * 16384 - 1130, .rows = 512 },
    .{ .cap = 65536, .pos = 4 * 65536 - 520, .rows = 512, .paged = true, .heads = 20 },
};

/// Round half to even (the E4M3 conversion's rounding).
fn even(v: f64) f64 {
    const f = @floor(v);
    const r = v - f;
    return if (r > 0.5 or (r == 0.5 and @mod(f, 2) == 1)) f + 1 else f;
}

/// cvt.rn.satfinite.e4m3: nearest even, saturating at 448 (kv8 rows stay within 256).
fn e4m3(x: f32) u8 {
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a: f64 = @abs(@as(f64, x));
    if (a < 0x1p-6) return sign | @as(u8, @intFromFloat(even(a * 512.0))); // subnormals; 8 is the first normal's code
    const fr = std.math.frexp(a);
    var ex: i32 = fr.exponent - 1;
    var q = even((fr.significand * 2 - 1) * 8);
    if (q == 8) {
        q = 0;
        ex += 1;
    }
    const code: u32 = @as(u32, @intCast(ex + 7)) << 3 | @as(u32, @intFromFloat(q));
    return sign | @as(u8, @intCast(@min(code, 0x7E)));
}

fn e4m3Value(c: u8) f32 {
    const e: i32 = (c >> 3) & 15;
    const m: f32 = @floatFromInt(c & 7);
    const v = if (e == 0) m * 0x1p-9 else std.math.ldexp(1.0 + m / 8.0, e - 7);
    return if (c & 0x80 != 0) -v else v;
}

/// kv8.quantize_rows of one bf16 row: codes of x * inv, then the scale 2^(ceil(log2 amax) - 8).
fn quantize(x: []const u16, out: *[row8]u8) void {
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(@as(f32, @bitCast(@as(u32, v) << 16))));
    const bits: u32 = @bitCast(amax);
    const e: i32 = std.math.clamp(@as(i32, @intCast((bits >> 23) & 0xFF)) + @intFromBool(bits & 0x7FFFFF != 0) - 8, 1, 254);
    const inv: f32 = @bitCast(@as(u32, @intCast(254 - e)) << 23);
    for (x, 0..) |v, i| out[i] = e4m3(@as(f32, @bitCast(@as(u32, v) << 16)) * inv);
    out[d..][0..4].* = @bitCast(@as(u32, @intCast(e)) << 23);
}

/// A kv8 row's values as bf16 (exact: few significant bits times a power of two).
fn dequantize(r: *const [row8]u8, out: []u16) void {
    const s: f32 = @bitCast(r[d..][0..4].*);
    for (out, 0..) |*v, i| v.* = @intCast(@as(u32, @bitCast(e4m3Value(r[i]) * s)) >> 16);
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    const ks = try glm.glue.Kernels.loadOvn(glue_mod, true);
    var idx = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .all);
    defer idx.unload();
    var on = idx;
    on.pk8_fn = try idx.module.function("tf_glm_index_scores_pk8");
    var rng = std.Random.DefaultPrng.init(0x9b8);
    const rand = rng.random();
    var fails = try writer(gpu, s, rand, &ks);
    for (cases) |c| fails += try scorers(gpu, s, rand, &ks, &idx, &on, c);
    try check.expect(fails == 0, "glm-pk8-check: {d} buffers differ", .{fails});
    check.pass("glm-pk8-check: kv8 pooled keys written as kv8.store_row and scored as their bf16 values at {d} cases", .{cases.len + 1});
}

/// tf_glm_pool_keys_pk8 against kv8.quantize_rows of tf_glm_pool_keys' rows, a prompt chunk from position 8.
fn writer(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, ks: *const glm.glue.Kernels) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const pos: usize = 8;
    const rows: usize = 4096;
    const n = (pos + rows) * d;
    const host = try gpu.gpa.alloc(u16, 2 * n + 4 * d);
    defer gpu.gpa.free(host);
    for (host) |*x| x.* = base.bf16(rand.floatNorm(f32));
    const ik = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(host[0..n]));
    const ig = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(host[n .. 2 * n]));
    const ape = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(host[2 * n ..]));
    const p0: i32 = @intCast(pos);
    const at = try base.fromHost(gpu, &bufs, std.mem.asBytes(&p0));
    const pools = (pos + rows) / 4 + 2;
    const pa = try base.zeros(gpu, &bufs, pools * d * 2);
    const pb = try base.zeros(gpu, &bufs, pools * row8);
    for ([2]cuda.Function{ ks.pool_keys, ks.pool_keys_pk8 }, [2]u64{ pa, pb }) |f, pk| {
        var a: cuda.Args = .{};
        inline for (.{ ik, ig, ape, pk, at, @as(i64, @intCast(rows)), @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows / 4 + 2) }, .block = .{ .x = 32 } }, s, &a);
    }
    try s.synchronize();
    const ha = try gpu.gpa.alloc(u16, pools * d);
    defer gpu.gpa.free(ha);
    const hb = try gpu.gpa.alloc(u8, pools * row8);
    defer gpu.gpa.free(hb);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = pa, .len = pools * d * 2, .borrowed = true }).download(0, std.mem.sliceAsBytes(ha));
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = pb, .len = pools * row8, .borrowed = true }).download(0, hb);
    var diff: usize = 0;
    for (pos / 4..(pos + rows) / 4) |p| {
        var want: [row8]u8 = undefined;
        quantize(ha[p * d ..][0..d], &want);
        if (!std.mem.eql(u8, &want, hb[p * row8 ..][0..row8])) diff += 1;
    }
    if (diff != 0) std.debug.print("FAIL pk8 writer: {d} of {d} rows differ from kv8.quantize_rows\n", .{ diff, rows / 4 });
    return @intFromBool(diff != 0);
}

/// Random kv8 rows (slab or shuffled pages), and the same rows as bf16; both scorers in both formats.
fn scorers(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, ks: *const glm.glue.Kernels, idx: *const glm.index.Kernels, on: *const glm.index.Kernels, c: Case) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const gpa = gpu.gpa;
    const np = sp.poolBucket(c.pos, c.rows, c.cap);
    const nb = (sp.scoreColumns(c.pos, c.rows, np) + 63) / 64;
    const qi = try gpa.alloc(u16, c.rows * c.heads * d + c.rows * c.heads);
    defer gpa.free(qi);
    for (qi) |*x| x.* = base.bf16(rand.floatNorm(f32));
    const q = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(qi[0 .. c.rows * c.heads * d]));
    const w = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(qi[c.rows * c.heads * d ..]));
    const pages = (c.cap + 127) / 128;
    const ids = try gpa.alloc(i32, pages);
    defer gpa.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    if (c.paged) rand.shuffle(i32, ids);
    const k8 = try gpa.alloc(u8, pages * 128 * row8);
    defer gpa.free(k8);
    const k16 = try gpa.alloc(u16, pages * 128 * d);
    defer gpa.free(k16);
    var x: [d]u16 = undefined;
    for (0..c.cap) |p| { // slab rows by pool, or row p % 128 of page ids[p / 128]
        const at: usize = @as(usize, @intCast(ids[p / 128])) * 128 + p % 128;
        for (&x) |*v| v.* = base.bf16(rand.floatNorm(f32) * 2);
        quantize(&x, k8[at * row8 ..][0..row8]);
        dequantize(k8[at * row8 ..][0..row8], k16[at * d ..][0..d]);
    }
    const p8 = try base.fromHost(gpu, &bufs, k8);
    const p16 = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(k16));
    const pt: u64 = if (c.paged) try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(ids)) else 0;
    const ps8: i64 = if (c.paged) 128 * row8 else 0;
    const ps16: i64 = if (c.paged) 128 * d * 2 else 0;
    const p0: i32 = @intCast(c.pos);
    const at = try base.fromHost(gpu, &bufs, std.mem.asBytes(&p0));
    const bytes = c.rows * np * 4;
    var out: [4]u64 = undefined;
    for (&out) |*o| o.* = try base.zeros(gpu, &bufs, bytes);
    const scale: f32 = @floatCast(std.math.pow(f64, d, -0.5));
    const loop: usize = if (c.rows >= sp.score_loop_from) sp.score_loop else 1;
    for ([2]cuda.Function{ ks.scores, ks.scores_pk8 }, [2]u64{ p16, p8 }, [2]i64{ ps16, ps8 }, out[0..2]) |f, pk, ps, o| {
        var a: cuda.Args = .{};
        inline for (.{ q, w, @as(i64, @intCast(c.heads)), pk, o, at, @as(i64, @intCast(np)), @as(i64, @intCast(nb)), scale, sp.headScale(c.heads), @as(i64, @intCast(c.heads)), @as(c_int, @intCast(loop)), pt, ps }) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(c.rows), .y = @intCast((nb + loop - 1) / loop) }, .block = .{ .x = 128 } }, s, &a);
    }
    try idx.scores(s, q, w, c.heads, p16, out[2], at, c.rows, np, nb, scale, sp.headScale(c.heads), c.heads, pt, @intCast(ps16));
    try on.scores(s, q, w, c.heads, p8, out[3], at, c.rows, np, nb, scale, sp.headScale(c.heads), c.heads, pt, @intCast(ps8));
    try s.synchronize();
    return try base.same(gpu, "tf_glm_scores kv8", out[0], out[1], bytes, np, c) + try base.same(gpu, "tf_glm_index_scores kv8", out[2], out[3], bytes, np, c);
}
