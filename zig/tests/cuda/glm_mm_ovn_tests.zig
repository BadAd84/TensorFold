//! TF_GLM_OVN_MM_WIDE, MM_CLUSTER, MM_DECTILE, SWIGLU_REDUCE and ROUTER16 against today's kernels, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
pub const Gpu = check.Gpu;

const marker = 0xa5;

const Rig = struct {
    gpu: Gpu,
    s: cuda.Stream,
    mk: glm.mm.Kernels,
    glue: *const glm.glue.Kernels,
    rng: std.Random,
};

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// `n` bf16 values in [-scale, scale] from a hash (cheap at tens of millions in a debug build).
fn values(r: *Rig, n: usize, scale: f32) ![]u16 {
    const out = try r.gpu.gpa.alloc(u16, n);
    const seed = r.rng.int(u32);
    for (out, 0..) |*v, i| {
        const h = @as(u32, @truncate(i *% 0x9E3779B97F4A7C15 >> 23)) ^ seed;
        v.* = bf((@as(f32, @floatFromInt(h % 4001)) / 2000 - 1) * scale);
    }
    return out;
}

/// A weight [n][k] tiled as the loader tiles it: random bf16, or random codes with fp16 scales.
const Weight = struct {
    codes: cuda.DeviceBuffer,
    scales: cuda.DeviceBuffer,
    t: glm.mm.Tiled,

    fn deinit(w: *Weight) void {
        w.codes.free();
        w.scales.free();
    }
};

fn weight(r: *Rig, bits: u8, n: usize, k: usize) !Weight {
    const a = r.gpu.gpa;
    if (bits == 16) {
        const host = try values(r, n * k, 0.05);
        defer a.free(host);
        var src = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        defer src.free();
        var out: Weight = .{ .codes = try cuda.DeviceBuffer.alloc(r.gpu.d, n * k * 2), .scales = try cuda.DeviceBuffer.alloc(r.gpu.d, 16), .t = undefined };
        try glm.mm.tile(&r.mk, r.s, src.ptr, out.codes.ptr, n, k);
        try r.s.synchronize();
        out.t = .{ .w = out.codes.ptr, .n = n, .k = k };
        return out;
    }
    const words = try a.alloc(u32, n * k * bits / 32); // every bit pattern is a valid code vector
    defer a.free(words);
    const seed = r.rng.int(u32);
    for (words, 0..) |*v, i| v.* = @as(u32, @truncate(i *% 0x9E3779B97F4A7C15 >> 19)) ^ seed;
    const hs = try a.alloc(u16, n * (k / 64));
    defer a.free(hs);
    for (hs) |*v| v.* = @bitCast(@as(f16, @floatCast(0.002 + r.rng.float(f32) * 0.01)));
    var src = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(words));
    defer src.free();
    var ss = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(hs));
    defer ss.free();
    var out: Weight = .{ .codes = try cuda.DeviceBuffer.alloc(r.gpu.d, n * k * bits / 8), .scales = try cuda.DeviceBuffer.alloc(r.gpu.d, hs.len * 2), .t = undefined };
    try glm.mm.tilePacked(&r.mk, r.s, bits, src.ptr, ss.ptr, out.codes.ptr, out.scales.ptr, n, k);
    try r.s.synchronize();
    out.t = .{ .w = out.codes.ptr, .s = out.scales.ptr, .n = n, .k = k, .bits = bits };
    return out;
}

/// Both whole buffers equal (filled with the marker first: nothing written past what today's kernels write).
fn same(r: *Rig, what: []const u8, got: cuda.DeviceBuffer, want: cuda.DeviceBuffer) !void {
    try r.s.synchronize();
    const a = r.gpu.gpa;
    const g = try a.alloc(u8, got.len);
    defer a.free(g);
    const w = try a.alloc(u8, want.len);
    defer a.free(w);
    try got.download(0, g);
    try want.download(0, w);
    try check.sameBytes(what, g, w);
}

/// `rows` rows of x by `w` through `base` and `on` (one: prompt blocks), both outputs equal at fp32 and bf16.
fn compare(r: *Rig, what: []const u8, on: *const glm.mm.Kernels, w: glm.mm.Tiled, rows: usize, one: bool) !void {
    const stride = w.k + 8;
    const host = try values(r, rows * stride, 1.0);
    defer r.gpu.gpa.free(host);
    var x = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
    defer x.free();
    var part = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(16, glm.mm.partBytes(rows, w)));
    defer part.free();
    var a = try cuda.DeviceBuffer.alloc(r.gpu.d, rows * w.n * 4);
    defer a.free();
    var b = try cuda.DeviceBuffer.alloc(r.gpu.d, a.len);
    defer b.free();
    for ([_]bool{ false, true }) |wd| {
        try a.fill8(marker, r.s.handle);
        try b.fill8(marker, r.s.handle);
        try glm.mm.matmul(&r.mk, r.s, x.ptr, stride, w, a.ptr, part.ptr, rows, wd, one);
        try glm.mm.matmul(on, r.s, x.ptr, stride, w, b.ptr, part.ptr, rows, wd, one);
        try same(r, what, b, a);
    }
}

const Shape = struct { n: usize, k: usize };

fn wide(r: *Rig, m: cuda.Module, probe: cuda.Function) !void {
    // ragged columns and rows; K 8192 takes two L2 row bands of 6 at 12 row tiles (tile_of's tail)
    const shapes = [_]Shape{ .{ .n = 1032, .k = 8192 }, .{ .n = 3080, .k = 1024 } };
    const rows = [_]usize{ 1450, 600 };
    for ([_]u64{ 1, 2 }) |variant| {
        var on = r.mk;
        on.ovn = try glm.mm.Ovn.choose(m, probe, .{ .wide = variant });
        for ([_]u8{ 16, 4, 6, 8 }) |bits| for (shapes, rows) |sh, n_rows| {
            var w = try weight(r, bits, sh.n, sh.k);
            defer w.deinit();
            try compare(r, "glm-mm-ovn: MM_WIDE prompt against glm_mm's prompt blocks", &on, w.t, n_rows, true);
        };
    }
    check.pass("glm-mm-ovn: MM_WIDE (64 x 32 and 32 x 32 warps), bf16/4/6/8 bits, ragged rows and columns, two bands", .{});
}

fn decode(r: *Rig, m: cuda.Module, probe: cuda.Function) !void {
    const shapes = [_]Shape{ .{ .n = 1032, .k = 8192 }, .{ .n = 3080, .k = 1024 }, .{ .n = 4096, .k = 704 }, .{ .n = 1408, .k = 4096 } };
    const counts = [_]usize{ 1, 2, 3, 16, 17, 32, 33, 64, 65, 100 };
    const clusters = try probe.attribute(.binary_version) >= 90;
    // every 16-row window on 32-column tiles; then (sm_90 on) every sliced window in a cluster
    const choices = [_]glm.mm.Ovn.Choice{ .{ .dectile_below = 1 << 30 }, .{ .cluster_from = 0 } };
    for (choices[0..if (clusters) 2 else 1]) |c| {
        var on = r.mk;
        on.ovn = try glm.mm.Ovn.choose(m, probe, c);
        for ([_]u8{ 16, 4, 6, 8 }) |bits| for (shapes) |sh| {
            var w = try weight(r, bits, sh.n, sh.k);
            defer w.deinit();
            for (counts) |rows| try compare(r, "glm-mm-ovn: MM_DECTILE / MM_CLUSTER decode windows against today's", &on, w.t, rows, false);
        };
    }
    check.pass("glm-mm-ovn: MM_DECTILE{s}, bf16/4/6/8 bits, every bucket, sliced and one-slice shapes", .{if (clusters) " and MM_CLUSTER" else " (no clusters before sm_90)"});
}

fn swiglu(r: *Rig, m: cuda.Module, probe: cuda.Function) !void {
    const a = r.gpu.gpa;
    var on = r.mk;
    on.ovn = try glm.mm.Ovn.choose(m, probe, .{ .swiglu = true });
    const set: glm.glue.Set = .{ .glue = r.glue, .mm = &r.mk };
    const limit: f32 = 7.0;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 704, 640, 2048 }) |width| {
        const k: usize = 4096;
        var w = try weight(r, bits, 2 * width, k);
        defer w.deinit();
        for ([_]usize{ 1, 2, 3, 16, 17, 64, 100 }) |rows| {
            const host = try values(r, rows * k, 4.0); // large enough that gate and up clip at the limit
            defer a.free(host);
            var x = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
            defer x.free();
            var part = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(16, glm.mm.partBytes(rows, w.t)));
            defer part.free();
            var gu = try cuda.DeviceBuffer.alloc(r.gpu.d, rows * 2 * width * 2);
            defer gu.free();
            var xs = try cuda.DeviceBuffer.alloc(r.gpu.d, rows * (width / 64) * 4);
            defer xs.free();
            var want = try cuda.DeviceBuffer.alloc(r.gpu.d, rows * width * 2);
            defer want.free();
            var got = try cuda.DeviceBuffer.alloc(r.gpu.d, want.len);
            defer got.free();
            try want.fill8(marker, r.s.handle);
            try got.fill8(marker, r.s.handle);
            try glm.mm.matmul(&r.mk, r.s, x.ptr, k, w.t, gu.ptr, part.ptr, rows, false, false);
            try glm.glue.swiglu(set, r.s, gu.ptr, want.ptr, xs.ptr, limit, rows, width);
            try check.expect(try glm.mm.swigluReduce(&on, r.s, x.ptr, k, w.t, got.ptr, part.ptr, rows, limit, false), "glm-mm-ovn: SWIGLU_REDUCE declined a sliced decode gate/up", .{});
            try same(r, "glm-mm-ovn: SWIGLU_REDUCE against glm_mm_reduce + swiglu", got, want);
        }
    };
    check.pass("glm-mm-ovn: SWIGLU_REDUCE, bf16/4/6/8 bits, widths 704/640/2048, 1-100 rows", .{});
}

fn router(r: *Rig, gm: cuda.Module) !void {
    const a = r.gpu.gpa;
    const d: usize = 4096;
    const ne: usize = 256;
    const hw = try values(r, ne * d, 0.05);
    defer a.free(hw);
    var w = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(hw));
    defer w.free();
    for ([_][:0]const u8{ "tf_glm_router16_s3", "tf_glm_router16_s4" }) |name| {
        var ks = r.glue.*;
        ks.router16 = try gm.function(name);
        const base: glm.glue.Set = .{ .glue = r.glue };
        const on: glm.glue.Set = .{ .glue = &ks };
        for (1..17) |rows| {
            const stride = d + 8;
            const hx = try values(r, rows * stride, 1.0);
            defer a.free(hx);
            var x = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(hx));
            defer x.free();
            var part = try cuda.DeviceBuffer.alloc(r.gpu.d, glm.glue.router_ks * rows * ne * 4);
            defer part.free();
            var want = try cuda.DeviceBuffer.alloc(r.gpu.d, rows * ne * 4);
            defer want.free();
            var got = try cuda.DeviceBuffer.alloc(r.gpu.d, want.len);
            defer got.free();
            try glm.glue.router(base, r.s, x.ptr, stride, w.ptr, want.ptr, part.ptr, rows, d, ne);
            try glm.glue.router(on, r.s, x.ptr, stride, w.ptr, got.ptr, part.ptr, rows, d, ne);
            try same(r, "glm-mm-ovn: ROUTER16 against tf_glm_router_tile", got, want);
        }
    }
    check.pass("glm-mm-ovn: ROUTER16 (3 and 4 stages), 1-16 rows, 256 experts", .{});
}

pub fn run(gpu: Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer m.unload();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    var glue = try glm.glue.Kernels.load(gm);
    glue.router16 = null; // today's router tile, whatever the environment sets
    var prng = std.Random.DefaultPrng.init(29);
    var r: Rig = .{ .gpu = gpu, .s = s, .mk = try glm.mm.Kernels.load(m), .glue = &glue, .rng = prng.random() };
    r.mk.ovn = .{}; // today's kernels, whatever the environment sets
    try wide(&r, m, r.mk.reduce);
    try decode(&r, m, r.mk.reduce);
    try swiglu(&r, m, r.mk.reduce);
    try router(&r, gm);
    check.pass("glm-mm-ovn: every knob's kernels equal today's byte for byte", .{});
}
