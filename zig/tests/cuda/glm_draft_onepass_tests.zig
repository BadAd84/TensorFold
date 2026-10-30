//! TF_GLM_OVN_DRAFT_ONEPASS and DOWN4's fp32 tile against q4.matmul's 16-row bf16 launches, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
pub const Gpu = check.Gpu;

const counts = [_]usize{ 1, 15, 16, 17, 40, 127, 128 };

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn shape(gpu: check.Gpu, ks: *const glm.q4.Kernels, s: cuda.Stream, n: usize, k: usize) !void {
    const a = gpu.gpa;
    const rows = glm.q4.many_rows;
    const sz = glm.q4.bytes(n, k);
    var prng = std.Random.DefaultPrng.init(n * 31 + k);
    const r = prng.random();
    const words = try a.alloc(u32, sz[0] / 4);
    defer a.free(words);
    for (words) |*w| w.* = r.int(u32);
    const scales = try a.alloc(u16, sz[1] / 2);
    defer a.free(scales);
    for (scales) |*v| v.* = bf(0.01 + r.float(f32) * 0.02);
    const biases = try a.alloc(u16, sz[2] / 2);
    defer a.free(biases);
    for (biases) |*v| v.* = bf(r.floatNorm(f32) * 0.01);
    const hx = try a.alloc(u16, rows * k);
    defer a.free(hx);
    for (hx) |*v| v.* = bf(r.floatNorm(f32) * 0.25);
    const hxs = try a.alloc(f32, rows * (k / 64));
    defer a.free(hxs);
    for (hxs) |*v| v.* = r.floatNorm(f32);
    var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
    defer w.free();
    var sc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer sc.free();
    var bi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(biases));
    defer bi.free();
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    var xs = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hxs));
    defer xs.free();
    var solo = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
    defer solo.free();
    var many = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
    defer many.free();
    var wide = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
    defer wide.free();
    defer s.synchronize() catch @panic("glm-draft-onepass retains device work until completion");
    const got_solo = try a.alloc(u8, solo.len);
    defer a.free(got_solo);
    const got_many = try a.alloc(u8, many.len);
    defer a.free(got_many);
    const got_wide = try a.alloc(f32, rows * n);
    defer a.free(got_wide);
    const rounded = try a.alloc(u16, rows * n);
    defer a.free(rounded);
    const f32_tile = try ks.loadF32();
    const q: glm.q4.Q4 = .{ .w = w.ptr, .s = sc.ptr, .b = bi.ptr, .n = n, .k = k, .npad = glm.q4.npadOf(n) };
    for (counts) |m| {
        try solo.fill8(0xa5, s.handle);
        try many.fill8(0x5a, s.handle);
        var r0: usize = 0;
        while (r0 < m) : (r0 += 16) {
            try glm.q4.matmul(ks, s, x.ptr + r0 * k * 2, k, xs.ptr + r0 * (k / 64) * 4, q, solo.ptr + r0 * n * 2, @min(16, m - r0));
        }
        try glm.q4.matmulMany(ks, s, x.ptr, k, xs.ptr, q, many.ptr, m);
        try s.synchronize();
        try solo.download(0, got_solo);
        try many.download(0, got_many);
        try check.sameBytes("glm-draft-onepass: one launch against 16-row launches", got_many[0 .. m * n * 2], got_solo[0 .. m * n * 2]);
        for (got_many[m * n * 2 ..]) |byte| try check.expect(byte == 0x5a, "glm-draft-onepass: a row past {d} written", .{m});
        // DOWN4's split down: the fp32 tile's values round (RN) to the bf16 tile's bytes, the same arithmetic a row
        try glm.q4.matmulF32(ks, f32_tile, s, x.ptr, k, xs.ptr, q, wide.ptr, m);
        try s.synchronize();
        try wide.download(0, std.mem.sliceAsBytes(got_wide));
        for (rounded[0 .. m * n], got_wide[0 .. m * n]) |*o, v| o.* = bf(v);
        try check.sameBytes("glm-draft-onepass: the fp32 tile rounded against the bf16 tile", std.mem.sliceAsBytes(rounded[0 .. m * n]), got_many[0 .. m * n * 2]);
    }
}

/// MLPSPLIT's df_sumw: world 2 is df_sum2 byte for byte; world 3 is bf16(((g0 + g1) + g2)) in fp32, round to nearest.
fn sums(gpu: check.Gpu, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const D = 4096;
    const R = 24;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_dflash);
    defer m.unload();
    const sum2 = try m.function("df_sum2");
    const sumw = try m.function("df_sumw");
    var prng = std.Random.DefaultPrng.init(3);
    const g = try a.alloc(f32, 3 * R * D);
    defer a.free(g);
    for (g) |*v| v.* = prng.random().floatNorm(f32) * 3;
    var dg = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(g));
    defer dg.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 2 * R * D * 2);
    defer out.free();
    defer s.synchronize() catch @panic("glm-draft-onepass retains device work until completion");
    const got = try a.alloc(u16, 2 * R * D);
    defer a.free(got);
    const want = try a.alloc(u16, R * D);
    defer a.free(want);
    for ([_]i32{ 2, 3 }) |world| {
        for ([_]cuda.Function{ sum2, sumw }, 0..) |f, i| {
            var args: cuda.Args = .{};
            args.add(dg.ptr);
            args.add(out.ptr + i * R * D * 2);
            args.add(@as(i32, R));
            if (i == 1) args.add(world);
            try cuda.launch.launch(f, .{ .grid = .{ .x = R, .y = D / 1024 }, .block = .{ .x = 256 } }, s, &args);
        }
        try s.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got));
        if (world == 2) {
            try check.sameBytes("glm-draft-onepass: df_sumw at world 2 against df_sum2", std.mem.sliceAsBytes(got[R * D ..]), std.mem.sliceAsBytes(got[0 .. R * D]));
        } else {
            for (want, 0..) |*w, i| w.* = bf((g[i] + g[R * D + i]) + g[2 * R * D + i]);
            try check.sameBytes("glm-draft-onepass: df_sumw at world 3 against ((g0 + g1) + g2)", std.mem.sliceAsBytes(got[R * D ..]), std.mem.sliceAsBytes(want));
        }
    }
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var group = try cuda.Module.load(gpu.d, cuda.kernels.qmm_group);
    defer group.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer ops.unload();
    const ks = try glm.q4.Kernels.load(group, ops, gpu.ctx);
    // a clustered shape (8 K slices), gate/up (one slice), the 4-bit down (4 slices) and MLPSPLIT's down slice (4)
    for ([_][2]usize{ .{ 320, 4096 }, .{ 24576, 4096 }, .{ 4096, 12288 }, .{ 4096, 4096 } }) |dims| try shape(gpu, &ks, s, dims[0], dims[1]);
    try sums(gpu, s);
    check.pass("glm-draft-onepass: 1 to 128 rows in one launch and the fp32 tile, byte-identical to 16-row launches; df_sumw", .{});
}
