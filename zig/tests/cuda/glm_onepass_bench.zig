//! glm-onepass-bench (GPU): fused, one-pass and pipeline variants timed at the serve shape; median of 20 launches.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const latent = glm.latent;
const glue = glm.glue;
const op = glm.onepass;

const lw = 512;
const scale: f32 = 0.0625;
const keys = 6144;
const width = 2048; // selected keys a row (every row full), four 512-key chunks
const rows = 4096;
const reps = 20;
const warm = 3;

fn caches(r: *Rig, set: glue.Set) !struct { fp: u64, scales: u64 } {
    const x = try r.bfs(keys * lw, 1.0);
    const fp = try r.zeros(try latent.cacheBytes(keys, lw, true));
    const scales = fp + keys * lw;
    const pos: i32 = 0;
    try latent.latentWriteFormat(set, r.s, try r.dev(std.mem.sliceAsBytes(x)), lw, fp, scales, .{}, try r.dev(std.mem.asBytes(&pos)), keys, lw);
    return .{ .fp = fp, .scales = scales };
}

fn median(times: []f64) f64 {
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

fn timeArm(r: *Rig, t0: cuda.Event, t1: cuda.Event, times: []f64, arm: usize, fused: glue.Set, opf: cuda.Function, pipe: [5]cuda.Function, dq: u64, fp: u64, sc: u64, dtok: u64, dcnt: u64, out: u64, part: latent.Partials, heads: usize) !void {
    for (0..warm + reps) |i| {
        try t0.record(r.s);
        switch (arm) {
            0 => try latent.sparseAttentionFormat(fused, r.s, dq, fp, sc, .{}, dtok, dcnt, width, part, out, rows, heads, lw, scale),
            1 => try op.run(opf, r.s, dq, fp, sc, .{}, dtok, dcnt, width, out, rows, heads, scale),
            else => try op.runPipe(pipe[arm - 2], r.s, dq, fp, sc, .{}, dtok, dcnt, width, out, rows, heads, scale),
        }
        try t1.record(r.s);
        try t1.synchronize();
        if (i >= warm) times[i - warm] = @as(f64, try cuda.Event.elapsedMs(t0, t1));
    }
}

pub fn run(gpu: check.Gpu) !void {
    const major: u32 = @intCast(try gpu.ctx.attribute(.compute_capability_major));
    try check.expect(major >= 9, "glm-onepass-bench: needs sm_90 or later (this GPU: sm_{d}x)", .{major});
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x0bee);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    var opk = try op.Kernels.load(gpu.d, cuda.kernels.glm_onepass, major);
    defer opk.unload();
    var ppk = try op.PipeKernels.load(gpu.d, cuda.kernels.glm_onepass_pipe, major);
    defer ppk.unload();
    const pipe = [5]cuda.Function{ ppk.off, ppk.decode, ppk.scale, ppk.chunk, ppk.exact };
    const fused: glue.Set = .{ .glue = &ks, .mla = .{ .fuse_rows = 1, .skip = true } };
    const c = try caches(&r, .{ .glue = &ks });
    const tok = try r.a.alloc(i32, rows * width);
    for (tok) |*t| t.* = @intCast(r.rng.uintLessThan(usize, keys));
    const cnt = try r.a.alloc(i32, rows);
    for (cnt) |*n| n.* = width;
    const dtok = try r.dev(std.mem.sliceAsBytes(tok));
    const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
    const part: latent.Partials = .{ .po = try r.zeros(16), .pm = try r.zeros(16), .pl = try r.zeros(16) }; // fused ignores these
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    var times: [reps]f64 = undefined;
    const names = [_][]const u8{ "fused (today)", "one-pass", "pipe off", "pipe decode", "pipe scale", "pipe chunk", "pipe exact" };
    for ([_]usize{ 22, 21 }) |heads| {
        const dq = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * lw, 0.5)));
        const out = try r.zeros(rows * heads * lw * 2);
        std.debug.print("  [{d} rows {d} heads {d} keys, median of {d} after {d} warm-ups]\n", .{ rows, heads, width, reps, warm });
        for (names, 0..) |name, arm| {
            try timeArm(&r, t0, t1, &times, arm, fused, opk.f, pipe, dq, c.fp, c.scales, dtok, dcnt, out, part, heads);
            std.debug.print("RESULT glm-onepass-bench {d}h {s}: {d:.4} ms a launch\n", .{ heads, name, median(&times) });
        }
    }
    check.pass("glm-onepass-bench: timed the fused kernel, the one-pass and the pipeline variants at the serve shape", .{});
}
