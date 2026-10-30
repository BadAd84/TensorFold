//! TF_GLM_OVN_X3_*: GLM's EXL3 decode through every variant and pairing gives the plain path's slot-row bytes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const Ovn = cuda.exl3_experts.Ovn;

const E = 12; // routed experts (the shared one is id E, skipped)
const D = 512; // dims
const slots = glm.exl3.top_k + 1;
const limit: f32 = 7.0;

fn signs(r: *Rig, n: usize, scale: f16) ![]f16 {
    const out = try r.a.alloc(f16, n);
    for (out) |*v| v.* = if (r.rng.boolean()) scale else -scale;
    return out;
}

/// `E` K x N trellises back to back as the loader lays them out: the pointer table, with the base, suh and svh.
fn matrix(r: *Rig, k: usize, n: usize, shared_suh: ?[]const f16, base: *u64, suh: *u64, svh: *u64) !u64 {
    const words = k / 16 * (n / 16) * 64; // i16 an expert
    const tr = try r.a.alloc(i16, E * words);
    for (tr) |*v| v.* = r.rng.int(i16);
    const su = try r.a.alloc(f16, E * k);
    for (0..E) |e| @memcpy(su[e * k ..][0..k], if (shared_suh) |s| s else try signs(r, k, 1.0));
    base.* = try r.dev(std.mem.sliceAsBytes(tr));
    suh.* = try r.dev(std.mem.sliceAsBytes(su));
    svh.* = try r.dev(std.mem.sliceAsBytes(try signs(r, E * n, 0.02)));
    const ptrs = try r.a.alloc(u64, E);
    for (ptrs, 0..) |*ptr, e| ptr.* = base.* + e * words * 2;
    return r.dev(std.mem.sliceAsBytes(ptrs));
}

/// A layer of width `n`; `one`: every gate and up input shares one rotation (the loader's shx).
fn weights(r: *Rig, n: usize, one: bool) !glm.exl3.Experts {
    var w: glm.exl3.Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = one };
    const k2: [E]i32 = @splat(8);
    w.k2 = try r.dev(std.mem.sliceAsBytes(&k2));
    const rot = if (one) try signs(r, D, 1.0) else null;
    w.gt = try matrix(r, D, n, rot, &w.gb, &w.suh_g, &w.svh_g);
    w.ut = try matrix(r, D, n, rot, &w.ub, &w.suh_u, &w.svh_u);
    w.dt = try matrix(r, n, D, null, &w.db, &w.suh_d, &w.svh_d);
    return w;
}

/// Each row's 8 distinct routed picks, then the shared expert's id.
fn picks(r: *Rig, rows: usize) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [E]bool = @splat(false);
        for (0..slots - 1) |j| {
            var e = r.rng.uintLessThan(usize, E);
            while (seen[e]) e = r.rng.uintLessThan(usize, E);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + slots - 1] = E;
    }
    return out;
}

/// The family's decode dispatch with `w.ovn`: ey [rows][slots][D] fp32 bytes, the shared slot left as the canary.
fn decode(r: *Rig, k: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, w: glm.exl3.Experts, n: usize, x: []const u16, pk: []const i32, rows: usize) ![]u8 {
    const P = rows * glm.exl3.top_k;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E, glm.experts.tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros(4),
        .hist = try r.zeros(4),
    };
    const sz = glm.exl3.scratchBytes(rows, D, n, false, w.shx, .{});
    const sc: glm.exl3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
    const ey = try r.zeros(rows * slots * D * 4);
    const xd = try r.zeros(P * n * 2);
    try glm.exl3.run(k, ek, r.s, w, sc, plan, try r.dev(std.mem.sliceAsBytes(x)), D, try r.dev(std.mem.sliceAsBytes(pk)), slots, rows, E, D, n, limit, false, ey, xd);
    return r.back(ey, rows * slots * D * 4);
}

const variants = [_]Ovn{
    .{ .loads = .nc1 },
    .{ .loads = .nc2 },
    .{ .loads = .word },
    .{ .fuse_down = true },
    .{ .xrow = true },
    .{ .fuse_down = true, .xrow = true },
    .{ .loads = .nc2, .fuse_down = true },
    .{ .loads = .nc2, .xrow = true },
    .{ .loads = .word, .fuse_down = true, .xrow = true },
    .{ .loads = .nc1, .fuse_down = true, .xrow = true },
    .{ .loads = .nc2, .fuse_down = true, .xrow = true },
};

pub fn run_(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(29);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }

    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try glm.exl3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ek.unload();
    var runs: usize = 0;
    for ([_]usize{ 256, 640 }) |n| for ([_]bool{ true, false }) |one| { // 640: a TP=3 rank's 5 blocks, 10 down steps
        var w = try weights(&r, n, one);
        for ([_]usize{ 1, 2, 3, 8, 17 }) |rows| { // decode windows; 17 rows take two member tiles
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            w.ovn = .{};
            const want = try decode(&r, &k, &ek, w, n, x, pk, rows);
            try check.expect(!std.mem.allEqual(u8, want[0 .. (slots - 1) * D * 4], 0xff), "glm-exl3-ovn: the plain decode wrote row 0's routed slots", .{});
            for (variants) |o| {
                w.ovn = o;
                const got = try decode(&r, &k, &ek, w, n, x, pk, rows);
                const name = try std.fmt.allocPrint(r.a, "glm-exl3-ovn width {d}, one rotation {any}, {d} rows, loads {t}, fuse {any}, xrow {any}", .{ n, one, rows, o.loads, o.fuse_down, o.xrow });
                try check.sameBytes(name, got, want);
                runs += 1;
            }
        }
    };
    check.pass("glm-exl3-ovn: {d} variant runs equal the plain decode byte for byte", .{runs});
}
