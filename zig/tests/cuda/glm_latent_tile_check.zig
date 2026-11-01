//! glm-latent-tile-check / -bench (GPU): tile absorb/expand vs today's byte for byte, control red, then timed.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const latent = glm.latent;
const glue = glm.glue;

const d = 256; // qk and v head widths
const lw = 512; // latent width
const nan: u16 = 0x7fc0; // bf16 NaN
const ninf: u16 = 0xff80; // bf16 -inf

/// `got` and `base` are not equal: the reversed-butterfly control must move at least one output, or the check is blind.
fn differs(what: []const u8, got: []const u8, base: []const u8) !void {
    if (!std.mem.eql(u8, got, base)) return;
    std.debug.print("FAIL {s}: control equals the base byte for byte (the check cannot see a wrong tree)\n", .{what});
    return error.TestFailed;
}

/// One (rows, heads): absorb and expand tiles equal the base, the controls differ. NaN and -inf travel alike.
fn shape(r: *Rig, ks: *const glue.Kernels, rows: usize, heads: usize) !void {
    const off: glue.Set = .{ .glue = ks };
    const tile: glue.Set = .{ .glue = ks, .prompt = true, .mla = .{ .tile = true } };
    var name: [96]u8 = undefined;
    // absorb: qa [rows][heads][lw] = q [rows][heads][d] . wk [heads][d][lw]
    const q = try r.bfs(rows * heads * d, 1.0);
    q[0] = nan;
    if (rows > 1) q[heads * d + 7] = ninf; // -inf in row 1's chain, a different reduction than row 0's NaN
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const wk = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
    const ab = try r.zeros(rows * heads * lw * 2);
    const at = try r.zeros(rows * heads * lw * 2);
    const ar = try r.zeros(rows * heads * lw * 2);
    try latent.absorbQ(off, r.s, dq, wk, 0, ab, rows, heads, d, lw);
    try latent.absorbQ(tile, r.s, dq, wk, 0, at, rows, heads, d, lw);
    try latent.tileBlocks(ks.absorb_tile_red, r.s, .{ dq, wk, ar }, rows, heads, lw / 32);
    const base_a = try r.back(ab, rows * heads * lw * 2);
    try check.sameBytes(try std.fmt.bufPrint(&name, "absorb tile {d} rows x {d} heads", .{ rows, heads }), try r.back(at, rows * heads * lw * 2), base_a);
    try differs(try std.fmt.bufPrint(&name, "absorb red control {d}x{d}", .{ rows, heads }), try r.back(ar, rows * heads * lw * 2), base_a);
    // expand: out [rows][heads][d] = o_lat [rows][heads][lw] . wv [heads][d][lw]
    const o = try r.bfs(rows * heads * lw, 1.0);
    o[1] = nan;
    if (rows > 1) o[heads * lw + 3] = ninf;
    const dol = try r.dev(std.mem.sliceAsBytes(o));
    const wv = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
    const eb = try r.zeros(rows * heads * d * 2);
    const et = try r.zeros(rows * heads * d * 2);
    const er = try r.zeros(rows * heads * d * 2);
    try latent.expandV(off, r.s, dol, wv, 0, eb, rows, heads, d, lw);
    try latent.expandV(tile, r.s, dol, wv, 0, et, rows, heads, d, lw);
    try latent.tileBlocks(ks.expand_tile_red, r.s, .{ dol, wv, er }, rows, heads, d / 16);
    const base_e = try r.back(eb, rows * heads * d * 2);
    try check.sameBytes(try std.fmt.bufPrint(&name, "expand tile {d} rows x {d} heads", .{ rows, heads }), try r.back(et, rows * heads * d * 2), base_e);
    try differs(try std.fmt.bufPrint(&name, "expand red control {d}x{d}", .{ rows, heads }), try r.back(er, rows * heads * d * 2), base_e);
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x11e);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    for ([_]usize{ 22, 21, 16 }) |heads| try shape(&r, &ks, 4096, heads); // the serve shape at the three head blocks
    try shape(&r, &ks, 4000, 22); // a ragged last row block
    try shape(&r, &ks, 37, 22); // a small prompt
    check.pass("glm-latent-tile-check: tile absorb/expand equal the base byte for byte at every shape, controls differ", .{});
}

const warm = 3;
const reps = 20;
const Op = enum { absorb, expand };

fn one(set: glue.Set, s: cuda.Stream, op: Op, in0: u64, w: u64, out: u64, rows: usize, heads: usize) !void {
    switch (op) {
        .absorb => try latent.absorbQ(set, s, in0, w, 0, out, rows, heads, d, lw),
        .expand => try latent.expandV(set, s, in0, w, 0, out, rows, heads, d, lw),
    }
}

/// The median launch over `reps` after `warm` warm-ups, through CUDA events (ms a launch).
fn timeMs(s: cuda.Stream, t0: *cuda.Event, t1: *cuda.Event, ms: []f64, set: glue.Set, op: Op, in0: u64, w: u64, out: u64, rows: usize, heads: usize) !f64 {
    for (0..warm) |_| try one(set, s, op, in0, w, out, rows, heads);
    try s.synchronize();
    for (ms) |*m| {
        try t0.record(s);
        try one(set, s, op, in0, w, out, rows, heads);
        try t1.record(s);
        try t1.synchronize();
        m.* = @as(f64, try cuda.Event.elapsedMs(t0.*, t1.*));
    }
    return check.median(ms);
}

pub fn bench(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a);
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    const ms = try a.alloc(f64, reps);
    const base: glue.Set = .{ .glue = &ks, .prompt = true };
    const vec: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .vec = true } };
    const tile: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .tile = true } };
    const rows = 4096;
    for ([_]usize{ 22, 21 }) |heads| {
        const dq = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * d, 1.0)));
        const dol = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * lw, 1.0)));
        const wk = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
        const wv = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
        const qa = try r.zeros(rows * heads * lw * 2);
        const ov = try r.zeros(rows * heads * d * 2);
        const ab = try timeMs(s, &t0, &t1, ms, base, .absorb, dq, wk, qa, rows, heads);
        const av = try timeMs(s, &t0, &t1, ms, vec, .absorb, dq, wk, qa, rows, heads);
        const atl = try timeMs(s, &t0, &t1, ms, tile, .absorb, dq, wk, qa, rows, heads);
        std.debug.print("RESULT absorb {d} rows x {d} heads: base {d:.3} vec {d:.3} tile {d:.3} ms/launch\n", .{ rows, heads, ab, av, atl });
        const eb = try timeMs(s, &t0, &t1, ms, base, .expand, dol, wv, ov, rows, heads);
        const ev = try timeMs(s, &t0, &t1, ms, vec, .expand, dol, wv, ov, rows, heads);
        const et = try timeMs(s, &t0, &t1, ms, tile, .expand, dol, wv, ov, rows, heads);
        std.debug.print("RESULT expand {d} rows x {d} heads: base {d:.3} vec {d:.3} tile {d:.3} ms/launch\n", .{ rows, heads, eb, ev, et });
    }
    check.pass("glm-latent-tile-bench: base, vec and tile timed for absorb and expand at 4096 rows x 22 and 21 heads", .{});
}
