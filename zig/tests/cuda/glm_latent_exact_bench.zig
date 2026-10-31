//! glm-latent-exact-bench (GPU): MLA prompt absorb/expand at the serve shapes; median of 20, one RESULT line a variant.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const glue = glm.glue;

const lw = 512;
const d = 256;
const rows = 4096;
const warm = 3;
const reps = 20;

const Ev = struct { t0: cuda.Event, t1: cuda.Event };

/// A rowBlocks launch (tf_glm_absorb_q and every CUDA-core form share its grid and arguments).
fn rowLaunch(f: cuda.Function, s: cuda.Stream, x: u64, w: u64, out: u64, heads: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, w, out }) |v| a.add(@as(u64, v));
    inline for (.{ @as(i64, rows), @as(i64, @intCast(heads)), @as(i64, 64) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = 16, .z = rows / 64 }, .block = .{ .x = 128 } }, s, &a);
}

/// The tensor-core launch (mmaBlocks): grid (n / 128, heads, rows / 64), dynamic shared, no rb argument.
fn mmaLaunch(f: cuda.Function, s: cuda.Stream, x: u64, w: u64, out: u64, heads: usize, n: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(w);
    a.add(out);
    a.add(@as(i64, rows));
    a.add(@as(i64, @intCast(heads)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n / 128), .y = @intCast(heads), .z = rows / 64 }, .block = .{ .x = 128 }, .shared = glue.latent_mma_smem }, s, &a);
}

fn median20(ev: *Ev, s: cuda.Stream, comptime one: anytype, ctx: anytype) !f64 {
    for (0..warm) |_| try @call(.auto, one, ctx);
    var ms: [reps]f64 = undefined;
    for (&ms) |*m| {
        try ev.t0.record(s);
        try @call(.auto, one, ctx);
        try ev.t1.record(s);
        try ev.t1.synchronize();
        m.* = @as(f64, try cuda.Event.elapsedMs(ev.t0, ev.t1));
    }
    return check.median(&ms);
}

const Row = struct { f: cuda.Function, s: cuda.Stream, x: u64, w: u64, out: u64, heads: usize };
fn rowOne(c: Row) !void {
    return rowLaunch(c.f, c.s, c.x, c.w, c.out, c.heads);
}
const Mma = struct { f: cuda.Function, s: cuda.Stream, x: u64, w: u64, out: u64, heads: usize, n: usize };
fn mmaOne(c: Mma) !void {
    return mmaLaunch(c.f, c.s, c.x, c.w, c.out, c.heads, c.n);
}

/// absorb (n = lw) or expand (n = d) at `heads`: one RESULT line per variant, milliseconds a launch.
fn direction(gpu: check.Gpu, ev: *Ev, s: cuda.Stream, ks: glue.Kernels, mod: cuda.Module, heads: usize, absorb: bool) !void {
    const k: usize = if (absorb) d else lw;
    const n: usize = if (absorb) lw else d;
    const pre = if (absorb) "absorb" else "expand";
    var xb = try cuda.DeviceBuffer.alloc(gpu.d, rows * heads * k * 2);
    defer xb.free();
    var wb = try cuda.DeviceBuffer.alloc(gpu.d, heads * d * lw * 2);
    defer wb.free();
    var ob = try cuda.DeviceBuffer.alloc(gpu.d, rows * heads * n * 2);
    defer ob.free();
    const base = if (absorb) ks.absorb_q else ks.expand_v;
    const vec = if (absorb) ks.absorb_q_vec else ks.expand_v_vec;
    const exact = if (absorb) ks.absorb_exact else ks.expand_exact;
    const mma = if (absorb) ks.absorb_mma else ks.expand_mma;
    const ord = try mod.function(if (absorb) "tf_glm_absorb_exact_order" else "tf_glm_expand_exact_order");
    const tre = try mod.function(if (absorb) "tf_glm_absorb_exact_tree" else "tf_glm_expand_exact_tree");
    const rc: Row = .{ .f = base, .s = s, .x = xb.ptr, .w = wb.ptr, .out = ob.ptr, .heads = heads };
    const names = [_][]const u8{ "base", "vec", "exact", "order(byte-changing)", "tree(byte-changing)" };
    const fns = [_]cuda.Function{ base, vec, exact, ord, tre };
    for (names, fns) |nm, f| {
        const t = try median20(ev, s, rowOne, .{Row{ .f = f, .s = s, .x = xb.ptr, .w = wb.ptr, .out = ob.ptr, .heads = heads }});
        std.debug.print("RESULT {s} {d}h {s}: {d:.4} ms\n", .{ pre, heads, nm, t });
    }
    _ = rc;
    const tm = try median20(ev, s, mmaOne, .{Mma{ .f = mma, .s = s, .x = xb.ptr, .w = wb.ptr, .out = ob.ptr, .heads = heads, .n = n }});
    std.debug.print("RESULT {s} {d}h mma/LATENT_MMA(byte-changing): {d:.4} ms\n", .{ pre, heads, tm });
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    var ev: Ev = .{ .t0 = try cuda.Event.init(gpu.d, true), .t1 = try cuda.Event.init(gpu.d, true) };
    defer ev.t0.deinit();
    defer ev.t1.deinit();
    for ([_]usize{ 22, 21 }) |heads| {
        try direction(gpu, &ev, s, ks, mod, heads, true);
        try direction(gpu, &ev, s, ks, mod, heads, false);
    }
    check.pass("glm-latent-exact-bench: base, vec, exact, order, tree and LATENT_MMA timed at 4,096 rows, 22 and 21 heads", .{});
}
