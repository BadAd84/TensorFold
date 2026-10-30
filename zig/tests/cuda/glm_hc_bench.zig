//! glm-hc-bench: hc_pre's mixing partials (today's, TF_GLM_OVN_HC_ROWS=16/32, _HC_DEC) and the GLUE_ROW forms by rows.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const glue = glm.glue;

const d = 4096;
const s = 4;
const wide = s * d;
const world = 3;
const most = 4096;
const rows_list = [_]usize{ 1, 2, 4, 8, 16, 32, 63, 64, 341, 683, 1366, most };
const reps = 50;
const flush_bytes = 256 << 20; // past GB10's L2: written before each cold launch

/// The buffers every form reads and writes, sized for `most` rows.
const Bufs = struct { x: u64, g: u64, fnw: u64, base: u64, scale: u64, nw: u64, out: u64, xs: u64, post: u64, comb: u64, part: u64 };

fn filled(gpu: Gpu, list: *std.ArrayList(cuda.DeviceBuffer), host: []const u8) !u64 {
    try list.append(gpu.gpa, try cuda.DeviceBuffer.fromHost(gpu.d, host));
    return list.items[list.items.len - 1].ptr;
}

/// `n` values: bf16 bits of uniform [-scale, scale) (bf16) or the constant `scale` (f32).
fn values(a: std.mem.Allocator, rng: std.Random, comptime T: type, n: usize, scale: f32) ![]T {
    const v = try a.alloc(T, n);
    for (v) |*o| o.* = if (T == u16) @truncate(@as(u32, @bitCast((rng.float(f32) * 2 - 1) * scale)) >> 16) else scale;
    return v;
}

/// One form's GPU ms a call at `rows`: warm (reps back to back) and cold (L2 overwritten before each call).
fn time(st: cuda.Stream, flush: cuda.DeviceBuffer, ev: [2]cuda.Event, ctx: anytype, rows: usize) ![2]f64 {
    try ctx.call(st, rows); // first launch: module load and caches out of the timing
    try ev[0].record(st);
    for (0..reps) |_| try ctx.call(st, rows);
    try ev[1].record(st);
    try ev[1].synchronize();
    const warm = @as(f64, try cuda.Event.elapsedMs(ev[0], ev[1])) / reps;
    var cold: f64 = 0;
    for (0..reps) |_| {
        try flush.fill8(0x5a, st.handle);
        try ev[0].record(st);
        try ctx.call(st, rows);
        try ev[1].record(st);
        try ev[1].synchronize();
        cold += try cuda.Event.elapsedMs(ev[0], ev[1]);
    }
    return .{ warm, cold / reps };
}

/// hc_pre (partials, then hc_finish) on `set`.
const Pre = struct {
    set: glue.Set,
    b: Bufs,
    fn call(p: Pre, st: cuda.Stream, rows: usize) !void {
        try glue.hcPre(p.set, st, p.b.x, p.b.fnw, p.b.base, p.b.scale, .{ .bf16 = p.b.nw }, p.b.out, p.b.xs, p.b.post, p.b.comb, p.b.part, 1e-6, 1e-6, 20, rows, d, s);
    }
};

/// The prompt glue: hcPost then hcPre (`row` false), or hcRow (GLUE_ROW; `set` picks rows4, 16 or 32 a block).
const Glue = struct {
    set: glue.Set,
    b: Bufs,
    row: bool,
    fn call(p: Glue, st: cuda.Stream, rows: usize) !void {
        const b = p.b;
        if (p.row) return glue.hcRow(p.set, st, b.x, b.g, b.fnw, b.base, b.scale, .{ .bf16 = b.nw }, b.out, b.xs, b.post, b.comb, 1e-6, 1e-6, 20, rows, d, s, world);
        try glue.hcPost(p.set, st, b.x, b.x, b.g, b.post, b.comb, rows, d, s, world);
        try (Pre{ .set = p.set, .b = b }).call(st, rows);
    }
};

pub fn run(gpu: Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var st = try cuda.Stream.init(gpu.d, false);
    defer st.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    var off = try glue.Kernels.load(mod);
    try glue.hcLoad(&off, mod, 0, false);
    var k16 = off;
    try glue.hcLoad(&k16, mod, 16, true);
    var k32 = off;
    try glue.hcLoad(&k32, mod, 32, false);
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    var list: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (list.items) |*x| x.free();
        list.deinit(gpu.gpa);
    }
    const b: Bufs = .{ // comb 0.25 (each column sums to 1) and post 0.5 keep repeated in-place glue finite
        .x = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, u16, most * wide, 1.0))),
        .g = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, world * most * d, 0.01))),
        .fnw = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, u16, 24 * wide, 0.02))),
        .base = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, 24, 0.1))),
        .scale = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, 3, 0.9))),
        .nw = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, u16, d, 1.0))),
        .out = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, u16, most * d, 0))),
        .xs = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, most * (d / 64), 0))),
        .post = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, most * s, 0.5))),
        .comb = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, most * s * s, 0.25))),
        .part = try filled(gpu, &list, std.mem.sliceAsBytes(try values(a, rng, f32, most * glue.hc_blocks * 32, 0))),
    };
    var flush = try cuda.DeviceBuffer.alloc(gpu.d, flush_bytes);
    defer flush.free();
    var ev = [2]cuda.Event{ try cuda.Event.init(gpu.d, true), try cuda.Event.init(gpu.d, true) };
    defer for (&ev) |*e| e.deinit();
    const pres = [_]struct { []const u8, Pre, usize }{
        .{ "today (hc_partial8 / hc_partial)", .{ .set = .{ .glue = &off }, .b = b }, most },
        .{ "HC_ROWS=16", .{ .set = .{ .glue = &k16, .prompt = true }, .b = b }, most },
        .{ "HC_ROWS=32", .{ .set = .{ .glue = &k32, .prompt = true }, .b = b }, most },
        .{ "HC_DEC (hc_partial48)", .{ .set = .{ .glue = &k16 }, .b = b }, 63 },
    };
    const glues = [_]struct { []const u8, Glue }{
        .{ "hc_post + hc_pre", .{ .set = .{ .glue = &off, .prompt = true }, .b = b, .row = false } },
        .{ "hc_post + hc_pre, HC_ROWS=32", .{ .set = .{ .glue = &k32, .prompt = true }, .b = b, .row = false } },
        .{ "GLUE_ROW (rows4)", .{ .set = .{ .glue = &off, .prompt = true }, .b = b, .row = true } },
        .{ "GLUE_ROW + HC_ROWS=16", .{ .set = .{ .glue = &k16, .prompt = true }, .b = b, .row = true } },
        .{ "GLUE_ROW + HC_ROWS=32", .{ .set = .{ .glue = &k32, .prompt = true }, .b = b, .row = true } },
    };
    for (rows_list) |rows| {
        for (pres) |p| if (rows <= p[2]) {
            const ms = try time(st, flush, ev, p[1], rows);
            std.debug.print("RESULT hc_pre {s} rows {d}: warm {d:.4} ms cold {d:.4} ms\n", .{ p[0], rows, ms[0], ms[1] });
        };
        if (rows >= 64) for (glues) |g| {
            const ms = try time(st, flush, ev, g[1], rows);
            std.debug.print("RESULT glue {s} rows {d}: warm {d:.4} ms cold {d:.4} ms\n", .{ g[0], rows, ms[0], ms[1] });
        };
    }
    check.pass("glm-hc-bench: every form timed at 1 to {d} rows (hc_pre includes hc_finish, the same in every form)", .{most});
}
