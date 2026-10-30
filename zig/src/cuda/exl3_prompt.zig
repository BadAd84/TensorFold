//! The common EXL3 prompt path, after @BobClawblaw's #976 design and MiaAI-Lab's transposed kernels.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const launch = @import("launch.zig");
const grouped = @import("grouped.zig");
const knob = @import("ovn.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

pub const tile = 64;
pub const Scratch = struct { xg: u64, xu: u64, xd: u64, y: u64, rows: usize, slots: usize, order: u64 = 0 };

/// The speed-hunt switches (`TF_GLM_OVN_<name>`), all off by default; off, every launch below is the one it always was.
pub const Ovn = struct {
    ey_direct: bool = false, // EY_DIRECT: the down kernel writes the caller's slot rows (`Out`), no `y` scratch
    k64: u8 = 0, // X3_PROMPT_K64: 1 ots0106's 64-k stages, 2 plus ots0127's hoisted addressing and scales
    order: bool = false, // X3_PROMPT_ORDER: production's item order (single- and multi-item experts interleaved)
    ybf16: bool = false, // X3_YBF16, a numerics trade: the down kernel's slot rows in bf16

    pub fn fromEnv() !Ovn {
        const k64 = try knob.int("X3_PROMPT_K64", 0);
        if (k64 > 2) return error.BadOvnKnob;
        return .{ .ey_direct = knob.on("EY_DIRECT"), .k64 = @intCast(k64), .order = knob.on("X3_PROMPT_ORDER"), .ybf16 = knob.on("X3_YBF16") };
    }

    fn any(o: Ovn) bool {
        return o.ey_direct or o.k64 != 0 or o.order or o.ybf16;
    }
};

/// Where the down kernel writes pair p: row p / slots of `y`, slot p % slots, `slots` slot rows a row.
pub const Out = struct { y: u64, slots: usize };

/// The 64-k kernels' dynamic shared bytes (exl3_prompt_ovn.cuh's ksmem_bytes, asserted there).
const k64_shared: u32 = 48 * 1024;

pub fn eligible(l: ex.Layer) bool {
    return l.codebook == .mcg and l.uniform_k2 == 8 and l.gu_stride == 0 and l.dims > 0 and l.width > 0 and l.dims % 256 == 0 and l.width % 128 == 0 and (l.act_mode == 0 or l.act_mode == 1);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.Exl3Shape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.Exl3Shape;
}
fn aligned(n: usize) !usize {
    return (try add(n, 255)) / 256 * 256;
}
fn sizes(l: ex.Layer, rows: usize, slots: usize, o: Ovn) ![5]usize {
    if (!eligible(l) or rows == 0 or slots == 0 or slots > 32 or l.experts == 0 or l.experts > grouped.max_experts) return error.Exl3Shape;
    const pairs = try mul(rows, slots);
    if (pairs > std.math.maxInt(i32) or l.dims > std.math.maxInt(i32) or l.width > std.math.maxInt(i32)) return error.Exl3Shape;
    const rotated = try aligned(try mul(try mul(if (l.shared_input) rows else pairs, l.dims), 2));
    const y = if (o.ey_direct) 0 else try aligned(try mul(try mul(pairs, l.dims), if (o.ybf16) 2 else 4));
    const order = if (o.order) try aligned(try mul(grouped.maxItems(pairs, l.experts, tile), 4)) else 0;
    return .{ rotated, if (l.shared_input) 0 else rotated, try aligned(try mul(try mul(pairs, l.width), 2)), y, order };
}
pub fn bytes(l: ex.Layer, rows: usize, slots: usize, o: Ovn) !usize {
    const ns = try sizes(l, rows, slots, o);
    var sum: usize = 0;
    for (ns) |n| sum = try add(sum, n);
    return sum;
}
pub fn scratch(base: u64, l: ex.Layer, rows: usize, slots: usize, o: Ovn) !Scratch {
    const ns = try sizes(l, rows, slots, o);
    const total = try bytes(l, rows, slots, o);
    _ = std.math.add(u64, base, total) catch return error.Exl3Shape;
    const y = base + ns[0] + ns[1] + ns[2];
    return .{ .xg = base, .xu = if (l.shared_input) base else base + ns[0], .xd = base + ns[0] + ns[1], .y = if (o.ey_direct) 0 else y, .rows = rows, .slots = slots, .order = if (o.order) y + ns[3] else 0 };
}
fn go(f: Function, s: Stream, grid: [3]usize, threads: u32, values: anytype) !void {
    try goShared(f, s, grid, threads, 0, values);
}
fn goShared(f: Function, s: Stream, grid: [3]usize, threads: u32, shared: u32, values: anytype) !void {
    var args: launch.Args = .{};
    inline for (values) |v| args.add(v);
    try launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = shared }, s, &args);
}
fn i(n: usize) i32 {
    return @intCast(n);
}
pub const Kernels = struct {
    rotate: Function,
    gateup: Function,
    down: Function,
    ovn: Ovn = .{},
    extra: ?Extra = null, // the switches' kernels, resolved only when one is on

    pub const Extra = struct { gateup: Function, down: Function, order: Function };

    pub fn resolve(mod: Module) !Kernels {
        var k: Kernels = .{ .rotate = try mod.function("tf_exl3_prompt_rotate"), .gateup = try mod.function("tf_exl3_prompt_gu"), .down = try mod.function("tf_exl3_prompt_down") };
        k.ovn = try Ovn.fromEnv();
        try k.prepare(mod);
        return k;
    }
    /// The kernels `ovn` selects (none while every switch is off); a test sets `ovn` on a copy and calls this.
    pub fn prepare(k: *Kernels, mod: Module) !void {
        k.extra = null;
        if (!k.ovn.any()) return;
        var name: [32]u8 = undefined;
        const gu = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_exl3_ovn_gu_{d}", .{k.ovn.k64}, 0));
        const dn = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_exl3_ovn_down_{d}", .{k.ovn.k64}, 0));
        if (k.ovn.k64 != 0) {
            try gu.allowDynamicShared(k64_shared);
            try dn.allowDynamicShared(k64_shared);
        }
        k.extra = .{ .gateup = gu, .down = dn, .order = try mod.function("tf_exl3_ovn_order") };
    }
    pub fn run(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan, out: Out) !void {
        if (!eligible(l) or rows == 0 or rows > sc.rows or stride < l.dims or stride > std.math.maxInt(i32)) return error.Exl3Shape;
        _ = try bytes(l, rows, sc.slots, k.ovn);
        const pairs = rows * sc.slots;
        try router.route(s, picks, pairs, l.experts, tile, p);
        if (l.shared_input) {
            try go(k.rotate, s, .{ rows, l.dims / 128, 1 }, 32, .{ x, i(stride), l.suh_g, sc.xg, i(l.dims) });
        } else {
            var args: launch.Args = .{};
            inline for (.{ x, i(stride), @as(i32, @backingInt(ex.DType.bf16)), picks, l.suh_g, l.suh_u, sc.xg, sc.xu, i(l.dims), i(sc.slots), i(l.experts) }) |v| args.add(v);
            try launch.launch(e.rot_in, .{ .grid = .{ .x = @intCast(pairs), .y = @intCast(l.dims / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &args);
        }
        const items = grouped.maxItems(pairs, l.experts, tile);
        if (k.extra) |xk| return k.switched(xk, s, l, sc, p, out, items);
        if (out.y != sc.y or out.slots != sc.slots) return error.Exl3Shape; // other rows than the scratch's: EY_DIRECT
        try go(k.gateup, s, .{ l.width / 128, items, 1 }, 256, .{ sc.xg, sc.xu, l.gate, l.up, p.items, p.counts, p.members, l.svh_g, l.svh_u, l.suh_d, sc.xd, i(l.dims), i(l.width), i(if (l.shared_input) sc.slots else 1), l.limit, l.act_mode });
        try go(k.down, s, .{ l.dims / 256, items, 1 }, 256, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), i(sc.slots) });
    }
    /// The switched launches: the item order first, then gate/up and down with the order and the caller's rows.
    fn switched(k: Kernels, xk: Extra, s: Stream, l: ex.Layer, sc: Scratch, p: grouped.Plan, out: Out, items: usize) !void {
        const o = k.ovn;
        // a scratch built for other switches, or rows the down kernel cannot write
        if (out.y == 0 or out.slots < sc.slots or (o.order and sc.order == 0)) return error.Exl3Shape;
        if (o.order) try go(xk.order, s, .{ 1, 1, 1 }, 1024, .{ p.items, p.counts, sc.order, i(items), i(l.experts) });
        const order: u64 = if (o.order) sc.order else 0;
        const shared: u32 = if (o.k64 != 0) k64_shared else 0;
        try goShared(xk.gateup, s, .{ l.width / 128, items, 1 }, 256, shared, .{ sc.xg, sc.xu, l.gate, l.up, p.items, p.counts, p.members, l.svh_g, l.svh_u, l.suh_d, sc.xd, i(l.dims), i(l.width), i(if (l.shared_input) sc.slots else 1), l.limit, l.act_mode, order });
        try goShared(xk.down, s, .{ l.dims / 256, items, 1 }, 256, shared, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, out.y, i(l.width), i(l.dims), i(sc.slots), i(out.slots), @as(i32, @intFromBool(o.ybf16)), order });
    }
};
