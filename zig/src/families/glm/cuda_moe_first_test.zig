//! TF_GLM_OVN_SHARED_FIRST under a recording driver: today's launches with today's argument bytes, reordered.
const std = @import("std");
const cuda = @import("cuda");
const moe = @import("cuda_moe.zig");
const glue = @import("cuda_glue.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const t = std.testing;

/// One launch: its kernel, grid and block, and a hash of its arguments' bytes (each argument's own bytes, no padding).
const Launch = struct { f: usize, grid: [3]c_uint, block: [3]c_uint, args: u64 };

const Rec = struct {
    var log: [64]Launch = undefined;
    var n: usize = 0;

    fn note(f: cuda.abi.Function, grid: [3]c_uint, block: [3]c_uint, params: ?[*]?*anyopaque) void {
        const ptrs: *const [cuda.Args.max_args]?*anyopaque = @ptrCast(params.?);
        const a: *const cuda.Args = @alignCast(@fieldParentPtr("ptrs", ptrs));
        var h = std.hash.Wyhash.init(a.count);
        for (a.offsets[0..a.count], a.sizes[0..a.count]) |off, size| h.update(a.storage[off..][0..size]);
        log[n] = .{ .f = @intFromPtr(f), .grid = grid, .block = block, .args = h.final() };
        n += 1;
    }

    fn plain(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f, .{ gx, gy, gz }, .{ bx, by, bz }, params);
        return cuda.abi.success;
    }

    fn ex(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f, .{ c.grid_x, c.grid_y, c.grid_z }, .{ c.block_x, c.block_y, c.block_z }, params);
        return cuda.abi.success;
    }

    /// The launches since the last take, copied out.
    fn take(a: std.mem.Allocator) ![]Launch {
        defer n = 0;
        return a.dupe(Launch, log[0..n]);
    }
};

/// A GLM-5.3-Flash-shaped NVFP4 MoE layer (4,096 dims, 256 experts, top-8) at distinct addresses.
const layer: moe.Layer = .{
    .router = 0x1000_0000,
    .bias = 0x1100_0000,
    .routed = .{ .format = .nvfp4, .up = 0x1200_0000, .down = 0x1300_0000, .up_scale = 0x1400_0000, .down_scale = 0x1500_0000, .width = 512, .dims = 4096, .experts = 256, .limit = 7.0 },
    .count = 256,
    .shared_gu = .{ .bf16 = 0x1600_0000 },
    .shared_down = .{ .bf16 = 0x1700_0000 },
    .shared = 512,
    .top_k = 8,
    .scale = 2.5,
    .norm = true,
};

const scratch: moe.Scratch = .{
    .mlog = 0x2000_0000,
    .router_part = 0x2100_0000,
    .pick = 0x2200_0000,
    .wts = 0x2300_0000,
    .plan = .{ .members = 0x2400_0000, .items = 0x2500_0000, .counts = 0x2600_0000, .rank = 0x2700_0000, .hist = 0x2800_0000 },
    .eact = 0x2900_0000,
    .ey = 0x2a00_0000,
    .sgu = 0x2b00_0000,
    .sact = 0x2c00_0000,
    .sxs = 0x2d00_0000,
    .sy = 0x2e00_0000,
    .mm_part = 0x2f00_0000,
};

const x: u64 = 0x3000_0000;
const out: u64 = 0x3100_0000;
const rows = 4;

fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

/// A list's launches one after another, as one expected sequence.
fn join(a: std.mem.Allocator, parts: []const []const Launch) ![]Launch {
    return std.mem.concat(a, Launch, parts);
}

test "TF_GLM_OVN_SHARED_FIRST: a decode window's shared expert before its routed experts, every launch's bytes as today" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.plain;
    d.api.cuLaunchKernelEx = Rec.ex;
    var gk: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 0..) |name, i| @field(gk, name) = function(&d, i + 1);
    var ek: experts.Kernels = undefined;
    for (&ek.nvfp4.fns, 0..) |*f, i| f.* = function(&d, 500 + i);
    for (&ek.nvfp4.prompt_fns, 0..) |*f, i| f.* = function(&d, 520 + i);
    for (&ek.nvfp4.fact_fns, 0..) |*f, i| f.* = function(&d, 540 + i);
    ek.nvfp4.resident = @splat(1 << 20);
    ek.nvfp4.warps = 4;
    inline for (@typeInfo(cuda.grouped.Router).@"struct".field_names, 0..) |name, i| @field(ek.router, name) = function(&d, 600 + i);
    var xk: exl3.Kernels = undefined; // an NVFP4 layer never reads it
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    Rec.n = 0;
    for ([4][2]bool{ .{ false, false }, .{ true, false }, .{ false, true }, .{ true, true } }) |knobs| {
        var k: moe.Kernels = .{ .set = .{ .glue = &gk }, .experts = &ek, .exl3 = &xk, .put_slot = function(&d, 700), .ovn = .{ .combine_sy = knobs[0], .topk_sum = knobs[1] } };
        // the parts on their own: the router and pick, the routed experts, the shared expert, the slot sum
        try moe.route(k, s, layer, scratch, x, 4096, rows);
        const R = try Rec.take(a);
        try moe.routed(k, s, layer, scratch, x, 4096, rows, false);
        const E = try Rec.take(a);
        try moe.sharedFront(k, s, layer, scratch, x, 4096, 0, rows);
        const S = try Rec.take(a);
        try moe.fillTail(k, s, layer, scratch, 0, rows, false, out);
        const T = try Rec.take(a);
        try t.expect(R.len > 0 and E.len > 0 and S.len > 0 and T.len > 0);
        try moe.run(k, s, layer, scratch, x, 4096, rows, false, out);
        const off = try Rec.take(a);
        try t.expectEqualSlices(Launch, try join(a, &.{ R, E, S, T }), off); // knob off: today's block
        k.ovn.shared_first = true;
        try moe.run(k, s, layer, scratch, x, 4096, rows, false, out);
        try t.expectEqualSlices(Launch, try join(a, &.{ R, S, E, T }), try Rec.take(a)); // the same launches, S moved
        try moe.run(k, s, layer, scratch, x, 4096, rows, false, null); // a fill by row pieces later: the body alone
        try t.expectEqualSlices(Launch, try join(a, &.{ R, E }), try Rec.take(a));
        try moe.run(k, s, layer, scratch, x, 4096, rows, true, out); // a prompt chunk: today's order under the knob
        const on_prompt = try Rec.take(a);
        k.ovn.shared_first = false;
        try moe.run(k, s, layer, scratch, x, 4096, rows, true, out);
        try t.expectEqualSlices(Launch, try Rec.take(a), on_prompt);
    }
}

test "TF_GLM_OVN_SHARED_FIRST: on for decode windows only, read from its own knob" {
    try t.expect(!moe.sharedFirst(.{}, false));
    try t.expect(moe.sharedFirst(.{ .shared_first = true }, false));
    try t.expect(!moe.sharedFirst(.{ .shared_first = true }, true));
    defer cuda.ovn.fake("SHARED_FIRST", null);
    cuda.ovn.fake("SHARED_FIRST", "1");
    try t.expect(moe.Ovn.read().shared_first);
    cuda.ovn.fake("SHARED_FIRST", "0");
    try t.expect(!moe.Ovn.read().shared_first);
    cuda.ovn.fake("SHARED_FIRST", null);
    try t.expect(!moe.Ovn.read().shared_first);
}
