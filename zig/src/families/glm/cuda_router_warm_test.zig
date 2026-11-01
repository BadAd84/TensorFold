//! TF_GLM_OVN_ROUTER_WARM host tests: the row classes, the knob, the layer pick, the router warm launches.
const std = @import("std");
const cuda = @import("cuda");
const rw = @import("cuda_router_warm.zig");
const glue = @import("cuda_glue.zig");
const moe = @import("cuda_moe.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const weights = @import("cuda_weights.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const t = std.testing;

test "TF_GLM_OVN_ROUTER_WARM: the row-class list and its cap by the buffer rows" {
    try t.expectEqualSlices(u32, &.{ 1, 2, 16, 17, 32, 33, 64, 65, 128 }, &rw.row_classes);
    try t.expectEqualSlices(u32, &.{ 1, 2, 16, 17, 32, 33, 64, 65, 128 }, rw.classes(4096));
    try t.expectEqualSlices(u32, &.{ 1, 2, 16, 17, 32, 33, 64, 65, 128 }, rw.classes(128)); // 128 fits
    try t.expectEqualSlices(u32, &.{ 1, 2, 16, 17, 32, 33, 64 }, rw.classes(64));
    try t.expectEqualSlices(u32, &.{ 1, 2, 16, 17, 32 }, rw.classes(32));
    try t.expectEqualSlices(u32, &.{ 1, 2, 16 }, rw.classes(16));
    try t.expectEqualSlices(u32, &.{ 1, 2 }, rw.classes(2));
    try t.expectEqualSlices(u32, &.{1}, rw.classes(1));
    try t.expectEqual(@as(usize, 0), rw.classes(0).len);
}

test "TF_GLM_OVN_ROUTER_WARM: 0 off, 1 the router warm, 2 the kernel warm, any other value refused" {
    defer cuda.ovn.fake("ROUTER_WARM", null);
    cuda.ovn.fake("ROUTER_WARM", null);
    try t.expectEqual(rw.Level.off, try rw.level()); // unset
    cuda.ovn.fake("ROUTER_WARM", "0");
    try t.expectEqual(rw.Level.off, try rw.level());
    cuda.ovn.fake("ROUTER_WARM", "1");
    try t.expectEqual(rw.Level.router, try rw.level());
    cuda.ovn.fake("ROUTER_WARM", "2");
    try t.expectEqual(rw.Level.all, try rw.level());
    cuda.ovn.fake("ROUTER_WARM", "3");
    try t.expectError(error.BadOvnKnob, rw.level());
    cuda.ovn.fake("ROUTER_WARM", "x");
    try t.expectError(error.BadOvnKnob, rw.level());
}

/// A layer whose only read-by-pickLayers fields (the mixer and ffn tags, the MoE and the DSA index) are set.
fn layerOf(mixer: weights.Mixer, ffn: weights.Ffn) weights.Layer {
    return .{ .index = 0, .attn_hc = .{ .fn_w = 0, .base = 0, .scale = 0 }, .ffn_hc = .{ .fn_w = 0, .base = 0, .scale = 0 }, .in_norm = .{ .bf16 = 0 }, .post_norm = .{ .bf16 = 0 }, .mixer = mixer, .ffn = ffn };
}

test "TF_GLM_OVN_ROUTER_WARM: pickLayers takes the first MoE, the first DSA (its indexer gate) and the first KDA" {
    var moe_a = layerVal;
    moe_a.router = 0x1000_0000;
    var moe_b = layerVal;
    moe_b.router = 0x9000_0000; // a later MoE layer: never picked
    var idx: dsa_block.Index = undefined;
    idx.gate = 0x4000_0000;
    idx.d = 128;
    var dsa_l: dsa_block.Layer = undefined;
    dsa_l.index = idx;
    const layers = [_]weights.Layer{
        layerOf(.{ .kda = undefined }, .{ .dense = undefined }), // layer 0: the first KDA, dense
        layerOf(.{ .mla = dsa_l }, .{ .moe = moe_a }), // layer 1: the first DSA and the first MoE
        layerOf(.{ .kda = undefined }, .{ .moe = moe_b }), // layer 2: a later KDA and MoE
    };
    const p = rw.pickLayers(&layers);
    try t.expect(p.moe != null and p.moe.?.router == 0x1000_0000); // the FIRST MoE, not moe_b's 0x9000_0000
    try t.expect(p.index != null and p.index.?.gate == 0x4000_0000 and p.index.?.d == 128);
    try t.expect(p.kda_layer != null); // the first KDA layer is present
    try t.expect(p.mla_layer != null); // and the first DSA layer
    // a checkpoint without a kind leaves it null (skipped by the warm)
    const dense_only = [_]weights.Layer{layerOf(.{ .kda = undefined }, .{ .dense = undefined })};
    const q = rw.pickLayers(&dense_only);
    try t.expect(q.moe == null and q.index == null and q.mla_layer == null and q.kda_layer != null);
}

/// One recorded launch: its kernel handle.
const Rec = struct {
    var log: [128]usize = undefined;
    var n: usize = 0;

    fn note(f: cuda.abi.Function) void {
        log[n] = @intFromPtr(f);
        n += 1;
    }
    fn plain(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f);
        return cuda.abi.success;
    }
    fn ex(_: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f);
        return cuda.abi.success;
    }
};

fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

/// A GLM-5.3-Flash-shaped NVFP4 MoE layer (4,096 dims, 256 experts, top-8); the fields the route reads.
const layerVal: moe.Layer = .{
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

const scratchVal: moe.Scratch = .{
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

const normed: u64 = 0x3000_0000; // 16-byte aligned: the router takes its tiled (cp.async) path, as the real serve does
const gate: rw.IndexGate = .{ .gate = 0x4000_0000, .igr = 0x4100_0000, .router_part = 0x4200_0000, .ne = 128 };

/// Distinct handles for every glue kernel, its optional (knob-only) ones null so route/gate take the tiled path.
fn fakeKernels(d: *const cuda.Driver, gk: *glue.Kernels) void {
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 0..) |name, i| {
        switch (@typeInfo(@FieldType(glue.Kernels, name))) {
            .optional => @field(gk, name) = null,
            else => @field(gk, name) = function(d, i + 1),
        }
    }
}

test "TF_GLM_OVN_ROUTER_WARM: the router warm launches the route then the indexer gate, once at each row class" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.plain;
    d.api.cuLaunchKernelEx = Rec.ex;
    var gk: glue.Kernels = undefined;
    fakeKernels(&d, &gk);
    var ek: experts.Kernels = undefined;
    var xk: exl3.Kernels = undefined;
    const set: glue.Set = .{ .glue = &gk };
    const mk: moe.Kernels = .{ .set = set, .experts = &ek, .exl3 = &xk, .put_slot = function(&d, 900) };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const tile = @intFromPtr(gk.router_tile.handle);
    const ssum = @intFromPtr(gk.slice_sum.handle);
    const topk = @intFromPtr(gk.topk.handle);

    // both the route and the gate, at the 9 classes that fit 128 buffer rows
    Rec.n = 0;
    const both = try rw.routerWarm(mk, set, s, layerVal, scratchVal, gate, normed, 4096, 128);
    try t.expectEqual(@as(usize, 18), both); // 9 routes + 9 gates
    try t.expectEqual(@as(usize, 45), Rec.n); // each class: route (tile, slice_sum, topk) then gate (tile, slice_sum)
    try t.expectEqualSlices(usize, &.{ tile, ssum, topk, tile, ssum }, Rec.log[0..5]);

    // the cap moves with the buffer rows: 64 rows keep 7 classes
    Rec.n = 0;
    try t.expectEqual(@as(usize, 14), try rw.routerWarm(mk, set, s, layerVal, scratchVal, gate, normed, 4096, 64));
    try t.expectEqual(@as(usize, 35), Rec.n);

    // the route alone (no DSA layer: no gate) and the gate alone (no MoE layer)
    Rec.n = 0;
    try t.expectEqual(@as(usize, 9), try rw.routerWarm(mk, set, s, layerVal, scratchVal, null, normed, 4096, 128));
    try t.expectEqual(@as(usize, 27), Rec.n); // 9 routes x (tile, slice_sum, topk)
    Rec.n = 0;
    try t.expectEqual(@as(usize, 9), try rw.routerWarm(mk, set, s, null, scratchVal, gate, normed, 4096, 128));
    try t.expectEqual(@as(usize, 18), Rec.n); // 9 gates x (tile, slice_sum)
    try t.expectEqualSlices(usize, &.{ tile, ssum }, Rec.log[0..2]);
}
