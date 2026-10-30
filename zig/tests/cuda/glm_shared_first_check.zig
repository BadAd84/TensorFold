//! TF_GLM_OVN_SHARED_FIRST against the knob-off order, byte for byte: every layer's glued rows and two windows' logits.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// One engine pass: the prompt's first token, a 4-row window's and a 1-row window's tokens and logits.
const Pass = struct { first: u32 = 0, outs: [5]u32 = @splat(0), logits: []u16 = &.{} };

/// `glm-shared-first-check <ckpt> <rank> <world> <idfile>`: two engines on every rank, the knob off, then on.
pub fn sharedFirstCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    var caps: [2]glm.rotlayer.Capture = undefined;
    var runs: [2]Pass = undefined;
    for (&runs, &caps, 0..) |*p, *cap, k| p.* = try pass(gpu, a, checkpoint, &lib, comm, rank, world, set, k == 1, cap);
    var failed = false;
    var layers: usize = 0;
    var rows: usize = 0;
    for (caps[0].sums, caps[1].sums, caps[0].rows, caps[1].rows, 0..) |sa, sb, ra, rb, j| {
        if (ra == 0 and rb == 0) continue;
        layers += 1;
        rows += ra;
        if (sa != sb or ra != rb) {
            std.debug.print("FAIL shared-first: rank {d}: layer {d}'s glued rows differ ({d} and {d} rows)\n", .{ rank, j, ra, rb });
            failed = true;
        }
    }
    if (layers == 0) {
        std.debug.print("FAIL shared-first: rank {d}: no layer's rows were captured\n", .{rank});
        return error.TestFailed;
    }
    if (rank == 0) {
        const A = runs[0];
        const B = runs[1];
        if (A.first != B.first or !std.mem.eql(u32, &A.outs, &B.outs) or !std.mem.eql(u16, A.logits, B.logits)) {
            std.debug.print("FAIL shared-first: first token {d} against {d}; first logit difference at {?d}\n", .{ B.first, A.first, std.mem.indexOfDiff(u16, A.logits, B.logits) });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("TF_GLM_OVN_SHARED_FIRST at TP={d}: rank {d}'s glued rows of {d} layers ({d} in all, a prompt and two decode windows){s} equal the knob-off order's byte for byte", .{ world, rank, layers, rows, if (rank == 0) " and the windows' tokens and logits" else "" });
}

/// One engine on this rank with the knob set to `on`, its forwards' glued rows into `cap`: rank 0 leads.
fn pass(gpu: Gpu, a: std.mem.Allocator, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: c_int, world: c_int, set: anytype, on: bool, cap: anytype) !Pass {
    const glm = @import("glm_cuda");
    const capacity = 16384;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = false, .settings = set });
    defer e.deinit();
    if (e.m.side != null) { // TF_GLM_OVN_SHARED_SIDE takes the decode windows' MoE first: the knob would run nowhere
        std.debug.print("FAIL shared-first: rank {d}: unset TF_GLM_OVN_SHARED_SIDE\n", .{rank});
        return error.TestFailed;
    }
    e.m.moe_ovn.shared_first = on; // every rank alike, whatever the environment says
    const c = &e.w.config;
    cap.* = .{ .d = gpu.d, .host = try a.alloc(u8, glm.engine.prompt_rows * c.hc * c.hidden * 2) };
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, lib, comm, e.stream, capacity);
    defer link.deinit();
    glm.rotlayer.capture = cap;
    defer glm.rotlayer.capture = null;
    var p: Pass = .{};
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        return p;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    try x.bind(0);
    try x.begin(null);
    const ids = try a.alloc(u32, 1500);
    for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
    p.first = try x.prefill(ids, null, null);
    const V = e.share.vocab.len();
    p.logits = try a.alloc(u16, 5 * V);
    const win = [_]u32{ p.first, 101, 202, 303 };
    try x.verify(&win, p.outs[0..4], 0);
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(p.logits.ptr, e.bd.logits, 4 * V * 2), "cuMemcpyDtoH");
    try x.keep(win.len);
    try x.verify(p.outs[3..4], p.outs[4..5], 0); // a one-row window after the commit
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(p.logits[4 * V ..].ptr, e.bd.logits, V * 2), "cuMemcpyDtoH");
    return p;
}
