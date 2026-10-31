//! TF_GLM_OVN_PRX against NCCL, byte for byte: every layer's glued rows and two windows' logits.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// One engine pass: the prompt's first token, a 4-row window's and a 1-row window's tokens and logits.
const Pass = struct { first: u32 = 0, outs: [5]u32 = @splat(0), logits: []u16 = &.{} };

/// `glm-prx-check <ckpt> <rank> <world> <idfile>`: on every rank NCCL, then the channel copying, then it with ZC.
pub fn prxCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
    var group: cuda.tp_group.Group = .{ .lib = lib, .comm = comm, .rank = @intCast(rank), .world = @intCast(world) };
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try group.openPrx(gpu.gpa, gpu.io, gpu.d, s);
    const ch = group.prx orelse {
        std.debug.print("FAIL prx: rank {d}: the channel did not come up (see the warning above)\n", .{rank});
        return error.TestFailed;
    };
    defer ch.deinit();
    var caps: [3]glm.rotlayer.Capture = undefined;
    var runs: [3]Pass = undefined;
    var calls: [3]u64 = undefined;
    var zcs: [3]u64 = undefined;
    for (&runs, &caps, &calls, &zcs, 0..) |*p, *cap, *n, *z, k| {
        group.prx = if (k > 0) ch else null; // every rank alike: NCCL first, then the channel, then it in place
        ch.zc = k == 2;
        const before = ch.calls;
        const zc_before = ch.zc_calls;
        p.* = try pass(gpu, a, checkpoint, &group, rank, world, set, cap);
        n.* = ch.calls - before;
        z.* = ch.zc_calls - zc_before;
        try ch.check();
    }
    group.prx = null;
    var failed = false;
    if (calls[0] != 0 or calls[1] == 0 or calls[2] == 0 or zcs[1] != 0 or zcs[2] == 0) {
        std.debug.print("FAIL prx: rank {d}: exchanges by the channel {d}/{d}/{d}, read in place {d}/{d}/{d} (NCCL, copied, ZC)\n", .{ rank, calls[0], calls[1], calls[2], zcs[0], zcs[1], zcs[2] });
        failed = true;
    }
    var layers: usize = 0;
    var rows: usize = 0;
    for (caps[0].sums, caps[0].rows, 0..) |sa, ra, j| {
        if (ra == 0 and caps[1].rows[j] == 0 and caps[2].rows[j] == 0) continue;
        layers += 1;
        rows += ra;
        for (caps[1..], 1..) |cb, k| if (sa != cb.sums[j] or ra != cb.rows[j]) {
            std.debug.print("FAIL prx: rank {d}: layer {d}'s glued rows differ in pass {d} ({d} and {d} rows)\n", .{ rank, j, k, ra, cb.rows[j] });
            failed = true;
        };
    }
    if (layers == 0) {
        std.debug.print("FAIL prx: rank {d}: no layer's rows were captured\n", .{rank});
        return error.TestFailed;
    }
    if (rank == 0) {
        const A = runs[0];
        for (runs[1..], 1..) |B, k| if (A.first != B.first or !std.mem.eql(u32, &A.outs, &B.outs) or !std.mem.eql(u16, A.logits, B.logits)) {
            std.debug.print("FAIL prx: pass {d}: first token {d} against {d}; first logit difference at {?d}\n", .{ k, B.first, A.first, std.mem.indexOfDiff(u16, A.logits, B.logits) });
            failed = true;
        };
    }
    if (failed) return error.TestFailed;
    check.pass("TF_GLM_OVN_PRX at TP={d}: rank {d}'s glued rows of {d} layers ({d} in all, a two-chunk prompt and two decode windows){s} equal NCCL's byte for byte, copied ({d} exchanges by the NICs) and read in place ({d} of {d})", .{ world, rank, layers, rows, if (rank == 0) " and the windows' tokens and logits" else "", calls[1], zcs[2], calls[2] });
}

/// One engine on this rank over `group` (its channel set or not), its forwards' glued rows into `cap`: rank 0 leads.
fn pass(gpu: Gpu, a: std.mem.Allocator, checkpoint: []const u8, group: *cuda.tp_group.Group, rank: c_int, world: c_int, set: anytype, cap: anytype) !Pass {
    const glm = @import("glm_cuda");
    const capacity = 16384;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &group.lib, group.comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = false, .settings = set });
    defer e.deinit();
    e.group = group;
    const c = &e.w.config;
    cap.* = .{ .d = gpu.d, .host = try a.alloc(u8, glm.engine.prompt_rows * c.hc * c.hidden * 2) };
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &group.lib, group.comm, e.stream, capacity);
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
    // two chunks then 300 rows: a paired step (by the channel) and a pieced chunk alone (NCCL unless PRX_ALONE=1)
    const ids = try a.alloc(u32, 2 * glm.engine.prompt_rows + 300);
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
