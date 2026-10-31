//! TF_GLM_OVN_PRX against NCCL, byte for byte: every layer's glued rows and two windows' logits.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// One engine pass: the prompt's first token, two windows' tokens and logits; rank 0 counts its steps' channel calls.
const Pass = struct {
    first: u32 = 0,
    outs: [5]u32 = @splat(0),
    logits: []u16 = &.{},
    pair_steps: u64 = 0, // paired prompt steps (Pair.steps)
    pair_calls: u64 = 0, // channel exchanges those steps enqueued
    alone_calls: u64 = 0, // channel exchanges the chunks alone enqueued (PRX_ALONE's only)
};

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
    // the default path (PRX_ALONE off) is a paired step's: it needs the pair and other ranks to exchange with
    try check.expect(set.pair, "prx: glm-prx-check needs TF_GLM_PROMPT_PAIR on (unset or 1)", .{});
    try check.expect(world > 1, "prx: glm-prx-check needs other ranks", .{});
    var group: cuda.tp_group.Group = .{ .lib = lib, .comm = comm, .rank = @intCast(rank), .world = @intCast(world) };
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try group.openPrx(gpu.gpa, gpu.io, gpu.d, s);
    const ch = group.prx orelse {
        std.debug.print("FAIL prx: rank {d}: the channel did not come up (see the warning above)\n", .{rank});
        return error.TestFailed;
    };
    defer ch.deinit();
    const alone = cuda.ovn.on("PRX_ALONE");
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
    if (rank == 0) for (runs, 0..) |p, k| {
        // one paired step in every pass; by the channel in passes 1 and 2; chunks alone by it only with PRX_ALONE
        const pair_ok = p.pair_steps == 1 and (k == 0 or p.pair_calls > 0);
        const alone_ok = if (k > 0 and alone) p.alone_calls > 0 else p.alone_calls == 0;
        if (!pair_ok or !alone_ok) {
            std.debug.print("FAIL prx: pass {d}: {d} paired step(s) took {d} channel exchanges, chunks alone {d} (PRX_ALONE {})\n", .{ k, p.pair_steps, p.pair_calls, p.alone_calls, alone });
            failed = true;
        }
    };
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
    const how = if (alone) "every chunk by the channel (PRX_ALONE)" else "the paired step by the channel, the chunk alone by NCCL";
    check.pass("TF_GLM_OVN_PRX at TP={d}: rank {d}'s glued rows of {d} layers ({d} in all, a paired step, a chunk alone, two decode windows){s} equal NCCL's byte for byte, {s}: copied ({d} exchanges by the NICs) and read in place ({d} of {d})", .{ world, rank, layers, rows, if (rank == 0) " and the windows' tokens and logits" else "", how, calls[1], zcs[2], calls[2] });
}

/// One engine on this rank over `group` (its channel set or not), its forwards' glued rows into `cap`: rank 0 leads.
fn pass(gpu: Gpu, a: std.mem.Allocator, checkpoint: []const u8, group: *cuda.tp_group.Group, rank: c_int, world: c_int, set: anytype, cap: anytype) !Pass {
    const glm = @import("glm_cuda");
    const capacity = glm.prompt_pair.prx_check_capacity;
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
    // two chunks then 300 rows (prompt_pair's host test pins pairedAt's rules): a paired step, a chunk alone
    const ids = try a.alloc(u32, glm.prompt_pair.prx_check_len);
    for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
    try check.expect(e.pairedAt(0, ids.len, null, null), "prx: the prompt's first step would not pair (pairedAt)", .{});
    var at: usize = 0;
    p.first = while (true) { // Leader.chunk, paired where pairedAt allows, as the serve's prompt steps
        const steps = e.pair.?.steps;
        const before = if (group.prx) |ch| ch.calls else 0;
        const r = try x.chunk(ids, at, true, null, null, null);
        const took = (if (group.prx) |ch| ch.calls else 0) - before;
        if (e.pair.?.steps != steps) {
            p.pair_steps += e.pair.?.steps - steps;
            p.pair_calls += took;
        } else p.alone_calls += took;
        if (r.first) |f| break f;
        at += r.rows;
    };
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
