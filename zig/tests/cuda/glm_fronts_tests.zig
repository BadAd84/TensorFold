//! GLM-5.3-Flash's pieced prompt pass with TF_GLM_OVN_PEER_FIRST and _FRONTS on rank 0 against its plain twin.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `glm-fronts-check`: rank 0's twin 0 with both knobs off, twin 1 on; first token, next window and state bytes equal.
pub fn frontsCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192 }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    try check.expect(world > 1, "glm-fronts-check needs two ranks or more (a piece splits only across ranks)", .{});
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const C = glm.engine.prompt_rows;
    const capacity = @max(dc.capacity, 4 * C);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) { // the followers keep the knobs as the environment sets them, for both twins
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's prompt chunks (PEER_FIRST {d}, FRONTS {d}) to the end", .{ rank, e.pieces.peer_first, e.pieces.fronts });
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    // twin 1's knobs: the environment's, or 1 (the floor of every reordered call) where it leaves one unset
    const on = [2]usize{ @max(e.pieces.peer_first, 1), @max(e.pieces.fronts, 1) };
    // a tail chunk of 300 rows (too few: the knobs stand down), a full chunk plus a long tail, one full chunk
    const lens = [_]usize{ 2 * C + 300, C + C / 2, C };
    var failed = false;
    for (lens, 0..) |len, i| {
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        var firsts: [2]u32 = undefined;
        var outs: [2][4]u32 = undefined;
        var logits: [2][]u16 = undefined;
        var states: [2][]u8 = undefined;
        for (0..2) |twin| {
            e.pieces.peer_first = if (twin == 1) on[0] else 0;
            e.pieces.fronts = if (twin == 1) on[1] else 0;
            try x.bind(@intCast(2 * i + twin));
            try x.begin(null);
            var at: usize = 0;
            const t = while (true) { // never paired: a paired step runs pairBody, not the pieced pass
                const r = x.chunk(ids, at, false, null, null, null) catch |err| {
                    std.debug.print("FAIL fronts: prompt of {d}, twin {d}: the step at {d}: {s}\n", .{ len, twin, at, @errorName(err) });
                    return err;
                };
                if (r.first) |f| break f;
                at += r.rows;
            };
            firsts[twin] = t;
            const win = [_]u32{ t, 101, 202, 303 };
            try x.verify(&win, &outs[twin], 0);
            logits[twin] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[twin].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
            try e.stream.synchronize();
            const st = &e.seqs.items[2 * i + twin].?.st;
            states[twin] = try a.alloc(u8, st.arena.buf.len);
            try st.arena.buf.download(0, states[twin]);
            // for comparing runs: the same digest from runs whose followers had the knobs off and on proves theirs
            std.debug.print("DIGEST fronts: prompt of {d}, twin {d}: logits {x} state {x}\n", .{ len, twin, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(logits[twin])), std.hash.Wyhash.hash(0, states[twin]) });
        }
        e.pieces.peer_first = 0;
        e.pieces.fronts = 0;
        if (firsts[0] != firsts[1] or !std.mem.eql(u32, &outs[0], &outs[1]) or !std.mem.eql(u16, logits[0], logits[1])) {
            std.debug.print("FAIL fronts: prompt of {d}: first {d}/{d}, the next window's tokens or logits differ\n", .{ len, firsts[0], firsts[1] });
            failed = true;
        }
        const st = &e.seqs.items[2 * i].?.st;
        const tw = &e.seqs.items[2 * i + 1].?.st;
        const diff = std.mem.indexOfDiff(u8, states[0], states[1]);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or !std.mem.eql(u1, st.cur, tw.cur) or diff != null) {
            std.debug.print("FAIL fronts: prompt of {d}: the state differs (pos {d}/{d}, first byte {?d} of {d})\n", .{ len, st.pos, tw.pos, diff, states[0].len });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("PEER_FIRST {d} and FRONTS {d} on rank 0 at TP={d}: prompts of {any} rows equal their plain twins in the first token, the next window, and every state byte", .{ on[0], on[1], world, lens });
}
