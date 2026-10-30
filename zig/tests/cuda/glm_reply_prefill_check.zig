//! TF_GLM_OVN_REPLY_PREFILL: turn 2 after a reply's background pass (A fresh, B knob off, C pass, E aside, D red).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `glm-reply-prefill-check`: B, C and E equal A in the first token and the next window's tokens and logits; D differs.
pub fn replyPrefillCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
    const capacity = 16384;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's turns, background passes and the one that stepped aside to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    try x.snapshots(2 << 30);
    const V = e.share.vocab.len();
    const rp = glm.reply_prefill;
    const rows: usize = if (try rp.settings()) |s| s.rows else rp.default_rows; // the knob's cap, else its default
    const grid: u32 = if (glm.kda_chunk.snapGrid()) |g| g else 1; // KDA_CHUNKED keeps states on 32-row sub-chunks
    // turn 1's prompt and history mark, the warm prefix (its reply and the next turn's opener), turn 2's prompt
    const Case = struct { turn1: usize, history: u32, warm: usize, turn2: usize };
    const cases = [_]Case{
        .{ .turn1 = 3000, .history = 2990, .warm = 3300, .turn2 = 3500 }, // a 300-row reply, a short tool result
        .{ .turn1 = 1900, .history = 1890, .warm = 2700, .turn2 = 4000 }, // the pass crosses the 2,048 grid
        .{ .turn1 = 5000, .history = 4977, .warm = 5123, .turn2 = 5160 }, // off the cap's grid, a 37-row result
        .{ .turn1 = 9000, .history = 8995, .warm = 9800, .turn2 = 11000 },
    };
    const Keep = struct {
        e: *glm.engine.Engine,
        id: u64 = 0,
        at: u32 = 0,
        fn call(ctx: *anyopaque, at: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ctx));
            k.id = k.e.keepState() catch 0;
            k.at = at;
        }
    };
    const Gate = struct { // a background pass's gate: real work arrives at call `fire` (0: never)
        n: usize = 0,
        fire: usize,
        fn check(p: ?*anyopaque) bool {
            const g: *@This() = @ptrCast(@alignCast(p.?));
            g.n += 1;
            return g.fire != 0 and g.n >= g.fire;
        }
    };
    const Run = struct {
        fn prompt(l: glm.follow.Leader, ids: []const u32, start: usize, from: ?glm.engine.Resume, marks: ?glm.engine.Marks) !u32 {
            var at = start;
            while (true) {
                const r = try l.chunk(ids, at, false, if (at == start) from else null, null, marks);
                if (r.first) |f| return f;
                at += r.rows;
            }
        }
        /// Lanes.prefillFn's steps for a background pass; `most`: the most rows one step ran.
        fn background(l: glm.follow.Leader, ids: []const u32, start: usize, from: glm.engine.Resume, marks: glm.engine.Marks, gate: *Gate, cap: usize, most: *usize) !u32 {
            var at = start;
            while (true) {
                const r = try l.chunkOf(ids, at, false, if (at == start) from else null, .{ .ptr = gate, .check = Gate.check }, marks, false, cap);
                most.* = @max(most.*, r.rows);
                if (r.first) |f| return f;
                at += r.rows;
            }
        }
    };
    var failed = false;
    for (cases, 0..) |case, i| {
        const history = case.history / grid * grid;
        const w: u32 = @intCast((case.warm - 1) / grid * grid); // the pass's mark: a lookahead row before its end
        const ids = try a.alloc(u32, case.turn2);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        const changed = try a.dupe(u32, ids);
        const flip = @min(case.turn2 - 1, w + 5);
        changed[flip] = if (changed[flip] == 4321) 4322 else 4321;
        // turn 1 on slot 1: its prompt pass keeps the history mark, as the request's own pass does
        var turn1: Keep = .{ .e = e };
        try x.bind(1);
        try x.begin(null);
        _ = try Run.prompt(x, ids[0..case.turn1], 0, null, .{ .at = &.{history}, .ctx = &turn1, .call = Keep.call });
        if (turn1.id == 0) {
            std.debug.print("FAIL reply prefill: case {d}: turn 1 kept no state at {d}\n", .{ i, history });
            return error.TestFailed;
        }
        const h: glm.engine.Resume = .{ .id = turn1.id, .at = history };
        // C's pass on slot 2, E's on slot 3 (it steps aside at its second gate, keeping the row it reached)
        var warm: Keep = .{ .e = e };
        var aside: Keep = .{ .e = e };
        var most: usize = 0;
        var never: Gate = .{ .fire = 0 };
        var arrives: Gate = .{ .fire = 2 };
        try x.bind(2);
        try x.begin(null);
        _ = try Run.background(x, ids[0..case.warm], history, h, .{ .at = &.{w}, .ctx = &warm, .call = Keep.call }, &never, rows, &most);
        try x.bind(3);
        try x.begin(null);
        const stepped = Run.background(x, ids[0..case.warm], history, h, .{ .at = &.{w}, .ctx = &aside, .call = Keep.call, .interrupted = Keep.call }, &arrives, rows, &most);
        const aside_ok = if (stepped) |_| false else |err| err == error.Cancelled; // it stopped at the gate, not the end
        const first_end = rp.capEnd(history, glm.engine.chunkEnd(history, case.warm, &.{w}), rows);
        if (!aside_ok or aside.id == 0 or aside.at != first_end or warm.id == 0 or warm.at != w) {
            std.debug.print("FAIL reply prefill: case {d}: pass kept {d} at {d} (mark {d}); the stepped-aside one stopped {}, kept {d} at {d} (want {d})\n", .{ i, warm.id, warm.at, w, aside_ok, aside.id, aside.at, first_end });
            return error.TestFailed;
        }
        if (rows > 0 and most > rows) {
            std.debug.print("FAIL reply prefill: case {d}: a background step ran {d} rows, past the cap of {d}\n", .{ i, most, rows });
            failed = true;
        }
        var firsts: [5]u32 = undefined;
        var outs: [5][4]u32 = undefined;
        var logits: [5][]u16 = undefined;
        var pos: [5]usize = undefined;
        for (0..5) |k| { // A fresh, B from the history, C from the pass's mark, E from where it stepped aside, D red
            const slot: u32 = @intCast(if (k == 4) 4 else k);
            const from: ?glm.engine.Resume = switch (k) {
                0 => null,
                1 => h,
                2, 4 => .{ .id = warm.id, .at = w },
                else => .{ .id = aside.id, .at = aside.at },
            };
            try x.bind(slot);
            try x.begin(null);
            firsts[k] = Run.prompt(x, if (k == 4) changed else ids, if (from) |r| r.at else 0, from, null) catch |err| {
                std.debug.print("FAIL reply prefill: case {d}, twin {d}: {s}\n", .{ i, k, @errorName(err) });
                return err;
            };
            const win = [_]u32{ firsts[k], 101, 202, 303 };
            try x.verify(&win, &outs[k], 0);
            logits[k] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[k].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
            pos[k] = e.cur.st.pos;
        }
        for (1..4) |k| {
            if (firsts[0] != firsts[k] or !std.mem.eql(u32, &outs[0], &outs[k]) or !std.mem.eql(u16, logits[0], logits[k]) or pos[0] != pos[k]) {
                const d = std.mem.indexOfDiff(u16, logits[0], logits[k]);
                std.debug.print("FAIL reply prefill: case {d}, twin {d} (1 knob off, 2 the pass, 3 stepped aside) first {d}/{d}, first logit difference at {?d}\n", .{ i, k, firsts[0], firsts[k], d });
                failed = true;
            }
        }
        if (std.mem.eql(u16, logits[0], logits[4])) {
            std.debug.print("FAIL reply prefill: case {d}: the red control (token {d} changed past {d}) left the logits unchanged\n", .{ i, flip, w });
            failed = true;
        }
        for ([_]u32{ 1, 2, 3, 4 }) |s| try x.release(s);
        for ([_]u64{ turn1.id, warm.id, aside.id }) |id| e.dropState(id); // the other ranks drop them at the next gate
    }
    if (failed) return error.TestFailed;
    check.pass("reply prefill at TP={d}, chunks capped at {d} rows: {d} next turns resumed from a background pass, and from one that stepped aside at its second gate, equal the turn fresh and resumed from its history in the first token and the next window's tokens and logits; every red control differed", .{ world, rows, cases.len });
}
