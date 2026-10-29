//! GLM-5.3-Flash's prompt passes kept and resumed at rows off the 2,048-row grid, against the same prompt fresh.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `glm-resume-check`: per case, A runs a prompt fresh, B runs it with a mark at R (its chunk ends there and the state
/// is kept), C resumes from that state; their first tokens and the next window's tokens and logits must be equal bit
/// for bit. D, a red control, resumes the same state with one prompt token changed past R and must differ.
pub fn resumeCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
        check.pass("rank {d} followed rank 0's kept, resumed and fresh prompt passes to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    try x.snapshots(2 << 30);
    const V = e.share.vocab.len();
    const C = glm.engine.prompt_rows;
    const Case = struct { len: usize, at: u32 };
    const cases = [_]Case{
        .{ .len = 6000, .at = 2 * C }, // on the grid: the check's own control, true before this change too
        .{ .len = 6000, .at = 2 * C + 3 }, // a row three into a pool of four
        .{ .len = 6000, .at = C + 1 },
        .{ .len = 4000, .at = 2051 }, // the dense limit (keyWidth)
        .{ .len = 9000, .at = 5001 },
        .{ .len = 3000, .at = 1 }, // a one-row first chunk
        .{ .len = 2200, .at = 2110 }, // a short tail after the resume
        .{ .len = 7000, .at = 4160 },
    };
    const Keep = struct {
        e: *glm.engine.Engine,
        id: u64 = 0,
        fn call(ctx: *anyopaque, _: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ctx));
            k.id = k.e.keepState() catch 0;
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
    };
    var failed = false;
    for (cases, 0..) |case, i| {
        const ids = try a.alloc(u32, case.len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        const changed = try a.dupe(u32, ids);
        const flip = @min(case.len - 1, case.at + 5);
        changed[flip] = if (changed[flip] == 4321) 4322 else 4321;
        var keep: Keep = .{ .e = e };
        const marks: glm.engine.Marks = .{ .at = &.{case.at}, .ctx = &keep, .call = Keep.call };
        var firsts: [4]u32 = undefined;
        var outs: [4][4]u32 = undefined;
        var logits: [4][]u16 = undefined;
        var pos: [4]usize = undefined;
        for (0..4) |k| { // A fresh, B kept at the mark, C resumed, D resumed with a changed token
            try x.bind(@intCast(k));
            try x.begin(null);
            const from: ?glm.engine.Resume = if (k >= 2) .{ .id = keep.id, .at = case.at } else null;
            if (k >= 2 and keep.id == 0) {
                std.debug.print("FAIL resume: prompt of {d}: no state was kept at {d}\n", .{ case.len, case.at });
                return error.TestFailed;
            }
            firsts[k] = Run.prompt(x, if (k == 3) changed else ids, if (k >= 2) case.at else 0, from, if (k == 1) marks else null) catch |err| {
                std.debug.print("FAIL resume: prompt of {d}, twin {d}, resumed at {d}: {s}\n", .{ case.len, k, case.at, @errorName(err) });
                return err;
            };
            const win = [_]u32{ firsts[k], 101, 202, 303 };
            try x.verify(&win, &outs[k], 0);
            logits[k] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[k].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
            pos[k] = e.cur.st.pos;
        }
        for (1..3) |k| {
            if (firsts[0] != firsts[k] or !std.mem.eql(u32, &outs[0], &outs[k]) or !std.mem.eql(u16, logits[0], logits[k]) or pos[0] != pos[k]) {
                const d = std.mem.indexOfDiff(u16, logits[0], logits[k]);
                std.debug.print("FAIL resume: prompt of {d}, mark at {d}: twin {d} (1 cut there, 2 resumed) first {d}/{d}, first logit difference at {?d}\n", .{ case.len, case.at, k, firsts[0], firsts[k], d });
                failed = true;
            }
        }
        if (std.mem.eql(u16, logits[0], logits[3])) {
            std.debug.print("FAIL resume: prompt of {d}: the red control (token {d} changed past {d}) left the logits unchanged\n", .{ case.len, flip, case.at });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("resumed prompts at TP={d}: {d} prompts kept and resumed at rows on and off the grid (1 to 5,001) equal their fresh passes in the first token and the next window's tokens and logits, and a chunk cut at each mark changes nothing; every red control differed", .{ world, cases.len });
}
