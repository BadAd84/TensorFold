//! GLM-5.3-Flash's prompt chunks under TF_GLM_OVN_ONE_CHUNK against the same prompts cut at every mark.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const max_keeps = 8;

/// The marks' keeps of one pass, in the order the pass made them.
const Keep = struct {
    e: *anyopaque,
    ids: [max_keeps]u64 = @splat(0),
    ats: [max_keeps]u32 = @splat(0),
    n: usize = 0,
    fn call(ctx: *anyopaque, at: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ctx));
        const glm = @import("glm_cuda");
        const e: *glm.engine.Engine = @ptrCast(@alignCast(k.e));
        if (k.n == max_keeps) return;
        k.ids[k.n] = e.keepState() catch 0;
        k.ats[k.n] = at;
        k.n += 1;
    }
};

/// One pass's outcome: its first token, the next window's tokens and logits, its steps and its keeps.
const Pass = struct { first: u32, outs: [4]u32, logits: []u16, pos: usize, steps: usize, keep: Keep };

/// A kept state's bytes (its lease in order) and host words, as a restore would read them.
fn kept(e: anytype, a: std.mem.Allocator, id: u64) ![]u8 {
    const glm = @import("glm_cuda");
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const arena = e.snap_pages.?.arena.?;
    const page = glm.snapshots.pages.page;
    const words = 3 + 2 + s.cur.len;
    const out = try a.alloc(u8, s.len + words * 8);
    var off: usize = 0;
    while (off < s.len) {
        const n = @min(s.len - off, page - off % page);
        const at = arena.ptr + @as(u64, s.lease.ids.items[off / page]) * page + off % page;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(out[off..].ptr, at, n), "cuMemcpyDtoH");
        off += n;
    }
    const host = try a.alloc(u64, words);
    host[0] = s.pos;
    host[1] = s.mtp_len;
    host[2] = s.len;
    host[3] = s.bases[0];
    host[4] = s.bases[1];
    for (s.cur, 0..) |b, i| host[5 + i] = b;
    @memcpy(out[s.len..], std.mem.sliceAsBytes(host));
    return out;
}

/// `glm-onechunk-check` (TF_GLM_OVN_ONE_CHUNK=1): marks kept inside chunks against chunks cut at them, bit for bit.
pub fn oneChunkCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
    const capacity = 16384; // index rings of 8,192 rows: case 0 and case 2 rebase them
    // no MTP head: a head that absorbs prompt rows keeps every mark at a chunk stop (cuda_one_chunk.keepsInside)
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = false, .settings = set });
    defer e.deinit();
    if (e.one == null) {
        std.debug.print("FAIL onechunk: TF_GLM_OVN_ONE_CHUNK is not set on rank {d}\n", .{rank});
        return error.TestFailed;
    }
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's prompts cut at every mark and in one chunk to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    try x.snapshots(8 << 30); // case 0's six states at TP=1 hold about 450 MB each
    const V = e.share.vocab.len();
    const C = glm.engine.prompt_rows;
    const Case = struct { len: usize, from: u32, marks: []const u32 };
    const cases = [_]Case{
        // two marks inside the first chunk, one inside the last, whose rings rebase at its start
        .{ .len = 2 * C + 1808, .from = 0, .marks = &.{ 1500, 2049, 2 * C + 1805 } },
        // the warm turn: resumed off the grid, the history's mark, the grid point, the next turn's mark 3 short
        .{ .len = 6000, .from = 3001, .marks = &.{ 3500, 5997 } },
        // the knob-off path rebases at 8,192 past the step's start: the step stops there (cut's rebase rule)
        .{ .len = 9000, .from = 8000, .marks = &.{ 8100, 8997 } },
        // three marks in one chunk: the third is past the slots and ends the step
        .{ .len = 3000, .from = 0, .marks = &.{ 700, 1400, 2101, 2997 } },
    };
    const Run = struct {
        fn prompt(l: glm.follow.Leader, ids: []const u32, start: usize, from: ?glm.engine.Resume, marks: ?glm.engine.Marks, steps: *usize) !u32 {
            var at = start;
            while (true) {
                const r = try l.chunk(ids, at, false, if (at == start) from else null, null, marks);
                steps.* += 1;
                if (r.first) |f| return f;
                at += r.rows;
            }
        }
        fn pass(l: glm.follow.Leader, al: std.mem.Allocator, slot: u32, ids: []const u32, from: ?glm.engine.Resume, at: []const u32, keep: bool, vocab: usize) !Pass {
            try l.bind(slot);
            try l.begin(null);
            var p: Pass = .{ .first = 0, .outs = undefined, .logits = undefined, .pos = 0, .steps = 0, .keep = .{ .e = l.e } };
            const marks: glm.engine.Marks = .{ .at = at, .ctx = &p.keep, .call = Keep.call };
            p.first = try prompt(l, ids, if (from) |r| r.at else 0, from, if (keep) marks else null, &p.steps);
            const win = [_]u32{ p.first, 101, 202, 303 };
            try l.verify(&win, &p.outs, 0);
            p.logits = try al.alloc(u16, win.len * vocab);
            try l.e.d.check(l.e.d.api.cuMemcpyDtoH_v2(p.logits.ptr, l.e.bd.logits, win.len * vocab * 2), "cuMemcpyDtoH");
            p.pos = l.e.cur.st.pos;
            return p;
        }
        fn same(p: *const Pass, q: *const Pass) bool {
            return p.first == q.first and std.mem.eql(u32, &p.outs, &q.outs) and std.mem.eql(u16, p.logits, q.logits) and p.pos == q.pos;
        }
    };
    var failed = false;
    var inner: usize = 0;
    for (cases, 0..) |case, i| {
        const ids = try a.alloc(u32, case.len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        glm.one_chunk.on = false;
        var from: ?glm.engine.Resume = null;
        if (case.from != 0) { // the resumed state, kept as before at a chunk stop
            const base = try Run.pass(x, a, 4, ids, null, &.{case.from}, true, V);
            if (base.keep.n != 1 or base.keep.ids[0] == 0) {
                std.debug.print("FAIL onechunk: case {d}: no state kept at {d} to resume\n", .{ i, case.from });
                return error.TestFailed;
            }
            from = .{ .id = base.keep.ids[0], .at = case.from };
        }
        const A = try Run.pass(x, a, 0, ids, from, case.marks, true, V);
        glm.one_chunk.on = true;
        const B = try Run.pass(x, a, 1, ids, from, case.marks, true, V);
        if (!Run.same(&A, &B)) {
            const d = std.mem.indexOfDiff(u16, A.logits, B.logits);
            std.debug.print("FAIL onechunk: case {d}: one chunk first {d}, cut {d}; first logit difference at {?d}\n", .{ i, B.first, A.first, d });
            failed = true;
        }
        if (B.steps >= A.steps) {
            std.debug.print("FAIL onechunk: case {d}: {d} steps under the knob against {d} cut at every mark\n", .{ i, B.steps, A.steps });
            failed = true;
        }
        if (A.keep.n != case.marks.len or B.keep.n != A.keep.n) {
            std.debug.print("FAIL onechunk: case {d}: {d} and {d} keeps for {d} marks\n", .{ i, A.keep.n, B.keep.n, case.marks.len });
            failed = true;
        }
        for (0..@min(A.keep.n, B.keep.n)) |k| {
            const sa = try kept(e, a, A.keep.ids[k]);
            const sb = try kept(e, a, B.keep.ids[k]);
            if (A.keep.ats[k] != B.keep.ats[k] or !std.mem.eql(u8, sa, sb)) {
                const d = std.mem.indexOfDiff(u8, sa, sb);
                std.debug.print("FAIL onechunk: case {d}: the state kept at {d} differs at byte {?d} of {d}\n", .{ i, A.keep.ats[k], d, sa.len });
                failed = true;
            }
        }
        inner += A.steps - B.steps;
        if (B.keep.n > 0) { // C: B's first kept state resumed to the end, against A's whole pass
            const C1 = try Run.pass(x, a, 2, ids, .{ .id = B.keep.ids[0], .at = B.keep.ats[0] }, &.{}, false, V);
            if (!Run.same(&A, &C1)) {
                std.debug.print("FAIL onechunk: case {d}: resumed at {d} from the knob's kept state, first {d} against {d}\n", .{ i, B.keep.ats[0], C1.first, A.first });
                failed = true;
            }
        }
        for (A.keep.ids[0..A.keep.n]) |id| e.dropState(id);
        for (B.keep.ids[0..B.keep.n]) |id| e.dropState(id);
        if (from) |r| e.dropState(r.id);
    }
    if (failed) return error.TestFailed;
    check.pass("one-chunk prompts at TP={d}: {d} prompts in {d} fewer steps keep every mark's state byte for byte and equal their cut passes in the first token and the next window's tokens and logits; each resumed from the knob's first kept state equals its whole pass", .{ world, cases.len, inner });
}
