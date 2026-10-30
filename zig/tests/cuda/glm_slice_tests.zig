//! GLM-5.3-Flash's prompt chunks in layer slices (TF_GLM_OVN_FILL_SLICE_MS) against the same chunks whole.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `glm-slice-check` (every rank with TF_GLM_OVN_FILL_SLICE_MS=1: a slice of one layer): per prompt, A runs its chunks
/// whole, B runs each chunk a layer a slice with a decode window of another stream between slices, C is B with one
/// prompt token changed (a red control). A and B's first tokens, next window's tokens and logits must be equal bit for
/// bit, and C's logits must differ.
pub fn sliceCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
    if (e.slices == null) {
        std.debug.print("FAIL slice: TF_GLM_OVN_FILL_SLICE_MS is not set on rank {d}\n", .{rank});
        return error.TestFailed;
    }
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's whole and sliced prompt chunks to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    const C = glm.engine.prompt_rows;
    const decoder = 3; // the stream whose decode windows run between B's slices
    try x.bind(decoder);
    try x.begin(null);
    const short = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    var tok = (try x.chunk(&short, 0, false, null, null, null)).first.?;
    const lens = [_]usize{ C + 300, 2 * C - 7, 3 * C - 144 };
    var slices: usize = 0;
    var failed = false;
    for (lens, 0..) |len, i| {
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        const changed = try a.dupe(u32, ids);
        changed[len / 2] = if (changed[len / 2] == 4321) 4322 else 4321;
        var firsts: [3]u32 = undefined;
        var outs: [3][4]u32 = undefined;
        var logits: [3][]u16 = undefined;
        var pos: [3]usize = undefined;
        for (0..3) |k| { // A whole, B sliced with decode windows between, C sliced with a changed token
            const prompt = if (k == 2) changed else ids;
            try x.bind(@intCast(k));
            try x.begin(null);
            var at: usize = 0;
            firsts[k] = while (true) {
                var r = try x.chunkOf(prompt, at, false, null, null, null, k > 0);
                while (r.sliced) {
                    slices += 1;
                    try x.bind(decoder); // another stream's round between two slices, as the lane host runs one
                    var out: [2]u32 = undefined;
                    try x.verify(&[_]u32{ tok, 777 }, &out, 0);
                    try x.keep(1);
                    tok = out[0];
                    try x.bind(@intCast(k));
                    r = try x.slicePart(true);
                }
                if (r.first) |f| break f;
                at += r.rows;
            };
            const win = [_]u32{ firsts[k], 101, 202, 303 };
            try x.verify(&win, &outs[k], 0);
            logits[k] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[k].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
            pos[k] = e.cur.st.pos;
            try x.release(@intCast(k));
        }
        if (firsts[0] != firsts[1] or !std.mem.eql(u32, &outs[0], &outs[1]) or !std.mem.eql(u16, logits[0], logits[1]) or pos[0] != pos[1]) {
            const d = std.mem.indexOfDiff(u16, logits[0], logits[1]);
            std.debug.print("FAIL slice: prompt of {d}: sliced first {d}, whole {d}; first logit difference at {?d}\n", .{ len, firsts[1], firsts[0], d });
            failed = true;
        }
        if (std.mem.eql(u16, logits[0], logits[2])) {
            std.debug.print("FAIL slice: prompt of {d}: the red control (token {d} changed) left the logits unchanged\n", .{ len, len / 2 });
            failed = true;
        }
    }
    if (slices == 0) {
        std.debug.print("FAIL slice: no chunk ran in slices (TF_GLM_OVN_FILL_SLICE_MS too large for one layer?)\n", .{});
        failed = true;
    }
    if (failed) return error.TestFailed;
    check.pass("sliced prompt chunks at TP={d}: {d} prompts in {d} slices with another stream's decode window between slices equal their whole chunks in the first token and the next window's tokens and logits; every red control differed", .{ world, lens.len, slices });
}
