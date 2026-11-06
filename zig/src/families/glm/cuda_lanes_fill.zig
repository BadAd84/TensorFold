//! The lane backend's fills past one chunk a call. TF_GLM_OVN_MULTI_FILL: prefill_many over fresh, marked and resumed
//! prompts; each pass takes every unfinished prompt's next piece (from where it stands to its next mark or its end),
//! shortest first, up to a chunk's rows, until every prompt's first token is drawn: Leader.chunk's cuts, several
//! streams a pass. TF_GLM_OVN_FILL_SLICE_MS: a chunk part-way through its layers (`onward`, `settle`).
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const pieces_mod = @import("cuda_prompt_pieces.zig");
const prompt_slices = @import("cuda_prompt_slices.zig");
const cuda_lanes = @import("cuda_lanes.zig");
const Lanes = cuda_lanes.Lanes;
const Lane = cuda_lanes.Lane;

const max = engine.max_together;

/// TF_GLM_OVN_FILL_SLICE_MS: the stream's chunk part-way through its layers runs its next slice (every layer left
/// once no other stream decodes); true once its prompt pass has its first token, also one `settle` drew.
pub fn onward(self: *Lanes, s: *lanes.Stream, l: *Lane) !bool {
    if (l.settled) {
        l.settled = false;
        s.slice_pending = false;
        return true;
    }
    errdefer {
        l.slicing = false;
        l.fill = null;
        s.slice_pending = false;
    }
    try self.x.bind(l.slot);
    const step = try self.x.slicePart(@import("cuda_prompt_solo.zig").othersDecode(&self.lanes, s));
    s.slice_pending = step.sliced;
    if (step.sliced) return false;
    l.slicing = false;
    if (step.first) |first| {
        l.first = self.take(first);
        l.fill = null;
        return true;
    }
    l.fill = l.fill.? + step.rows;
    if (s.settle_step) try self.x.e.stream.synchronize(); // its tail would land in the next decode round's time
    return false;
}

/// Before any other prompt work (it overwrites the prompt buffers): a chunk part-way through its layers runs all the
/// rest now; its stream's next prefill step then reports what it drew (`onward`) or goes on from the next chunk.
pub fn settle(self: *Lanes) !void {
    if (!prompt_slices.pending(self.x.e)) return;
    var it = self.lanes.valueIterator();
    while (it.next()) |l| if (l.slicing) {
        try self.x.bind(l.slot);
        const step = try self.x.slicePart(false);
        l.slicing = false;
        if (step.first) |first| {
            l.first = self.take(first);
            l.fill = null;
            l.settled = true;
        } else l.fill = l.fill.? + step.rows;
        return;
    };
    return error.ChunkPending; // a pending chunk with no lane: release forgets it, so none should be left
}

/// A prompt a batched pass takes: no prompt logprobs or images (their rows go through other steps), one chunk at most
/// past its kept state (a stale one then fills from 0 here, a chunk a pass).
pub fn admits(s: anytype) bool {
    const rows = if (s.reuse.saved != null) s.prompt_len -| s.reuse.at else s.prompt_len;
    return s.prompt_logprobs == null and s.images.len == 0 and rows <= engine.prompt_rows;
}

/// Where a piece from `start` ends: the first mark past it, else the prompt's end (Engine.chunkEnd inside a chunk).
pub fn cut(start: usize, len: usize, marks: []const u32) usize {
    return engine.chunkEnd(start, len, marks);
}

/// One pass's pieces: each unfinished stream by its next piece's rows, shortest first, ties by place, prompt_rows cap.
pub const Pick = struct { who: usize, end: usize };
pub fn plan(lens: []const usize, at: []const usize, marks: []const []const u32, out: *[max]Pick) usize {
    var order: [max]usize = undefined; // the unfinished streams, by their next piece's rows then their place
    var ends: [max]usize = undefined;
    var n: usize = 0;
    for (lens, at, marks, 0..) |len, a, mk, i| {
        if (a >= len) continue;
        ends[i] = cut(a, len, mk);
        var j = n;
        while (j > 0 and ends[order[j - 1]] - at[order[j - 1]] > ends[i] - a) : (j -= 1) order[j] = order[j - 1];
        order[j] = i;
        n += 1;
    }
    var rows: usize = 0;
    var k: usize = 0;
    for (order[0..n]) |i| {
        const r = ends[i] - at[i];
        if (k > 0 and rows + r > engine.prompt_rows) break; // shortest first: the rest are longer still
        out[k] = .{ .who = i, .end = ends[i] };
        rows += r;
        k += 1;
    }
    return k;
}

/// The streams' prompts through passes of pieces; false: refused before any stream opened (each then goes alone).
pub fn multiFill(self: *Lanes, ss: []const *lanes.Stream) !bool {
    if (!self.x.e.m.tiled or ss.len > max) return false;
    for (ss) |s| if (!admits(s)) return false;
    var opened: usize = 0;
    errdefer for (ss[0..opened]) |s| Lanes.releaseFn(self, s);
    var at: [max]usize = undefined; // where each prompt stands
    var from: [max]?engine.Resume = undefined; // its kept state, until its first piece restores it
    var marks: [max]?engine.Marks = undefined;
    var keeps: [max]@import("cuda_share.zig").Keeps = undefined; // each stream's, while its marks are read
    for (ss, 0..) |s, i| {
        from[i] = try self.opening(s);
        opened += 1;
        at[i] = if (from[i]) |r| r.at else 0;
        marks[i] = self.marksOf(s, &keeps[i]);
    }
    while (true) {
        var lens: [max]usize = undefined;
        var mts: [max][]const u32 = undefined; // each stream's mark positions (empty: none)
        for (ss, 0..) |s, i| {
            lens[i] = s.prompt_len;
            mts[i] = if (marks[i]) |m| m.at else &.{};
        }
        var picks: [max]Pick = undefined;
        const k = plan(lens[0..ss.len], at[0..ss.len], mts[0..ss.len], &picks);
        if (k == 0) return true;
        var pcs: [max]pieces_mod.Piece = undefined;
        var mk: [max]?engine.Marks = undefined;
        for (picks[0..k], 0..) |p, j| {
            const i = p.who;
            const ids = ss[i].prompt();
            const m = marks[i];
            const marked = at[i] > 0 and m != null and std.mem.indexOfScalar(u32, m.?.at, @intCast(at[i])) != null;
            pcs[j] = .{ .slot = self.lanes.getPtr(ss[i]).?.slot, .start = at[i], .rows = p.end - at[i], .len = ids.len, .part = ids[at[i]..@min(ids.len, p.end + 1)], .from = from[i], .mark = marked };
            mk[j] = m;
        }
        var firsts: [max]?u32 = undefined;
        try self.x.prefillPieces(pcs[0..k], mk[0..k], firsts[0..k]);
        for (picks[0..k], pcs[0..k], firsts[0..k]) |p, pc, f| {
            at[p.who] = pc.start + pc.rows;
            from[p.who] = null;
            if (f) |t| self.lanes.getPtr(ss[p.who]).?.first = self.take(t);
        }
    }
}

test "a piece ends at the first mark past its start, else at the prompt's end" {
    try std.testing.expectEqual(@as(usize, 300), cut(0, 300, &.{}));
    try std.testing.expectEqual(@as(usize, 120), cut(0, 300, &.{ 120, 200 }));
    try std.testing.expectEqual(@as(usize, 200), cut(120, 300, &.{ 120, 200 })); // its own start does not cut
    try std.testing.expectEqual(@as(usize, 300), cut(200, 300, &.{ 120, 200, 300 })); // a mark at the end is no cut
}

test "multifill plan: shortest next piece first, ties by place, finished streams skipped" {
    var out: [max]Pick = undefined;
    const e: []const u32 = &.{};
    {
        const lens = [_]usize{ 300, 100, 100, 200 };
        const at = [_]usize{ 0, 0, 0, 0 };
        const mk = [_][]const u32{ e, e, e, e };
        try std.testing.expectEqual(@as(usize, 4), plan(&lens, &at, &mk, &out));
        const got = [_]usize{ out[0].who, out[1].who, out[2].who, out[3].who };
        try std.testing.expectEqual([_]usize{ 1, 2, 3, 0 }, got); // 100 (place 1), 100 (place 2), 200, 300
    }
    { // the first done, the second half filled (its remaining rows shortest), the third fresh
        const lens = [_]usize{ 300, 300, 300 };
        const at = [_]usize{ 300, 128, 0 };
        const mk = [_][]const u32{ e, e, e };
        try std.testing.expectEqual(@as(usize, 2), plan(&lens, &at, &mk, &out));
        const got = [_]usize{ out[0].who, out[1].who };
        try std.testing.expectEqual([_]usize{ 1, 2 }, got);
    }
    { // every stream is done
        const lens = [_]usize{ 10, 20 };
        const at = [_]usize{ 10, 20 };
        const mk = [_][]const u32{ e, e };
        try std.testing.expectEqual(@as(usize, 0), plan(&lens, &at, &mk, &out));
    }
}

test "multifill plan: the pass is capped at prompt_rows, one piece always goes" {
    const R = engine.prompt_rows;
    var out: [max]Pick = undefined;
    const e: []const u32 = &.{};
    { // two halves of the budget fit together (the cap is a strict >, not >=)
        const lens = [_]usize{ R / 2, R / 2 };
        const at = [_]usize{ 0, 0 };
        const mk = [_][]const u32{ e, e };
        try std.testing.expectEqual(@as(usize, 2), plan(&lens, &at, &mk, &out));
    }
    { // just over the budget: only the first goes, the rest wait for the next pass
        const lens = [_]usize{ R / 2 + 1, R / 2 + 1 };
        const at = [_]usize{ 0, 0 };
        const mk = [_][]const u32{ e, e };
        try std.testing.expectEqual(@as(usize, 1), plan(&lens, &at, &mk, &out));
    }
    { // a lone full-width piece still goes (progress is never starved by the cap)
        const lens = [_]usize{R};
        const at = [_]usize{0};
        const mk = [_][]const u32{e};
        try std.testing.expectEqual(@as(usize, 1), plan(&lens, &at, &mk, &out));
    }
}

test "multifill plan: a piece ends at the first mark past its start" {
    var out: [max]Pick = undefined;
    const lens = [_]usize{1000};
    const at = [_]usize{0};
    const mk = [_][]const u32{&[_]u32{ 120, 500 }};
    try std.testing.expectEqual(@as(usize, 1), plan(&lens, &at, &mk, &out));
    try std.testing.expectEqual(@as(usize, 120), out[0].end);
}

test "multifill admits: logprobs, images and over-long prompts fall back to solo fills" {
    const S = struct {
        prompt_len: usize,
        reuse: struct { saved: ?*const u8 = null, at: usize = 0 } = .{},
        prompt_logprobs: ?*const u8 = null,
        images: []const u8 = &.{},
    };
    const keep: u8 = 0;
    try std.testing.expect(admits(S{ .prompt_len = 10 }));
    try std.testing.expect(admits(S{ .prompt_len = engine.prompt_rows }));
    try std.testing.expect(!admits(S{ .prompt_len = engine.prompt_rows + 1 }));
    try std.testing.expect(!admits(S{ .prompt_len = 10, .prompt_logprobs = &keep }));
    try std.testing.expect(!admits(S{ .prompt_len = 10, .images = &[_]u8{1} }));
    // a resumed prompt counts only the rows past its kept state
    try std.testing.expect(admits(S{ .prompt_len = engine.prompt_rows + 5, .reuse = .{ .saved = &keep, .at = 10 } }));
    try std.testing.expect(!admits(S{ .prompt_len = 2 * engine.prompt_rows, .reuse = .{ .saved = &keep, .at = 10 } }));
}
