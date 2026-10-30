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
    const step = try self.x.slicePart(self.othersDecode(s));
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

/// The streams' prompts through passes of pieces; false: refused before any stream opened (each then goes alone).
pub fn multiFill(self: *Lanes, ss: []const *lanes.Stream) !bool {
    if (!self.x.e.m.tiled or ss.len > max) return false;
    for (ss) |s| if (!admits(s)) return false;
    var opened: usize = 0;
    errdefer for (ss[0..opened]) |s| Lanes.releaseFn(self, s);
    var at: [max]usize = undefined; // where each prompt stands
    var from: [max]?engine.Resume = undefined; // its kept state, until its first piece restores it
    var marks: [max]?engine.Marks = undefined;
    for (ss, 0..) |s, i| {
        from[i] = try self.opening(s);
        opened += 1;
        at[i] = if (from[i]) |r| r.at else 0;
        marks[i] = Lanes.marksOf(s);
    }
    while (true) {
        var order: [max]usize = undefined; // unfinished prompts by their next piece's rows, then their place
        var ends: [max]usize = undefined;
        var n: usize = 0;
        for (ss, 0..) |s, i| {
            if (at[i] >= s.prompt_len) continue;
            ends[i] = cut(at[i], s.prompt_len, if (marks[i]) |m| m.at else &.{});
            var j = n;
            while (j > 0 and ends[order[j - 1]] - at[order[j - 1]] > ends[i] - at[i]) : (j -= 1) order[j] = order[j - 1];
            order[j] = i;
            n += 1;
        }
        if (n == 0) return true;
        var pcs: [max]pieces_mod.Piece = undefined;
        var mk: [max]?engine.Marks = undefined;
        var who: [max]usize = undefined;
        var rows: usize = 0;
        var k: usize = 0;
        for (order[0..n]) |i| {
            const r = ends[i] - at[i];
            if (k > 0 and rows + r > engine.prompt_rows) break; // shortest first: the rest are longer still
            const ids = ss[i].prompt();
            const m = marks[i];
            const marked = at[i] > 0 and m != null and std.mem.indexOfScalar(u32, m.?.at, @intCast(at[i])) != null;
            pcs[k] = .{ .slot = self.lanes.getPtr(ss[i]).?.slot, .start = at[i], .rows = r, .len = ids.len, .part = ids[at[i]..@min(ids.len, ends[i] + 1)], .from = from[i], .mark = marked };
            mk[k] = m;
            who[k] = i;
            rows += r;
            k += 1;
        }
        var firsts: [max]?u32 = undefined;
        try self.x.prefillPieces(pcs[0..k], mk[0..k], firsts[0..k]);
        for (who[0..k], pcs[0..k], firsts[0..k]) |i, p, f| {
            at[i] = p.start + p.rows;
            from[i] = null;
            if (f) |t| self.lanes.getPtr(ss[i]).?.first = self.take(t);
        }
    }
}

test "a piece ends at the first mark past its start, else at the prompt's end" {
    try std.testing.expectEqual(@as(usize, 300), cut(0, 300, &.{}));
    try std.testing.expectEqual(@as(usize, 120), cut(0, 300, &.{ 120, 200 }));
    try std.testing.expectEqual(@as(usize, 200), cut(120, 300, &.{ 120, 200 })); // its own start does not cut
    try std.testing.expectEqual(@as(usize, 300), cut(200, 300, &.{ 120, 200, 300 })); // a mark at the end is no cut
}
