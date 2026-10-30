//! TF_GLM_OVN_ONE_CHUNK's step planning and its wire words, on the CPU (test-glm-snapshots-host).
const std = @import("std");
const glm = @import("glm");
const t = std.testing;
const oc = glm.one_chunk;
const C = glm.engine.prompt_rows;

/// The knob-off path's chunk ends from `start` to `end`: what a step's cuts must equal, with its own stops.
fn knobOff(start: usize, end: usize, len: usize, marks: []const u32, out: []u32) []u32 {
    var n: usize = 0;
    var x = start;
    while (true) {
        x = glm.engine.chunkEnd(x, len, marks);
        if (x >= end) return out[0..n];
        out[n] = @intCast(x);
        n += 1;
    }
}

fn expectCuts(p: oc.Plan, want: []const oc.Cut) !void {
    try t.expectEqual(want.len, p.n);
    for (want, p.cuts[0..p.n]) |w, c| try t.expectEqual(w, c);
}

test "GLM snapshots: one chunk runs a warm turn through both marks and the grid point, as the knob-off path cuts it" {
    const marks = [_]u32{ 200366, 202394 }; // the history's mark and the next turn's, 3 short of the prompt's end
    const p = oc.cut(200031, 202397, &marks, 196608, 8192, true);
    try t.expectEqual(@as(usize, 202397), p.end); // no realignment at 200,704: nothing is left to pair
    try expectCuts(p, &.{ .{ .at = 200366, .mark = true }, .{ .at = 200704, .mark = false }, .{ .at = 202394, .mark = true } });
    var buf: [8]u32 = undefined;
    const off = knobOff(p.start, p.end, 202397, &marks, &buf);
    try t.expectEqual(off.len, p.n);
    for (off, p.cuts[0..p.n]) |a, c| try t.expectEqual(a, c.at);
    try t.expectEqual(@as(usize, 2), p.marks());
}

test "GLM snapshots: one chunk ends at the grid when the rest does not fit, and starts on it as before" {
    const p = oc.cut(9, 3 * C, &.{ 100, C + 5 }, 0, 8192, true);
    try t.expectEqual(@as(usize, C), p.end);
    try expectCuts(p, &.{.{ .at = 100, .mark = true }});
    const q = oc.cut(C, 3 * C, &.{}, 0, 8192, true);
    try t.expectEqual(@as(usize, 2 * C), q.end);
    try t.expectEqual(@as(usize, 0), q.n);
    const r = oc.cut(C + 9, 2 * C - 7, &.{}, 0, 8192, true); // off the grid with the rest in one chunk: to the end
    try t.expectEqual(@as(usize, 2 * C - 7), r.end);
    try t.expectEqual(@as(usize, 0), r.n);
}

test "GLM snapshots: one chunk stops where the knob-off path rebases past its start, and rebases with it at its start" {
    const p = oc.cut(8000, 9000, &.{ 8100, 8997 }, 0, 8192, true);
    try t.expectEqual(@as(usize, 8192), p.end); // [8192, 8997) rebases at 8,192
    try expectCuts(p, &.{.{ .at = 8100, .mark = true }});
    const q = oc.cut(8192, 9000, &.{8997}, 0, 8192, true); // that next step rebases at its own start, as the cut one
    try t.expectEqual(@as(usize, 9000), q.end);
    try expectCuts(q, &.{.{ .at = 8997, .mark = true }});
}

test "GLM snapshots: one chunk stops at a mark past its slots, a mark it cannot keep and a cut past the header" {
    const p = oc.cut(0, 3000, &.{ 700, 1400, 2101, 2997 }, 0, 8192, true);
    try t.expectEqual(@as(usize, 2101), p.end);
    try t.expectEqual(@as(usize, oc.slots), p.marks());
    const q = oc.cut(0, 3000, &.{ 700, 1400 }, 0, 8192, false); // the MTP head or DRAFT_KEEPWIN: marks end steps
    try t.expectEqual(@as(usize, 700), q.end);
    try t.expectEqual(@as(usize, 0), q.n);
    const r = oc.cut(5, 2 * C, &.{ 100, 200, 300, 400, 500, 600, 700, 800 }, 0, 8192, true);
    try t.expect(r.marks() <= oc.slots and r.n <= oc.max_cuts);
}

test "GLM snapshots: one chunk's cuts cross in a chunk step's three free words; none there is no plan" {
    var h: glm.follow.Header = .{ .kind = @backingInt(glm.follow.Kind.chunk), .depth = 200031, .top_k = 202397 };
    try t.expectEqual(@as(?oc.Plan, null), try oc.read(&h, 200031, 2366));
    const p = oc.cut(200031, 202397, &.{ 200366, 202394 }, 196608, 8192, true);
    oc.put(&h, p);
    const back = (try oc.read(&h, 200031, 2366)).?;
    try t.expectEqual(p.end, back.end);
    try expectCuts(back, p.cuts[0..p.n]);
    oc.put(&h, null);
    try t.expectEqual(@as(f64, 0), h.temperature);
    var bad = h;
    oc.put(&bad, p);
    try t.expectError(error.BadStep, oc.read(&bad, 200400, 1997)); // a cut before the step's start
}
