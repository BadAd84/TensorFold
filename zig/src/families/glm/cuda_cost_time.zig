//! TF_GLM_OVN_COST_TIME: the lane host's window costs timed at load (DFlash2 only) instead of the two-GB10 constants.
//! One `costs` step: every rank runs `run` (the same forwards, so the exchanges pair up) on small sequences in slots
//! 1..8, verify windows of 1..8 rows and shared windows of 2, 4 and 8 streams (and, past 32 shared rows, 5 to 8
//! streams of 8 rows up to TF_GLM_OVN_SHARED_ROWS), then frees all it made; rank 0 keeps each shape's fastest pass.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const Seq = @import("cuda_engine_types.zig").Seq;
const Engine = engine.Engine;
const Cost = lanes.config.Cost;

const first_slot = 1; // the timed sequences' slots: no stream holds one before the server starts
const slots = 8;
const capacity = 1024; // each timed sequence's tokens: the passes below commit about a hundred rows
const reps = 5;
const max_rows = fw.window_rows;

/// The timed tables by rows: shared to 64 rows, past the load's TF_GLM_OVN_SHARED_ROWS extrapolated and unread.
pub const Timed = struct { window: [max_rows]Cost, shared: [engine.max_shared_rows]Cost };

/// A timed shared window: `streams` windows of `rows` rows each.
pub const Shape = struct { streams: usize, rows: usize };
const most_shapes = 3 * 3 + (slots - 4);

/// Shared shapes timed at `rows`: 2/4/8 streams x 1/2/4 rows (as at 32), then 5 to 8 streams of 8 rows up to `rows`.
pub fn shapes(rows: usize, out: *[most_shapes]Shape) []const Shape {
    var n: usize = 0;
    for ([_]usize{ 2, 4, 8 }) |s| for ([_]usize{ 1, 2, 4 }) |r| {
        out[n] = .{ .streams = s, .rows = r };
        n += 1;
    };
    for (5..slots + 1) |s| if (s * max_rows <= rows) {
        out[n] = .{ .streams = s, .rows = max_rows };
        n += 1;
    };
    return out[0..n];
}

/// Whether this load times its costs: the knob, DFlash2 (the only user of the tables), shared windows, no vision
/// (an image set binds per slot). Every rank decides alike: these are agreed settings.
pub fn wanted(e: *const Engine) bool {
    return cuda.ovn.on("COST_TIME") and e.drafter != null and e.bs != null and e.images == null;
}

fn now(e: *const Engine) i96 {
    return std.Io.Clock.awake.now(e.io).toNanoseconds();
}

fn ms(e: *const Engine, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(e) - t0)) / 1e6;
}

/// Every rank (Leader.costs, then the `costs` step): the timed passes; rank 0's tables (the others' are unused).
pub fn run(e: *Engine) !Timed {
    const end = first_slot + slots;
    if (e.seqs.items.len < end) try e.seqs.appendNTimes(e.gpa, null, end - e.seqs.items.len);
    for (e.seqs.items[first_slot..end]) |q| if (q != null) return error.SlotTaken;
    defer { // every timed slot freed (Engine.release), and the drafter rings its passes made, as before the step
        for (first_slot..end) |slot| e.release(@intCast(slot));
        if (e.drafter) |dr| if (dr.slots.items.len > first_slot) for (dr.slots.items[first_slot..@min(end, dr.slots.items.len)]) |*s| {
            if (s.ring) |*r| r.free();
            s.* = .{};
        };
    }
    for (e.seqs.items[first_slot..end]) |*q| { // small sequences, so no rank makes a full one on these slots
        const s = try e.gpa.create(Seq);
        errdefer e.gpa.destroy(s);
        s.* = .{ .st = try fw.State.initFormat(e.gpa, e.d, &e.w, capacity, e.settings.kv8) };
        q.* = s;
    }
    var ids: [engine.max_shared_rows]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + 7 * i); // distinct tokens
    var out: [engine.max_shared_rows]u32 = undefined;
    for (first_slot..end) |slot| {
        try e.bind(@intCast(slot));
        try e.begin(null);
    }
    var window: [max_rows]f64 = @splat(std.math.inf(f64));
    try e.bind(first_slot);
    try e.verify(ids[0..max_rows], out[0..max_rows], 0); // warm-up
    try e.keep(1);
    for (0..reps) |_| for (1..max_rows + 1) |r| {
        const t0 = now(e);
        try e.verify(ids[0..r], out[0..r], 0);
        window[r - 1] = @min(window[r - 1], ms(e, t0));
        try e.keep(1);
    };
    var shape_buf: [most_shapes]Shape = undefined;
    const timed = shapes(e.settings.shared_rows, &shape_buf);
    var shared: [most_shapes]f64 = @splat(std.math.inf(f64));
    var parts: [slots]engine.Part = undefined;
    for (0..reps) |_| for (timed, 0..) |sh, si| {
        const n = sh.streams;
        const r = sh.rows;
        for (parts[0..n], 0..) |*p, k| p.* = .{ .slot = @intCast(first_slot + k), .ids = ids[k * r ..][0..r], .out = out[k * r ..][0..r] };
        const t0 = now(e);
        try e.verifyShared(parts[0..n]);
        shared[si] = @min(shared[si], ms(e, t0));
        for (parts[0..n]) |p| {
            try e.bind(p.slot);
            try e.keep(1);
        }
    };
    var t: Timed = undefined;
    for (&t.window, window, 1..) |*c, v, r| c.* = .{ .width = @intCast(r), .ms = v };
    var points: [most_shapes]Cost = undefined; // each shape's total rows and its time
    for (timed, shared[0..timed.len], points[0..timed.len]) |sh, v, *p| p.* = .{ .width = @intCast(sh.streams * sh.rows), .ms = v };
    t.shared = sharedTable(points[0..timed.len]);
    if (e.share.rank == 0) std.log.info("glm costs (TF_GLM_OVN_COST_TIME): window ms by rows {any}; shared shapes (streams x rows) {any}, their ms {any}", .{ window, timed, shared[0..timed.len] });
    return t;
}

/// The shared table by total rows: each measured total's mean, the line between them, extended past the ends.
fn sharedTable(points: []const Cost) [engine.max_shared_rows]Cost {
    var known: [engine.max_shared_rows + 1]?f64 = @splat(null);
    var count: [engine.max_shared_rows + 1]f64 = @splat(0);
    for (points) |p| {
        known[p.width] = (known[p.width] orelse 0) + p.ms;
        count[p.width] += 1;
    }
    var xs: [engine.max_shared_rows]u32 = undefined;
    var ys: [engine.max_shared_rows]f64 = undefined;
    var m: usize = 0;
    for (known, count, 0..) |k, c, w| if (k) |sum| {
        xs[m] = @intCast(w);
        ys[m] = sum / c;
        m += 1;
    };
    var out: [engine.max_shared_rows]Cost = undefined;
    for (&out, 1..) |*o, w| {
        var j: usize = 0; // the segment [j, j + 1] that holds w, or the first or last one
        while (j + 2 < m and w > xs[j + 1]) j += 1;
        const x0: f64 = @floatFromInt(xs[j]);
        const x1: f64 = @floatFromInt(xs[j + 1]);
        const at: f64 = @floatFromInt(w);
        o.* = .{ .width = @intCast(w), .ms = ys[j] + (ys[j + 1] - ys[j]) * (at - x0) / (x1 - x0) };
    }
    return out;
}

test "shared costs: measured totals averaged, lines between them, extended at both ends" {
    const pts = [_]Cost{ .{ .width = 2, .ms = 50 }, .{ .width = 4, .ms = 60 }, .{ .width = 4, .ms = 64 }, .{ .width = 8, .ms = 80 }, .{ .width = 32, .ms = 200 } };
    const t = sharedTable(&pts);
    try std.testing.expectEqual(@as(u32, 1), t[0].width);
    try std.testing.expectApproxEqAbs(@as(f64, 44), t[0].ms, 1e-9); // below 2: the 2..4 line (50, 62)
    try std.testing.expectApproxEqAbs(@as(f64, 62), t[3].ms, 1e-9); // 4: the mean of 60 and 64
    try std.testing.expectApproxEqAbs(@as(f64, 71), t[5].ms, 1e-9); // 6: halfway from 62 to 80
    try std.testing.expectApproxEqAbs(@as(f64, 200), t[31].ms, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 140), t[19].ms, 1e-9); // 20: halfway from 80 to 200
}

test "COST_TIME's shared shapes: today's nine at 32 rows; 8-row windows of 5 to 8 streams up to the load's rows" {
    var buf: [most_shapes]Shape = undefined;
    const at32 = shapes(32, &buf);
    try std.testing.expectEqual(@as(usize, 9), at32.len);
    const streams = [_]usize{ 2, 4, 8 };
    const per = [_]usize{ 1, 2, 4 };
    for (at32, 0..) |sh, i| { // streams 2, 4, 8 by rows 1, 2, 4, in today's order
        try std.testing.expectEqual(streams[i / 3], sh.streams);
        try std.testing.expectEqual(per[i % 3], sh.rows);
    }
    for ([_]usize{ 40, 48, 56, 64 }) |rows| {
        const s = shapes(rows, &buf);
        var widest: usize = 0;
        for (s) |sh| {
            try std.testing.expect(sh.streams <= slots and sh.rows <= max_rows);
            widest = @max(widest, sh.streams * sh.rows);
        }
        try std.testing.expectEqual(rows, widest); // the load's width itself is measured
        try std.testing.expectEqual(9 + (rows - 32) / 8, s.len); // and every 8 rows past 32 on the way
    }
}

test "shared costs past 32 rows: a measured 64-row point holds the line out to 64" {
    const pts = [_]Cost{ .{ .width = 2, .ms = 50 }, .{ .width = 32, .ms = 200 }, .{ .width = 64, .ms = 330 } };
    const t = sharedTable(&pts);
    try std.testing.expectEqual(@as(usize, 64), t.len);
    try std.testing.expectApproxEqAbs(@as(f64, 330), t[63].ms, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 265), t[47].ms, 1e-9); // 48: halfway from 200 to 330
    try std.testing.expectEqual(@as(u32, 64), t[63].width);
}
