//! TF_GLM_PROF_ROUTE=1 (with TF_GLM_PROF=1): the distinct routed experts in each MoE layer of a verify window.
const std = @import("std");

/// select's widest expert set (cuda_glue.select's TopkTooWide), the bitset a layer's ids land in.
const max_experts = 512;
/// The i32s one layer's readback holds: rows x slots of a verify window (8 x 9) with room for wider tests.
pub const max_picks = 64 * 32;

pub const Route = struct {
    picks: [max_picks]i32 = undefined, // a layer's pick rows, read back from the device
    rows: usize = 0,
    pairs: usize = 0, // routed pairs a layer: rows x top_k
    layers: u64 = 0, // MoE layers counted since the last report
    distinct: u64 = 0, // their distinct experts, summed
    min: u32 = std.math.maxInt(u32),
    max: u32 = 0,

    /// TF_GLM_PROF_ROUTE=1 asks for it.
    pub fn wanted() bool {
        const v = std.c.getenv("TF_GLM_PROF_ROUTE") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// One MoE layer's picks [rows][slots], the first top_k of each row routed: its distinct experts counted.
    pub fn add(r: *Route, picks: []const i32, rows: usize, slots: usize, top_k: usize) !void {
        const n = try distinct(picks, rows, slots, top_k);
        r.rows = rows;
        r.pairs = rows * top_k;
        r.layers += 1;
        r.distinct += n;
        r.min = @min(r.min, n);
        r.max = @max(r.max, n);
    }

    /// The report line of `calls` windows (nothing when no layer was counted), then the sums cleared.
    pub fn report(r: *Route, w: *std.Io.Writer, calls: u64) void {
        defer r.* = .{};
        if (r.layers == 0) return;
        const mean = @as(f64, @floatFromInt(r.distinct)) / @as(f64, @floatFromInt(r.layers));
        const unit = if (r.rows == 1) "row" else "rows";
        w.print("glm prof route {d} {s} ({d} calls): distinct experts a layer mean {d:.1} (min {d}, max {d} over layers), pairs {d}", .{ r.rows, unit, calls, mean, r.min, r.max, r.pairs }) catch {};
    }
};

/// The distinct expert ids among the first top_k slots of each of `rows` rows of `slots` picks.
pub fn distinct(picks: []const i32, rows: usize, slots: usize, top_k: usize) !u32 {
    if (top_k > slots or rows * slots > picks.len) return error.RouteShape;
    var seen: std.StaticBitSet(max_experts) = .empty;
    for (0..rows) |row| for (picks[row * slots ..][0..top_k]) |id| {
        if (id < 0 or id >= max_experts) return error.RouteIdOutOfRange;
        seen.set(@intCast(id));
    };
    return @intCast(seen.count());
}

test "a layer's distinct experts count each id once over the window's routed slots, not the shared slot" {
    // 3 rows of top_k 2 plus a shared slot (id 9): {1, 4}, {4, 7}, {1, 7} share three experts
    const picks = [_]i32{ 1, 4, 9, 4, 7, 9, 1, 7, 9 };
    try std.testing.expectEqual(@as(u32, 3), try distinct(&picks, 3, 3, 2));
    try std.testing.expectEqual(@as(u32, 2), try distinct(&picks, 1, 3, 2));
    try std.testing.expectError(error.RouteIdOutOfRange, distinct(&[_]i32{ -1, 0 }, 1, 2, 2));
    try std.testing.expectError(error.RouteShape, distinct(&picks, 4, 3, 2));
}

test "a report's mean, min and max run over the layers counted, then clear" {
    var r: Route = .{};
    try r.add(&[_]i32{ 1, 4, 9, 4, 7, 9 }, 2, 3, 2); // 3 distinct
    try r.add(&[_]i32{ 5, 5, 9, 5, 5, 9 }, 2, 3, 2); // 1 distinct
    try r.add(&[_]i32{ 1, 2, 9, 3, 4, 9 }, 2, 3, 2); // 4 distinct
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    r.report(&w, 3);
    try std.testing.expectEqualStrings("glm prof route 2 rows (3 calls): distinct experts a layer mean 2.7 (min 1, max 4 over layers), pairs 4", w.buffered());
    try std.testing.expectEqual(@as(u64, 0), r.layers);
    var empty: std.Io.Writer = .fixed(&buf);
    r.report(&empty, 3);
    try std.testing.expectEqual(@as(usize, 0), empty.buffered().len);
}
