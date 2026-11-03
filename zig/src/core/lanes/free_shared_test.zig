//! TF_GLM_OVN_WIN_FREE: a shared round's windows allocate with no cost model, so they fill likeliest-first.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
const win = @import("windows.zig");
const fake = @import("fake.zig");

const gpa = std.testing.allocator;

/// GLM's shape (cuda_lanes.facts): 8-row windows, a first copy of 3, 7 drafts, 76 + 4.1 ms a shared row.
fn model(free: bool, rows: u32) !Config {
    var lone: [8]Cost = undefined;
    for (&lone, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 52.0 + 6.8 * @as(f64, @floatFromInt(w - 1)) };
    var shared: [64]Cost = undefined;
    for (shared[0..rows], 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 76.0 + 4.1 * @as(f64, @floatFromInt(w - 1)) };
    return Config.init(gpa, .{
        .exact_width = 8,
        .first_copy_rows = 4,
        .mtp = true,
        .speculate = true,
        .speculate_early = false,
        .drafts = 7,
        .draft_streams = true,
        .mtp_step_ms = 1.6 / 7.0,
        .draft_probabilities = true,
        .window_costs = &lone,
        .shared_costs = shared[0..rows],
        .hidden_rows = true,
        .batch_rows = rows,
        .max_streams = 16,
        .free_shared = free,
    }, 8, 7);
}

/// One shared round's windows: a stream per `held` count, allocated, and the drafts each keeps.
fn kept(cfg: *const Config, held: []const usize, out: []usize) !void {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [16]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    var tokens: [16][7]u32 = undefined;
    var plans: [16]win.Plan = undefined;
    for (held, 0..) |h, i| {
        streams[i] = try Stream.init(gpa, .{ .id = "shared", .prompt = &.{ 1, 2 }, .max_new = 64 });
        made += 1;
        for (tokens[i][0..h], 0..) |*t, j| t.* = @intCast(10 + j);
        plans[i] = .{ .stream = &streams[i], .position = 2, .kind = .head, .tokens = tokens[i][0..h] };
    }
    _ = e.arena.reset(.retain_capacity);
    try win.allocate(&e, plans[0..held.len]);
    for (plans[0..held.len], out[0..held.len]) |p, *o| o.* = p.count();
}

test "TF_GLM_OVN_WIN_FREE off: shared windows keep the counts the 76 + 4.1 ms table gives them" {
    var off = try model(false, 32);
    defer off.deinit(gpa);
    try std.testing.expect(!off.free_shared);
    try std.testing.expect(!off.shared_costs.empty());
    try std.testing.expectApproxEqAbs(76.0, off.shared_costs.get(1).?, 1e-9);
    try std.testing.expectApproxEqAbs(76.0 + 4.1 * 31.0, off.shared_costs.get(32).?, 1e-9);
    var rows: [8]usize = undefined;
    // four streams' seven-draft windows: three drafts each, the fourth draft's chance not paying for its row
    try kept(&off, &.{ 7, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3, 3, 3 }, rows[0..4]);
    // eight streams: two each, well short of the 32 rows the forward could hold
    try kept(&off, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2, 2, 2, 2, 2 }, rows[0..8]);
    // the same at 64 rows: the table prices the third draft the same either way, so the count does not move
    var wide = try model(false, 64);
    defer wide.deinit(gpa);
    try kept(&wide, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2, 2, 2, 2, 2 }, rows[0..8]);
}

test "TF_GLM_OVN_WIN_FREE on: shared windows fill likeliest-first, every draft held, to the forward's rows" {
    var on = try model(true, 64);
    defer on.deinit(gpa);
    try std.testing.expect(on.free_shared);
    try std.testing.expect(on.shared_costs.empty()); // no row priced: the allocator works from chances alone
    var rows: [8]usize = undefined;
    try kept(&on, &.{ 7, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7, 7 }, rows[0..4]); // the 28 drafts the 32-row forward holds
    try kept(&on, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, rows[0..8]); // 56 drafts, 64 rows
    var narrow = try model(true, 32);
    defer narrow.deinit(gpa);
    try kept(&narrow, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3, 3, 3, 3, 3, 3, 3 }, rows[0..8]); // 24 drafts, 32 rows
    try std.testing.expectEqual(@as(usize, 32), 8 + 24); // exactly the cap, so no row is left unused
}

test "TF_GLM_OVN_WIN_FREE on: a stream with less room than the cap keeps only its room" {
    var on = try model(true, 32);
    defer on.deinit(gpa);
    var rows: [4]usize = undefined;
    try kept(&on, &.{ 2, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 2, 7, 7, 7 }, &rows); // the short stream's own two, the rest to the cap
    try kept(&on, &.{ 1, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 1, 7 }, rows[0..2]);
    // priced, the roomy streams stop short of the cap: three drafts at 32 rows, four at a lone 8-row window
    var off = try model(false, 32);
    defer off.deinit(gpa);
    try kept(&off, &.{ 2, 7, 7, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 3, 3 }, &rows);
    try kept(&off, &.{ 1, 7 }, &rows);
    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, rows[0..2]);
    // six streams at 32 rows: the fill spends every row (32) where the table leaves eight unspent (24)
    var six: [6]usize = undefined;
    try kept(&on, &.{ 7, 7, 7, 7, 7, 7 }, &six);
    try std.testing.expectEqualSlices(usize, &.{ 5, 5, 4, 4, 4, 4 }, &six);
    try std.testing.expectEqual(@as(usize, 32), 6 + 26);
    try kept(&off, &.{ 7, 7, 7, 7, 7, 7 }, &six);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3, 3, 3, 3, 3 }, &six);
    try std.testing.expectEqual(@as(usize, 24), 6 + 18); // eight rows the priced allocator never fills
}

/// allocateDrafts' counts for `probs` at a round the depth rule sizes: `fixed` pending rows, `cap` rows at most.
fn drafts(probs: []const []const f64, costs: *const Table, fixed: u32, cap: u64) ![]u32 {
    const ones = try gpa.alloc(u32, probs.len);
    defer gpa.free(ones);
    @memset(ones, fixed);
    return alloc.allocateDrafts(gpa, ones, probs, costs, 8.0, cap, .{});
}

test "TF_GLM_OVN_WIN_FREE on: a shared round's budgets fill each stream to min(room, max_draft), ties to the lowest" {
    const empty: Table = .{};
    var costs = try alloc.extendCosts(gpa, &.{ .{ .width = 1, .ms = 76.0 }, .{ .width = 16, .ms = 76.0 + 4.1 * 15.0 } }, 16);
    defer costs.deinit(gpa);
    // equal chances: every row ties, so the fill runs stream 0's first draft, then stream 1's, in turn
    const level = [_][]const f64{ &.{ 0.5, 0.5, 0.5 }, &.{ 0.5, 0.5, 0.5 } };
    const tie = try drafts(&level, &empty, 1, 8);
    defer gpa.free(tie);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, tie);
    // a falling chain: the free fill takes every row whose chance is above zero, the priced one stops at three
    const falling = [_][]const f64{ &.{ 0.9, 0.3, 0.2, 0.1 }, &.{ 0.9, 0.3, 0.2, 0.1 } };
    const on = try drafts(&falling, &empty, 1, 16);
    defer gpa.free(on);
    try std.testing.expectEqualSlices(u32, &.{ 4, 4 }, on);
    const off = try drafts(&falling, &costs, 1, 16);
    defer gpa.free(off);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, off);
    // a stream with room for one draft keeps one, and the other takes the rest of the fill
    const mix = [_][]const f64{ &.{0.5}, &.{ 0.5, 0.2, 0.1 } };
    const free_mix = try drafts(&mix, &empty, 1, 16);
    defer gpa.free(free_mix);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, free_mix);
    const priced_mix = try drafts(&mix, &costs, 1, 16);
    defer gpa.free(priced_mix);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, priced_mix);
}

test "TF_GLM_OVN_WIN_FREE composes with DEPTH_FULL: deeper asks reach the window, and replies are unchanged" {
    for ([_]bool{ false, true }) |full| {
        var off = try replies(false, full);
        defer off.deinit();
        var on = try replies(true, full);
        defer on.deinit();
        for (off.emitted[0..off.n], on.emitted[0..on.n]) |want, got| try std.testing.expectEqualSlices(u32, want, got);
        try std.testing.expect(on.verified >= off.verified); // no row the priced arm admits is dropped by the free one
        try std.testing.expect(on.asked >= off.asked);
    }
}

/// Two streams' replies plus the drafts their shared rounds verified: a free target and a windowed one.
const Run = struct {
    n: usize = 2,
    emitted: [2][]u32 = undefined,
    verified: u64 = 0,
    asked: i64 = 0,

    fn deinit(r: *Run) void {
        for (r.emitted) |x| gpa.free(x);
    }
};

fn field(fields: []const @import("events.zig").Field, k: []const u8) ?@import("events.zig").Value {
    for (fields) |x| if (std.mem.eql(u8, x.k, k)) return x.v;
    return null;
}

fn replies(free: bool, full: bool) !Run {
    var cfg = try model(free, 32);
    defer cfg.deinit(gpa);
    cfg.shared_depth_full = full;
    const prompts = [_][6]u32{ .{ 3, 1, 4, 1, 5, 9 }, .{ 2, 7, 1, 8, 2, 8 } };
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var log = @import("events.zig").Log.init(gpa);
    defer log.deinit();
    e.log = &log;
    var streams: [2]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    for (&streams, &prompts, 0..) |*s, *p, i| {
        s.* = try Stream.init(gpa, .{ .id = "shared", .prompt = p, .max_new = @intCast(40 + 9 * i) });
        made += 1;
        try e.addStream(s);
    }
    while (e.activeCount() > 0) try e.step();
    var r: Run = .{};
    for (&streams, &r.emitted) |s, *o| o.* = try gpa.dupe(u32, s.emitted());
    for (log.events.items) |x| {
        const kind = (field(x, "ev") orelse continue).str;
        if (std.mem.eql(u8, kind, "budgets")) {
            for (field(x, "depths").?.i64s) |d| r.asked += d;
        }
        if (std.mem.eql(u8, kind, "alloc")) {
            for (field(x, "plans").?.list) |p| r.verified += @intCast(p.list[1].int);
        }
    }
    return r;
}
