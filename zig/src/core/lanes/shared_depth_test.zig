//! DEPTH_FULL and SHARED_COPY on the fake target: shared rounds ask deeper and copy wider, replies stay the same.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;
const fake = @import("fake.zig");
const ev = @import("events.zig");
const t = std.testing;
const gpa = t.allocator;

const streams_most = 6;
const names = [_][]const u8{ "a", "b", "c", "d", "e", "f" };
const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
const prompts = [_][6]u32{ .{ 3, 1, 4, 1, 5, 9 }, .{ 2, 7, 1, 8, 2, 8 }, .{ 1, 6, 1, 8, 0, 3 }, .{ 1, 4, 1, 4, 2, 1 } };

/// GLM's shape (cuda_lanes.zig facts): 8-row windows, 7 drafts, a first copy of 3, 32 shared rows at 76 + 4.1 ms a row.
fn glmModel(depth_full: bool, shared_copy: bool) !Config {
    var costs: [32]Cost = undefined;
    for (&costs, 1..) |*c, row| c.* = .{ .width = @intCast(row), .ms = 76.0 + 4.1 * @as(f64, @floatFromInt(row - 1)) };
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
        .window_costs = costs[0..8],
        .shared_costs = &costs,
        .hidden_rows = true,
        .batch_rows = 32,
        .max_streams = 16,
        .shared_depth_full = depth_full,
        .shared_copy_width = shared_copy,
    }, 8, 7);
}

const Run = struct {
    emitted: [streams_most][]u32 = undefined,
    n: usize = 0,
    steps: u64 = 0,
    asked: i64 = 0, // head drafts asked over every shared round's budgets
    asked_full: u64 = 0, // budgets that asked most_drafts (7)
    copies: u64 = 0, // copy windows verified in shared rounds
    copy_widest: u64 = 0, // the most copied drafts one shared window verified
    grown: ?u32 = null, // the first stream's copy width when the others were admitted
    record: [32]u8 = undefined, // the decision log's lines and every reply's tokens, hashed
    verified: u64 = 0, // drafts verified in shared rounds, every window's

    fn deinit(r: *Run) void {
        for (r.emitted[0..r.n]) |x| gpa.free(x);
    }
};

fn field(fields: []const ev.Field, k: []const u8) ?ev.Value {
    for (fields) |x| if (std.mem.eql(u8, x.k, k)) return x.v;
    return null;
}

/// One arm: the knobs, a reply loop copies follow, swinging head chances, and steps the first stream runs alone.
const Arm = struct { depth_full: bool = false, shared_copy: bool = false, cycle: bool = false, swing: bool = false, alone_first: usize = 0 };

/// `n` streams admitted (the first `alone_first` steps ahead) and stepped to the end, with the arm's knobs and target.
fn run(n: usize, arm: Arm) !Run {
    var cfg = try glmModel(arm.depth_full, arm.shared_copy);
    defer cfg.deinit(gpa);
    const pattern = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = if (arm.cycle) 12 else null, .probe_prompt = 6, .pattern = &pattern };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, if (arm.swing) target.swung() else target.backend(), clock.clock());
    defer e.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    e.log = &log;
    var streams: [streams_most]Stream = undefined;
    var looks: [streams_most]SuffixLookup = undefined;
    var made: usize = 0;
    defer for (streams[0..made], looks[0..made]) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    var grown: ?u32 = null;
    for (streams[0..n], looks[0..n], 0..) |*s, *p, i| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        const spec_sampling: ?Sampling = if (i % 3 == 2) sampled else null;
        const longer: usize = if (i == 0 and arm.alone_first > 0) 120 else 0; // the first stream: alone, then shared
        const max_new: u32 = @intCast(60 + 7 * i + longer);
        s.* = try Stream.init(gpa, .{ .id = names[i], .prompt = &prompts[i % prompts.len], .max_new = max_new, .sampling = spec_sampling, .proposer = p.proposer() });
        made += 1;
        try e.addStream(s);
        if (i == 0) for (0..arm.alone_first) |_| try e.step();
        if (i == 0) grown = s.copy_width;
    }
    while (e.activeCount() > 0) try e.step();
    var r: Run = .{ .n = n, .steps = e.steps, .grown = grown };
    var made_out: usize = 0;
    errdefer for (r.emitted[0..made_out]) |x| gpa.free(x);
    for (streams[0..n], r.emitted[0..n]) |*s, *o| {
        o.* = try gpa.dupe(u32, s.emitted());
        made_out += 1;
    }
    var shared = false;
    for (log.events.items) |x| {
        const kind = (field(x, "ev") orelse continue).str;
        if (std.mem.eql(u8, kind, "step")) shared = false;
        if (std.mem.eql(u8, kind, "turns")) shared = true;
        if (std.mem.eql(u8, kind, "budgets")) for (field(x, "depths").?.i64s) |d| {
            r.asked += d;
            r.asked_full += @intFromBool(d == cfg.most_drafts);
        };
        if (shared and std.mem.eql(u8, kind, "alloc")) for (field(x, "plans").?.list) |p| {
            const count: u64 = @intCast(p.list[1].int);
            r.verified += count;
            if (!std.mem.eql(u8, p.list[0].str, "copy")) continue;
            r.copies += 1;
            r.copy_widest = @max(r.copy_widest, count);
        };
    }
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..log.events.items.len) |i| {
        const line = try log.line(gpa, i);
        defer gpa.free(line);
        hash.update(line);
        hash.update("\n");
    }
    for (r.emitted[0..n]) |x| hash.update(std.mem.sliceAsBytes(x));
    r.record = hash.finalResult();
    std.debug.print("SHARED_KNOBS n={d} {any} steps={d} asked={d} asked7={d} copies={d} widest={d} grown={?d} verified={d}\n", .{ n, arm, r.steps, r.asked, r.asked_full, r.copies, r.copy_widest, r.grown, r.verified });
    return r;
}

fn expectSameReplies(base: *const Run, other: *const Run) !void {
    for (base.emitted[0..base.n], other.emitted[0..other.n]) |want, got| try t.expectEqualSlices(u32, want, got);
}

test "TF_GLM_OVN_DEPTH_FULL: shared rounds ask every head for most_drafts, and every stream's reply is unchanged" {
    for ([_]usize{ 2, 4, 6 }) |n| for ([_]bool{ false, true }) |cycle| for ([_]bool{ false, true }) |swing| {
        var off = try run(n, .{ .cycle = cycle, .swing = swing });
        defer off.deinit();
        var on = try run(n, .{ .depth_full = true, .cycle = cycle, .swing = swing });
        defer on.deinit();
        try expectSameReplies(&off, &on);
        try t.expect(on.asked > off.asked); // the grant + 2 ratchet no longer holds the ask down
        try t.expect(on.asked_full > off.asked_full);
        if (cycle) try t.expect(off.copies > 0 and on.copies > 0); // copies fired in shared rounds on both arms
        if (swing) try t.expect(on.verified > off.verified); // a sure round after an unsure one verifies its drafts
    };
}

test "TF_GLM_OVN_SHARED_COPY: shared copies grow past first_copy as a lone stream's do, and replies are unchanged" {
    for ([_]usize{ 2, 4, 6 }) |n| {
        var off = try run(n, .{ .cycle = true });
        defer off.deinit();
        var on = try run(n, .{ .shared_copy = true, .cycle = true });
        defer on.deinit();
        var both = try run(n, .{ .depth_full = true, .shared_copy = true, .cycle = true, .swing = true });
        defer both.deinit();
        try expectSameReplies(&off, &on);
        try expectSameReplies(&off, &both);
        try t.expect(off.copies > 0 and on.copies > 0);
        try t.expect(off.copy_widest <= 3); // first_copy: 3 drafts
        try t.expect(on.copy_widest > 3);
    }
    // a stream whose copies grew alone: knob off, its shared copies drop back to first_copy; on, they keep its width
    var alone_off = try run(3, .{ .cycle = true, .alone_first = 10 });
    defer alone_off.deinit();
    var alone_on = try run(3, .{ .shared_copy = true, .cycle = true, .alone_first = 10 });
    defer alone_on.deinit();
    try expectSameReplies(&alone_off, &alone_on);
    try t.expect(alone_off.grown.? > 3 and alone_off.copies > 0 and alone_off.copy_widest <= 3);
    try t.expect(alone_on.copy_widest > 3);
}

test "TF_GLM_OVN_DEPTH_FULL and SHARED_COPY: replies match the fake target's own one-token decode" {
    var on = try run(4, .{ .depth_full = true, .shared_copy = true, .swing = true });
    defer on.deinit();
    for (on.emitted[0..4], 0..) |got, i| {
        const s: ?Sampling = if (i % 3 == 2) sampled else null;
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &prompts[i]);
        try t.expectEqual(@as(usize, 60 + 7 * i), got.len);
        for (got) |token| {
            try t.expectEqual(fake.next(history.items, s, history.items.len), token);
            try history.append(gpa, token);
        }
    }
}

// Hashes taken at 7b483cebd, before either knob existed, with this file and fake.zig's `swung` added.
test "TF_GLM_OVN_DEPTH_FULL and SHARED_COPY off: every decision and reply is the record from before the knobs" {
    const Pinned = struct { n: usize, arm: Arm, hex: []const u8 };
    const pinned = [_]Pinned{
        .{ .n = 3, .arm = .{ .cycle = true, .alone_first = 10 }, .hex = "0f7b9160a1e99c6ef122e32bd7b6268d219122a47e43ca6ac0df8f2e20e9232d" },
        .{ .n = 4, .arm = .{ .cycle = true, .swing = true }, .hex = "a444a3d6e0faf1272e017b1f28102ea458e1e50d9449e8ef3985793365b6e5fd" },
        .{ .n = 6, .arm = .{ .swing = true, .alone_first = 10 }, .hex = "67dd53c74f6fed2cc6e2da4af4f98ed7fc149cb2f6a57b289d0538311788eeb3" },
    };
    for (pinned) |p| {
        var off = try run(p.n, p.arm);
        defer off.deinit();
        const got = std.fmt.bytesToHex(off.record, .lower);
        std.debug.print("RECORD n={d} {s}\n", .{ p.n, &got });
        try t.expectEqualStrings(p.hex, &got);
    }
}
