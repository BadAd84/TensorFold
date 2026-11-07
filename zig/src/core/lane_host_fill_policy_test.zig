//! TENSORFOLD_PREFILL_POLICY through the lane host: a fake target records each fill's hint and order.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");
const Share = fill.Share;
const capRows = fill.capRows;
const run = fill.run;
const step = fill.step;
const longStep = fill.longStep;
const runnable = fill.runnable;

// TENSORFOLD_PREFILL_POLICY tests (the env is parsed in Share.fromEnv; these set host.share).
const PolicyCalls = struct {
    host: *LaneHost,
    streams: [16]*const @import("lanes").Stream = undefined,
    caps: [16]u32 = undefined,
    n: usize = 0,
    fn hook(ctx: *anyopaque, s: *@import("lanes").Stream, _: usize) void {
        const c: *PolicyCalls = @ptrCast(@alignCast(ctx));
        c.streams[c.n] = s;
        c.caps[c.n] = capRows(c.host); // the hint fillAt passed for this call
        c.n += 1;
    }
};

fn policyEvent(_: *anyopaque, _: @import("engine_api.zig").Id, _: *const @import("engine_api.zig").Event) void {}

/// A host with one live decoder (when `decoder`) and long prompts of `lens` rows admitted in order, then `share`.
fn policyHost(comptime body: fn (*LaneHost, *PolicyCalls) anyerror!void, decoder: bool, lens: []const usize, share: Share) !void {
    const lanes = @import("lanes");
    const api = @import("engine_api.zig");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    if (decoder) {
        const d: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 64, .drafts = false };
        try host.engine().submit(1, &d, .{ .ctx = &host, .event = policyEvent });
        try std.testing.expect(host.admitOne());
        while (host.filling.items.len > 0) _ = step(&host, 0);
        try std.testing.expectEqual(@as(usize, 1), runnable(&host));
    }
    const prompt: [4096]u32 = @splat(3);
    var requests: [3]api.Request = undefined;
    for (lens, 0..) |len, i| {
        requests[i] = .{ .prompt = prompt[0..len], .max_tokens = 8, .drafts = false };
        try host.engine().submit(@intCast(10 + i), &requests[i], .{ .ctx = &host, .event = policyEvent });
        try std.testing.expect(host.admitOne());
    }
    host.share = share;
    var calls: PolicyCalls = .{ .host = &host };
    target.prefill_hook = PolicyCalls.hook;
    target.prefill_hook_ctx = &calls;
    try body(&host, &calls);
}

test "prefill service: policy text and rows; unset or unknown is the default order, TOKENS clamped" {
    for ([_][]const u8{ "", "stock", "FCFS", "lifo", "batched", "x" }) |text| try std.testing.expectEqual(Share.Policy.stock, Share.policyOf(text));
    try std.testing.expectEqual(Share.Policy.fcfs, Share.policyOf("fcfs"));
    try std.testing.expectEqual(Share.Policy.fcfs, Share.policyOf(" fcfs\t"));
    try std.testing.expectEqual(Share.Policy.stock, (Share{}).policy);
    try std.testing.expectEqual(@as(u32, 2048), (Share{}).tokens);
    for ([_][]const u8{ "", "0", "x", "-1", "4294967296" }) |text| try std.testing.expectEqual(@as(u32, 2048), Share.tokensOf(text));
    for ([_]u32{ 2048, 4096, 8192, 16384 }) |n| { // the default and the measured options pass through unchanged
        var buf: [8]u8 = undefined;
        try std.testing.expectEqual(n, Share.tokensOf(try std.fmt.bufPrint(&buf, "{d}", .{n})));
    }
    try std.testing.expectEqual(@as(u32, 64), Share.tokensOf("1"));
    try std.testing.expectEqual(@as(u32, 64), Share.tokensOf("63"));
    try std.testing.expectEqual(@as(u32, 1000), Share.tokensOf("1000"));
    try std.testing.expectEqual(@as(u32, 8193), Share.tokensOf("8193"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("16384"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("16385"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("65536"));
}

test "prefill service: stock policy keeps the default hint and its one long step, call_rows or not" {
    try policyHost(struct {
        fn body(host: *LaneHost, calls: *PolicyCalls) !void {
            host.share.call_rows = 777; // read only under a policy
            try std.testing.expectEqual(@as(u32, 0), capRows(host));
            run(host);
            try std.testing.expectEqual(@as(usize, 1), calls.n); // one long step a visit
            try std.testing.expectEqual(@as(u32, 0), calls.caps[0]);
        }
    }.body, true, &.{ 2048, 2048, 2048 }, .{});
}

test "prefill service: fcfs fills only the oldest long prompt a visit, at TOKENS rows, beside a decoder" {
    try policyHost(struct {
        fn body(host: *LaneHost, calls: *PolicyCalls) !void {
            const first = &host.filling.items[0].stream;
            const second = &host.filling.items[1].stream;
            const third = &host.filling.items[2].stream;
            run(host);
            try std.testing.expectEqual(@as(usize, 1), calls.n);
            try std.testing.expectEqual(first, calls.streams[0]);
            try std.testing.expectEqual(@as(u32, 1024), calls.caps[0]);
            try std.testing.expectEqual(@as(u64, 1024), first.fill_at.load(.monotonic));
            try std.testing.expectEqual(@as(u64, 0), second.fill_at.load(.monotonic));
            try std.testing.expectEqual(@as(u64, 0), third.fill_at.load(.monotonic));
            try std.testing.expectEqual(@as(u32, 0), host.share.call_rows); // no hint outlives its call
            run(host); // the oldest again, to its end: it leaves the list and decodes
            try std.testing.expectEqual(first, calls.streams[1]);
            try std.testing.expectEqual(@as(usize, 2), host.filling.items.len);
            run(host);
            try std.testing.expectEqual(second, calls.streams[2]);
            try std.testing.expectEqual(@as(u32, 1024), calls.caps[2]);
        }
    }.body, true, &.{ 2048, 2048, 2048 }, .{ .policy = .fcfs, .tokens = 1024 });
}

test "srpt: a prompt's steps teach its row time only where the backend reports them" {
    const lanes = @import("lanes");
    const api = @import("engine_api.zig");
    const gpa = std.testing.allocator;
    for ([_]bool{ true, false }) |reports| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 2 };
        defer target.deinit();
        var vt = target.stepped().vtable.*;
        vt.reports_fill = reports;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        defer {
            @import("lane_host_end.zig").closeAll(&host);
            host.stop();
        }
        host.srpt.on = true;
        const prompt: [2048]u32 = @splat(3);
        const request: api.Request = .{ .prompt = &prompt, .max_tokens = 1, .drafts = false };
        try host.engine().submit(1, &request, .{ .ctx = &host, .event = policyEvent });
        try std.testing.expect(host.admitOne());
        _ = longStep(&host, 0); // half of it: a reporting backend shows those rows
        try std.testing.expectEqual(reports, host.srpt.fill_ns_row > 0);
        _ = longStep(&host, 0); // its last: a silent backend's whole prompt in one step is no row's time
        try std.testing.expectEqual(@as(usize, 0), host.filling.items.len);
        try std.testing.expectEqual(reports, host.srpt.fill_ns_row > 0);
    }
}

test "prefill service: fcfs alone with nothing decoding takes the engine's own cut" {
    try policyHost(struct {
        fn body(host: *LaneHost, calls: *PolicyCalls) !void {
            run(host);
            try std.testing.expectEqual(@as(usize, 1), calls.n);
            try std.testing.expectEqual(@as(u32, 0), calls.caps[0]);
            try std.testing.expectEqual(@as(usize, 1), host.filling.items.len); // the whole prompt went in
        }
    }.body, false, &.{ 2048, 2048 }, .{ .policy = .fcfs, .tokens = 1024 });
}
