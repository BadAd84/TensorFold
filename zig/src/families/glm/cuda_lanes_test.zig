//! cuda_lanes' checks on the host: a lane's slot chosen and given back, and when a prompt step may pair.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const Lanes = @import("cuda_lanes.zig").Lanes;

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}

test "a prompt step pairs only while no other stream is past its prompt pass" {
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const first = try self.claim(&streams[0]); // a stream's own lane, fresh as its first chunk starts
    try std.testing.expect(!self.othersDecode(&streams[0]));
    first.fill = fw.prefill_rows;
    (try self.claim(&streams[1])).fill = 2 * fw.prefill_rows; // another prompt pass, partway
    try std.testing.expect(!self.othersDecode(&streams[0]));
    _ = try self.claim(&streams[2]); // past its prompt pass: decoding
    try std.testing.expect(self.othersDecode(&streams[0]) and self.othersDecode(&streams[1]));
    try std.testing.expect(!self.othersDecode(&streams[2])); // the other two still fill
    _ = self.lanes.remove(&streams[2]); // released
    try std.testing.expect(!self.othersDecode(&streams[0]));
}

test "TF_GLM_OVN_SHARED_ROWS: the round loop's batch rows and shared costs reach the load's rows, 76 + 4.1 ms a row" {
    const gpa = std.testing.allocator;
    const cl = @import("cuda_lanes.zig");
    var e: engine.Engine = undefined;
    e.gpa = gpa;
    e.mtp = true;
    var dr: @import("dflash.zig").Drafter = undefined;
    e.drafter = &dr; // DFlash2: the windows priced by the cost tables
    for ([_]u32{ 32, 48, 64 }) |rows| {
        e.settings = .{ .shared_rows = rows };
        var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
        defer self.lanes.deinit(gpa);
        var cfg = try lanes.Config.init(gpa, self.facts(), cl.max_rows, cl.max_rows - 1);
        defer cfg.deinit(gpa);
        try std.testing.expectEqual(rows, cfg.batch_rows); // the window allocator's and the turn taker's cap
        try std.testing.expectEqual(@as(usize, rows + 1), cfg.shared_costs.slots.len);
        // the two-GB10 line itself at every width, not the table's 32-row end scaled up
        for ([_]u32{ 32, @min(33, rows), rows }) |w| try std.testing.expectApproxEqAbs(76.0 + 4.1 * @as(f64, @floatFromInt(w - 1)), cfg.shared_costs.get(w).?, 1e-9);
    }
}

test "TF_GLM_OVN_DEPTH_FULL and SHARED_COPY: each knob reaches the round loop's config, off by default" {
    const gpa = std.testing.allocator;
    const cuda = @import("cuda");
    const cl = @import("cuda_lanes.zig");
    defer cuda.ovn.fake("DEPTH_FULL", null);
    defer cuda.ovn.fake("SHARED_COPY", null);
    var e: engine.Engine = undefined;
    e.gpa = gpa;
    e.mtp = true;
    var dr: @import("dflash.zig").Drafter = undefined;
    e.drafter = &dr;
    e.settings = .{ .shared_rows = 32 };
    for ([_][2]?[]const u8{ .{ null, null }, .{ "1", null }, .{ null, "1" }, .{ "1", "1" }, .{ "0", "0" } }) |v| {
        cuda.ovn.fake("DEPTH_FULL", v[0]);
        cuda.ovn.fake("SHARED_COPY", v[1]);
        var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
        defer self.lanes.deinit(gpa);
        var cfg = try lanes.Config.init(gpa, self.facts(), cl.max_rows, cl.max_rows - 1);
        defer cfg.deinit(gpa);
        try std.testing.expectEqual(v[0] != null and v[0].?[0] == '1', cfg.shared_depth_full);
        try std.testing.expectEqual(v[1] != null and v[1].?[0] == '1', cfg.shared_copy_width);
    }
}

test "TF_GLM_OVN_WIN_FREE: the knob empties the shared table and leaves the lone window costs, off by default" {
    const gpa = std.testing.allocator;
    const cuda = @import("cuda");
    const cl = @import("cuda_lanes.zig");
    defer cuda.ovn.fake("WIN_FREE", null);
    var e: engine.Engine = undefined;
    e.gpa = gpa;
    e.mtp = true;
    var dr: @import("dflash.zig").Drafter = undefined;
    e.drafter = &dr;
    e.settings = .{ .shared_rows = 32 };
    for ([_]?[]const u8{ null, "0", "1" }) |v| {
        cuda.ovn.fake("WIN_FREE", v);
        var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
        defer self.lanes.deinit(gpa);
        var cfg = try lanes.Config.init(gpa, self.facts(), cl.max_rows, cl.max_rows - 1);
        defer cfg.deinit(gpa);
        const on = v != null and v.?[0] == '1';
        try std.testing.expectEqual(on, cfg.free_shared);
        try std.testing.expectEqual(on, cfg.shared_costs.empty()); // off: 76 + 4.1 ms a row to the load's rows
        // a lone stream's window costs are untouched either way
        try std.testing.expect(!cfg.family_costs.empty());
        try std.testing.expectApproxEqAbs(52.0, cfg.family_costs.get(1).?, 1e-9);
    }
}

test "structured output's grammar rows: a window past the load's shared rows is refused before the words are made" {
    var e: engine.Engine = undefined; // no device: a refusal allocates nothing
    e.c.vocab = 64; // two words a row
    var masks: [3 * 2]u32 = @splat(0);
    for ([_]u32{ 32, 48 }) |rows| {
        e.settings = .{ .shared_rows = rows };
        var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
        try std.testing.expectError(error.InvalidGrammarMasks, self.putBits(rows - 2, 3, &masks));
    }
}
