//! TF_GLM_OVN_DISK_PUMP: states staged a piece a step through the ring write the bytes one-step staging writes.
const std = @import("std");
const platform = @import("sys");
const fixture = @import("rows_async_fixture.zig");
const Fixture = fixture.Fixture;

test "disk pump: files staged a block a step through the ring equal one-step staging, across ring wraps" {
    const whole = try Fixture.open("pump-off", .off, false, 1 << 20);
    defer whole.close();
    const pumped = try Fixture.open("pump-on", .off, false, 1 << 20);
    defer pumped.close();
    pumped.r.pump_blocks = 1; // four ring slots: three states of three blocks each wrap it twice
    for ([_]u32{ 7, 8, 9 }, 1..) |salt, n| {
        const tokens = fixture.prompt(salt);
        for ([_]*Fixture{ whole, pumped }) |x| {
            try x.keep(&tokens, 900);
            try x.idle(n);
        }
    }
    try std.testing.expect(pumped.r.pump == null);
    try std.testing.expectEqual(@as(u64, 4), pumped.r.worker.?.ring.?.count);
    try std.testing.expect(pumped.r.worker.?.ring.?.head >= 9);
    const d = platform.c.opendir(whole.r.dir) orelse return error.OpenFailed;
    defer _ = platform.c.closedir(d);
    var checked: usize = 0;
    while (platform.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (!std.mem.endsWith(u8, name, ".st") and !std.mem.endsWith(u8, name, ".blk")) continue;
        try std.testing.expectEqual(try fixture.digest(whole.r.dir, name), try fixture.digest(pumped.r.dir, name));
        checked += 1;
    }
    try std.testing.expectEqual(fixture.files(pumped.r.dir, ".blk") + fixture.files(pumped.r.dir, ".st"), checked);
    try std.testing.expectEqual(@as(usize, 3), fixture.files(pumped.r.dir, ".st"));
    try std.testing.expectEqual(@as(usize, 0), fixture.files(pumped.r.dir, ".part"));
}

test "disk pump: an entry dropped while it stages fails its job, frees its ring slots and writes nothing" {
    const x = try Fixture.open("pump-drop", .off, false, 1 << 20);
    defer x.close();
    x.r.pump_blocks = 1;
    const tokens = fixture.prompt(11);
    try x.keep(&tokens, 900);
    try std.testing.expect(x.r.pump != null);
    x.s.remove(0);
    try std.testing.expect(x.r.pump == null);
    try x.settle();
    const g = x.r.worker.?.ring.?;
    try std.testing.expectEqual(g.head, g.tail.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
}
