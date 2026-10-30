//! TF_GLM_OVN_DISK_LOCAL on the host: another rank stood in by a fake Mirror that holds the files rank 0 staged.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const local = @import("rows_local.zig");
const chain = @import("rows_chain.zig");
const fixture = @import("rows_async_fixture.zig");
const Fixture = fixture.Fixture;
const a = fixture.a;

/// One other rank: the keys of the parts it holds, and what rank 0 asked of it.
const Other = struct {
    held: std.AutoHashMapUnmanaged(local.Key, void) = .empty,
    unlinked: std.ArrayList(local.Key) = .empty,
    staged: usize = 0,
    began: usize = 0,
    ended: usize = 0,
    rank0_ok: ?bool = null,
    missing: bool = false,
    swept: bool = false,
    mirror: local.Mirror = undefined,

    fn init(o: *Other) void {
        o.mirror = .{ .ctx = o, .file = file, .unlink = unlink, .other = other, .begin = begin, .end = end, .lacks = lacks, .sweep = sweep };
    }

    fn deinit(o: *Other) void {
        o.held.deinit(a);
        o.unlinked.deinit(a);
    }

    fn of(ctx: *anyopaque) *Other {
        return @ptrCast(@alignCast(ctx));
    }

    fn file(ctx: *anyopaque, _: pc.Saved, _: u32, f: local.File) anyerror!void {
        const o = of(ctx);
        try o.held.put(a, .{ .stem = chain.stem(f.name), .kind = f.kind }, {});
        o.staged += 1;
    }

    fn unlink(ctx: *anyopaque, k: local.Key) void {
        const o = of(ctx);
        _ = o.held.remove(k);
        o.unlinked.append(a, k) catch {};
    }

    fn other(_: *anyopaque, _: u64) void {}

    fn begin(ctx: *anyopaque, _: pc.Saved, _: u32, files: []const local.File) anyerror!void {
        const o = of(ctx);
        o.began += 1;
        o.missing = false;
        for (files) |f| if (!o.held.contains(.{ .stem = chain.stem(f.name), .kind = f.kind })) {
            o.missing = true;
        };
    }

    fn end(ctx: *anyopaque, ok: bool) anyerror!bool {
        const o = of(ctx);
        o.ended += 1;
        o.rank0_ok = ok;
        return ok and !o.missing;
    }

    fn lacks(ctx: *anyopaque, keys: []const local.Key, out: []bool) anyerror!void {
        const o = of(ctx);
        for (keys, out) |k, *x| x.* = !o.held.contains(k);
    }

    fn sweep(ctx: *anyopaque) anyerror!void {
        of(ctx).swept = true;
    }

    /// A block key it holds of state `name`'s (any one).
    fn blockOf(o: *const Other) ?local.Key {
        var it = o.held.keyIterator();
        while (it.next()) |k| if (k.kind == .block) return k.*;
        return null;
    }
};

fn unlinkedState(o: *const Other, name: chain.Name) bool {
    for (o.unlinked.items) |k| if (k.kind == .state and std.mem.eql(u8, &k.stem, &chain.stem(name))) return true;
    return false;
}

test "rows_local restore: every rank's part or none; a rank lacking its part drops the state for every rank" {
    const x = try Fixture.open("local-restore", .off, false, 1 << 20);
    defer x.close();
    var o: Other = .{};
    defer o.deinit();
    o.init();
    x.r.mirror = &o.mirror;
    x.r.readers = 3;
    const tokens = fixture.prompt(7);
    try x.keep(&tokens, 900);
    try x.idle(1);
    try x.settle();
    try std.testing.expect(o.staged >= 2); // its blocks and its own file, each staged on the other rank too
    // whole on both ranks: restored through the reader pool, byte for byte the image a fresh pass leaves
    const vt = x.f.snapshots().vtable;
    const want = try a.alloc(u8, @intCast(vt.bytes(&x.f, 900)));
    defer a.free(want);
    x.f.build(900, want);
    const saved = try rows.read.read(x.r, 0);
    const k: *@import("rows_fake.zig").Kept = @ptrCast(@alignCast(saved));
    try std.testing.expectEqualSlices(u8, want, k.img);
    vt.drop(&x.f, saved);
    try std.testing.expectEqual(@as(usize, 1), o.began);
    try std.testing.expectEqual(@as(?bool, true), o.rank0_ok);
    // the other rank loses one block: rank 0's own read is fine, the restore still fails on every rank
    _ = o.held.remove(o.blockOf().?);
    const name = x.r.states.items[0].name;
    const live = x.f.live;
    try std.testing.expectError(error.RankPartMissing, rows.read.read(x.r, 0));
    try std.testing.expectEqual(@as(usize, 2), o.ended);
    try std.testing.expectEqual(@as(?bool, true), o.rank0_ok);
    try std.testing.expectEqual(live, x.f.live); // the blank every rank leased is gone again
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len); // forgotten, so no later request tries it
    try std.testing.expect(unlinkedState(&o, name)); // and the other rank's part of it goes too
}

test "rows_local startup: a state whose block another rank lacks is forgotten everywhere; a whole one stays" {
    const x = try Fixture.open("local-agree", .off, false, 1 << 20);
    defer x.close();
    var o: Other = .{};
    defer o.deinit();
    o.init();
    x.r.mirror = &o.mirror;
    const first = fixture.prompt(7);
    try x.keep(&first, 900);
    try x.idle(1);
    const lost = o.held.count(); // the first state's keys: every block of it is its own (the prompts share none)
    const second = fixture.prompt(8);
    try x.keep(&second, 900);
    try x.idle(2);
    try x.settle();
    const gone = x.r.states.items[0].name;
    const kept = x.r.states.items[1].name;
    // the other rank lost a block of the first state only
    var it = o.held.keyIterator();
    var drop: ?local.Key = null;
    var seen: usize = 0;
    while (it.next()) |key| {
        seen += 1;
        if (key.kind == .block and drop == null and inState(x.r, 0, key.stem)) drop = key.*;
    }
    try std.testing.expect(seen > lost);
    _ = o.held.remove(drop.?);
    try std.testing.expectEqual(@as(usize, 1), try local.agree(x.r));
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expectEqualSlices(u8, &kept, &x.r.states.items[0].name);
    try std.testing.expect(unlinkedState(&o, gone));
    try std.testing.expect(o.swept);
}

/// Whether block stem `stem` is one state `i` names.
fn inState(r: *rows.Rows, i: usize, stem: chain.Stem) bool {
    var wants: std.ArrayList(rows.write.Want) = .empty;
    defer wants.deinit(a);
    r.wantsOf(a, &r.states.items[i], &wants) catch return false;
    for (wants.items) |w| if (std.mem.eql(u8, &chain.stem(w.name), &stem)) return true;
    return false;
}

test "rows_local pump: a state wider than the staging permit is skipped staged whole, and saved through the pump" {
    // ParityUpstream U6: one-step staging charges the whole state to the permit; the pump (DISK_LOCAL's default)
    // charges the state's own file alone and stages its blocks a few a round through the ring
    const permit: u64 = 28 << 10; // measured: staged whole skips below 34 KiB, the pump stages from 24 KiB
    for ([_]u32{ 0, 4 }) |blocks| {
        const x = try Fixture.open(if (blocks == 0) "local-permit-whole" else "local-permit-pump", .off, false, permit);
        defer x.close();
        x.r.pump_blocks = blocks;
        const tokens = fixture.prompt(9);
        try x.keep(&tokens, 1000);
        if (blocks == 0) {
            try std.testing.expectEqual(@as(u64, 1), x.r.counts.skipped);
            try std.testing.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
            continue;
        }
        try x.idle(1);
        try x.settle();
        try std.testing.expectEqual(@as(u64, 0), x.r.counts.skipped);
        const saved = try rows.read.read(x.r, 0);
        defer x.f.snapshots().vtable.drop(&x.f, saved);
        try x.f.snapshots().vtable.restore(&x.f, null, saved);
        try std.testing.expect(x.f.restored_ok);
    }
}
