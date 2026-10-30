//! TF_GLM_OVN_DISK_CARRY's replay, one file: a stored file's payload against the same ranges of a fresh image.
const std = @import("std");
const platform = @import("sys");
const file = @import("rows_file.zig");
const image = @import("rows_image.zig");

/// One rank's answer about one replayed state: its files hold the fresh bytes, one cannot be read, or where they part.
pub const Compare = union(enum) { same, unreadable, differs: u64 };

/// An answer as the word every rank gathers: 0 the same, 1 unreadable, else the first differing image offset plus 2.
pub fn word(c: Compare) u64 {
    return switch (c) {
        .same => 0,
        .unreadable => 1,
        .differs => |off| off +| 2,
    };
}

/// A gathered word back as its answer.
pub fn answer(w: u64) Compare {
    return switch (w) {
        0 => .same,
        1 => .unreadable,
        else => .{ .differs = w - 2 },
    };
}

/// A fresh image's bytes: `ranges` of it into `out`, back to back.
pub const Fresh = struct {
    ctx: *anyopaque,
    gather: *const fn (ctx: *anyopaque, ranges: []const image.Range, out: []u8) anyerror!void,
};

/// File `path` opened when a restore would take it: its kind, full name, block, position and payload's length.
pub fn opens(path: [:0]const u8, kind: file.Kind, name: file.Name, block: u32, at: u32, payload: u64) ?file.Opened {
    var o = file.open(path, kind, name) catch return null;
    if (o.trailer.payload == payload and o.trailer.block == block and o.trailer.at == at) return o;
    o.close();
    return null;
}

/// File `path` (as `opens` takes it) against `ranges` of a fresh image: its payload's hash, then every byte.
pub fn same(path: [:0]const u8, kind: file.Kind, name: file.Name, block: u32, at: u32, ranges: []const image.Range, stage: file.Stage, buf: []u8, fresh: Fresh) !Compare {
    const n: usize = @intCast(image.total(ranges));
    if (buf.len < n) return error.CarryBuffer;
    var o = opens(path, kind, name, block, at, n) orelse return .unreadable;
    defer o.close();
    const got = file.payload(&o, stage) catch |err| return if (err == error.RowsStage) err else .unreadable;
    try fresh.gather(fresh.ctx, ranges, buf[0..n]);
    const i = std.mem.indexOfDiff(u8, got, buf[0..n]) orelse return .same;
    var done: usize = 0;
    for (ranges) |r| {
        const len: usize = @intCast(r.len);
        if (i < done + len) return .{ .differs = r.off + (i - done) };
        done += len;
    }
    return error.CarryRanges;
}

/// A test's image: byte `i` of it is a function of `i`, plus `salt` for `i` in [from, to).
const Fake = struct {
    salt: u8 = 0,
    from: u64 = 0,
    to: u64 = 0,

    fn byte(f: *const Fake, i: u64) u8 {
        const v: u8 = @truncate(i *% 31 +% 7);
        return if (i >= f.from and i < f.to) v +% f.salt else v;
    }

    fn gather(ctx: *anyopaque, ranges: []const image.Range, out: []u8) anyerror!void {
        const f: *const Fake = @ptrCast(@alignCast(ctx));
        var o: usize = 0;
        for (ranges) |r| for (0..@intCast(r.len)) |k| {
            out[o] = f.byte(r.off + k);
            o += 1;
        };
    }
};

test "rows_carry compare: a file of the fresh bytes is the same; one byte off is found at its image offset" {
    const a = std.testing.allocator;
    var dir_buf: [96]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-rows-carry-compare-{d}", .{platform.c.getpid()}, 0);
    _ = platform.c.mkdir(dir, 0o700);
    defer _ = platform.c.rmdir(dir);
    const ranges = [_]image.Range{ .{ .off = 1000, .len = 300 }, .{ .off = 40, .len = 200 } }; // a ring that wraps
    const name: file.Name = @splat(7);
    var stored: Fake = .{};
    const payload = try a.alloc(u8, 500);
    defer a.free(payload);
    try Fake.gather(&stored, &ranges, payload);
    var w = try file.Writer.open(dir, "b", ".blk");
    try w.append(payload);
    _ = try w.finish(.{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = name, .block = 256 }, &.{}, false);
    var path_buf: [128]u8 = undefined;
    const path = try file.path(&path_buf, dir, "b", ".blk", false);
    defer _ = platform.c.unlink(path);
    const stage = try file.stage(a, 500);
    defer a.free(stage);
    const buf = try a.alloc(u8, 500);
    defer a.free(buf);
    var fresh: Fake = .{};
    const f: Fresh = .{ .ctx = &fresh, .gather = Fake.gather };
    try std.testing.expectEqual(Compare.same, try same(path, .block, name, 256, 0, &ranges, stage, buf, f));
    fresh = .{ .salt = 1, .from = 60, .to = 61 }; // the payload's byte 320, inside the second range
    try std.testing.expectEqual(Compare{ .differs = 60 }, try same(path, .block, name, 256, 0, &ranges, stage, buf, f));
    fresh = .{ .salt = 1, .from = 1299, .to = 1300 }; // the first range's last byte
    try std.testing.expectEqual(Compare{ .differs = 1299 }, try same(path, .block, name, 256, 0, &ranges, stage, buf, f));
    fresh = .{ .salt = 1, .from = 0, .to = 1 << 20 }; // every byte: the first in payload order is the first range's
    try std.testing.expectEqual(Compare{ .differs = 1000 }, try same(path, .block, name, 256, 0, &ranges, stage, buf, f));
    // another name, block, position or length, or no file: not read, so nothing proved either way
    fresh = .{};
    try std.testing.expectEqual(Compare.unreadable, try same(path, .block, @splat(8), 256, 0, &ranges, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(path, .block, name, 128, 0, &ranges, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(path, .block, name, 256, 3, &ranges, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(path, .block, name, 256, 0, ranges[0..1], stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(path, .state, name, 256, 0, &ranges, stage, buf, f));
    var gone_buf: [128]u8 = undefined;
    const gone = try file.path(&gone_buf, dir, "c", ".blk", false);
    try std.testing.expectEqual(Compare.unreadable, try same(gone, .block, name, 256, 0, &ranges, stage, buf, f));
}

test "rows_carry compare: every answer survives the gathered word" {
    for ([_]Compare{ .same, .unreadable, .{ .differs = 0 }, .{ .differs = 1 }, .{ .differs = 123456789 } }) |c| {
        try std.testing.expectEqual(c, answer(word(c)));
    }
}
