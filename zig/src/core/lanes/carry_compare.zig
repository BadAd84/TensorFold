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

/// File `path` (as `opens` takes it) against a fresh image: its hash, then the bytes `matches` name (null: all).
pub fn same(a: std.mem.Allocator, path: [:0]const u8, kind: file.Kind, name: file.Name, block: u32, at: u32, ranges: []const image.Range, matches: ?[]const image.Match, stage: file.Stage, buf: []u8, fresh: Fresh) !Compare {
    const n: usize = @intCast(image.total(ranges));
    if (buf.len < n) return error.CarryBuffer;
    var o = opens(path, kind, name, block, at, n) orelse return .unreadable;
    defer o.close();
    const got = file.payload(&o, stage) catch |err| return if (err == error.RowsStage) err else .unreadable;
    var pieces: std.ArrayList(Piece) = .empty;
    defer pieces.deinit(a);
    var want: std.ArrayList(image.Range) = .empty;
    defer want.deinit(a);
    try pair(ranges, matches, &pieces, &want, a);
    const m: usize = @intCast(image.total(want.items));
    try fresh.gather(fresh.ctx, want.items, buf[0..m]);
    var done: usize = 0;
    for (pieces.items) |p| {
        if (std.mem.indexOfDiff(u8, got[p.at..][0..p.len], buf[done..][0..p.len])) |i| return .{ .differs = p.off + i };
        done += p.len;
    }
    return .same;
}

/// Stored bytes `len` at payload byte `at` (image offset `off`), compared with the fresh range of the same place.
const Piece = struct { at: usize, off: u64, len: usize };

/// Each part of `ranges` (payload order) a match covers, and its fresh range (null matches: every byte, in place).
fn pair(ranges: []const image.Range, matches: ?[]const image.Match, pieces: *std.ArrayList(Piece), want: *std.ArrayList(image.Range), a: std.mem.Allocator) !void {
    var at: u64 = 0;
    for (ranges) |r| {
        defer at += r.len;
        const ms = matches orelse {
            try pieces.append(a, .{ .at = @intCast(at), .off = r.off, .len = @intCast(r.len) });
            try want.append(a, r);
            continue;
        };
        var i = first(ms, r.off);
        while (i < ms.len and ms[i].off < r.off + r.len) : (i += 1) {
            const lo = @max(r.off, ms[i].off);
            const hi = @min(r.off + r.len, ms[i].off + ms[i].len);
            if (lo >= hi) continue;
            try pieces.append(a, .{ .at = @intCast(at + lo - r.off), .off = lo, .len = @intCast(hi - lo) });
            try want.append(a, .{ .off = ms[i].fresh + lo - ms[i].off, .len = hi - lo });
        }
    }
}

/// The first of `ms` (in `order`) that ends past image offset `off`.
fn first(ms: []const image.Match, off: u64) usize {
    var lo: usize = 0;
    var hi: usize = ms.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (ms[mid].off + ms[mid].len <= off) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// Matches in the order `same` reads them: by stored offset (a family lists them disjoint).
pub fn order(ms: []image.Match) void {
    std.mem.sort(image.Match, ms, {}, struct {
        fn lt(_: void, x: image.Match, y: image.Match) bool {
            return x.off < y.off;
        }
    }.lt);
}

/// Stored image bytes [off, off + out.len) from file `path` (as `opens` takes it) holding `ranges`; false: not there.
pub fn storedBytes(path: [:0]const u8, kind: file.Kind, name: file.Name, block: u32, at: u32, ranges: []const image.Range, off: u64, stage: file.Stage, out: []u8) !bool {
    var done: u64 = 0;
    const at_payload = for (ranges) |r| { // only a file whose ranges hold the bytes is read
        if (off >= r.off and off + out.len <= r.off + r.len) break done + off - r.off;
        done += r.len;
    } else return false;
    var o = opens(path, kind, name, block, at, image.total(ranges)) orelse return false;
    defer o.close();
    const got = file.payload(&o, stage) catch |err| return if (err == error.RowsStage) err else false;
    @memcpy(out, got[@intCast(at_payload)..][0..out.len]);
    return true;
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
    try std.testing.expectEqual(Compare.same, try same(a, path, .block, name, 256, 0, &ranges, null, stage, buf, f));
    fresh = .{ .salt = 1, .from = 60, .to = 61 }; // the payload's byte 320, inside the second range
    try std.testing.expectEqual(Compare{ .differs = 60 }, try same(a, path, .block, name, 256, 0, &ranges, null, stage, buf, f));
    fresh = .{ .salt = 1, .from = 1299, .to = 1300 }; // the first range's last byte
    try std.testing.expectEqual(Compare{ .differs = 1299 }, try same(a, path, .block, name, 256, 0, &ranges, null, stage, buf, f));
    fresh = .{ .salt = 1, .from = 0, .to = 1 << 20 }; // every byte: the first in payload order is the first range's
    try std.testing.expectEqual(Compare{ .differs = 1000 }, try same(a, path, .block, name, 256, 0, &ranges, null, stage, buf, f));
    // another name, block, position or length, or no file: not read, so nothing proved either way
    fresh = .{};
    try std.testing.expectEqual(Compare.unreadable, try same(a, path, .block, @splat(8), 256, 0, &ranges, null, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(a, path, .block, name, 128, 0, &ranges, null, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(a, path, .block, name, 256, 3, &ranges, null, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(a, path, .block, name, 256, 0, ranges[0..1], null, stage, buf, f));
    try std.testing.expectEqual(Compare.unreadable, try same(a, path, .state, name, 256, 0, &ranges, null, stage, buf, f));
    var gone_buf: [128]u8 = undefined;
    const gone = try file.path(&gone_buf, dir, "c", ".blk", false);
    try std.testing.expectEqual(Compare.unreadable, try same(a, gone, .block, name, 256, 0, &ranges, null, stage, buf, f));
    // the stored bytes 1000..1004 read back from the file, wherever its ranges put them in the payload
    var got: [4]u8 = undefined;
    try std.testing.expect(try storedBytes(path, .block, name, 256, 0, &ranges, 1000, stage, &got));
    try std.testing.expectEqual(stored_fake.byte(1000), got[0]);
    try std.testing.expect(!try storedBytes(path, .block, name, 256, 0, &ranges, 238, stage, &got)); // past a range's end
}

const stored_fake: Fake = .{};

/// A fresh image holding a stored image's bytes `shift` later, `salt` added at fresh bytes [from, to).
const Moved = struct {
    shift: u64,
    salt: u8 = 0,
    from: u64 = 0,
    to: u64 = 0,

    fn gather(ctx: *anyopaque, ranges: []const image.Range, out: []u8) anyerror!void {
        const m: *const Moved = @ptrCast(@alignCast(ctx));
        var o: usize = 0;
        for (ranges) |r| for (0..@intCast(r.len)) |k| {
            const j = r.off + k;
            out[o] = stored_fake.byte(j -% m.shift) +% (if (j >= m.from and j < m.to) m.salt else 0);
            o += 1;
        };
    }
};

test "rows_carry compare: matched bytes answer their fresh place, and bytes no match names are not compared" {
    const a = std.testing.allocator;
    var dir_buf: [96]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-rows-carry-match-{d}", .{platform.c.getpid()}, 0);
    _ = platform.c.mkdir(dir, 0o700);
    defer _ = platform.c.rmdir(dir);
    const ranges = [_]image.Range{ .{ .off = 1000, .len = 300 }, .{ .off = 40, .len = 200 } };
    const name: file.Name = @splat(7);
    var stored_img: Fake = .{};
    var payload: [500]u8 = undefined;
    try Fake.gather(&stored_img, &ranges, &payload);
    var w = try file.Writer.open(dir, "m", ".blk");
    try w.append(&payload);
    _ = try w.finish(.{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = name, .block = 256 }, &.{}, false);
    var path_buf: [128]u8 = undefined;
    const path = try file.path(&path_buf, dir, "m", ".blk", false);
    defer _ = platform.c.unlink(path);
    const stage = try file.stage(a, 500);
    defer a.free(stage);
    var buf: [500]u8 = undefined;
    // the fresh image holds the stored bytes 5000 later: 40..140 answered there, 140..240 compared nowhere
    var ms = [_]image.Match{ .{ .off = 1000, .len = 300, .fresh = 6000 }, .{ .off = 40, .len = 100, .fresh = 5040 } };
    order(&ms);
    var moved: Moved = .{ .shift = 5000 };
    const f: Fresh = .{ .ctx = &moved, .gather = Moved.gather };
    try std.testing.expectEqual(Compare.same, try same(a, path, .block, name, 256, 0, &ranges, &ms, stage, &buf, f));
    try std.testing.expect(try same(a, path, .block, name, 256, 0, &ranges, null, stage, &buf, f) != .same); // in place
    moved.salt = 1;
    moved.from = 5100; // the fresh place of stored byte 100
    moved.to = 5101;
    try std.testing.expectEqual(Compare{ .differs = 100 }, try same(a, path, .block, name, 256, 0, &ranges, &ms, stage, &buf, f));
    moved.from = 5150; // stored byte 150's place: no match names it
    moved.to = 5240;
    try std.testing.expectEqual(Compare.same, try same(a, path, .block, name, 256, 0, &ranges, &ms, stage, &buf, f));
}

test "rows_carry compare: every answer survives the gathered word" {
    for ([_]Compare{ .same, .unreadable, .{ .differs = 0 }, .{ .differs = 1 }, .{ .differs = 123456789 } }) |c| {
        try std.testing.expectEqual(c, answer(word(c)));
    }
}
