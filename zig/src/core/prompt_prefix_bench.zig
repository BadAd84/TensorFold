//! Laptop timing of the kept-state walks (TF_GLM_OVN_KEPT_WALK set or not): agent turns over many long kept states.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const prefix = @import("prompt_prefix.zig");
const forks = @import("prompt_cache_fork.zig");

const system = 8192; // ids of the system block every conversation starts with
const step = 2048; // ids a turn adds (a reply and a tool result)

/// A family whose saves cost nothing: the walks are all that is timed.
const Family = struct {
    gpa: std.mem.Allocator,
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        const f: *Family = @ptrCast(@alignCast(ptr));
        return @ptrCast(try f.gpa.create(u8));
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *Family = @ptrCast(@alignCast(ptr));
        f.gpa.destroy(@as(*u8, @ptrCast(saved)));
    }
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

fn output(text: []const u8) !void {
    std.debug.print("{s}", .{text}); // stderr, on every host
}

/// One warm agent turn: the lookup, its restore, then the keeps it planned; the compares it ran.
fn turn(s: *pc.Store, a: std.mem.Allocator, prompt: []const u32) !u64 {
    const before = prefix.full;
    const l = try s.lookupUser(a, prompt, @intCast(prompt.len), @intCast(prompt.len - 64), 0, 0, &.{}, &.{}, &.{});
    if (l.entry) |e| s.resumed(e, prompt, true);
    for (l.marks) |m| _ = s.keepBorn(prompt, @intCast(prompt.len), m, null, &.{}, &.{});
    return prefix.full - before;
}

fn ms(io: std.Io, started: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

/// kept-walk-bench IDS KEPT CONVERSATIONS TURNS: per-turn and per-walk milliseconds and whole compares, one JSON line.
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return error.Arguments;
    const n = try std.fmt.parseInt(usize, args[1], 10);
    const kept = try std.fmt.parseInt(usize, args[2], 10);
    const convs = try std.fmt.parseInt(usize, args[3], 10);
    const turns = try std.fmt.parseInt(usize, args[4], 10);
    const per = kept / convs; // kept turns a conversation
    if (convs == 0 or per == 0 or n < system + (per + 1) * step) return error.Arguments;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const ids = try gpa.alloc([]u32, convs + 1); // the last: a new conversation, the fork walk's miss
    defer gpa.free(ids);
    for (ids, 0..) |*c, i| {
        c.* = try gpa.alloc(u32, n + turns * step);
        for (c.*, 0..) |*t, j| t.* = if (j < system and i > 0) ids[0][j] else r.uintLessThan(u32, 151_000);
    }
    defer for (ids) |c| gpa.free(c);
    var f: Family = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .fork = true }, 1 << 40);
    s.most = kept;
    defer s.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    for (0..per) |j| for (ids[0..convs]) |c| { // warm up: each conversation's earlier turns, up to `n` ids
        _ = try turn(&s, arena.allocator(), c[0 .. n - (per - j) * step]);
    };
    var times = try gpa.alloc(f64, turns * convs);
    defer gpa.free(times);
    var compares: u64 = 0;
    for (0..turns) |j| for (ids[0..convs], 0..) |c, i| {
        _ = arena.reset(.retain_capacity);
        const started = std.Io.Clock.awake.now(io);
        compares += try turn(&s, arena.allocator(), c[0 .. n + j * step]);
        times[j * convs + i] = ms(io, started);
    };
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    const next = ids[0][0 .. n + turns * step];
    const fresh = ids[convs][0..n];
    var walk: [3]f64 = .{ 1e9, 1e9, 1e9 };
    var walk_cmp: [3]u64 = undefined;
    for (0..5) |_| { // the walks alone on the warm store: eviction, the miss's fork, the lookup's index
        inline for (0..3) |w| {
            const before = prefix.full;
            const started = std.Io.Clock.awake.now(io);
            switch (w) {
                0 => std.mem.doNotOptimizeAway(s.victimBut(null, next, @intCast(next.len))),
                1 => std.mem.doNotOptimizeAway(forks.longestShared(pc.Entry, s.entries.items, fresh).len),
                else => std.mem.doNotOptimizeAway(s.find(next, &.{}, &.{})),
            }
            walk[w] = @min(walk[w], ms(io, started));
            walk_cmp[w] = prefix.full - before;
        }
    }
    var line: [512]u8 = undefined;
    try output(try std.fmt.bufPrint(&line, "{{\"gate\":{},\"ids\":{d},\"kept\":{d},\"convs\":{d},\"turn_ms_min\":{d:.3},\"turn_ms_median\":{d:.3},\"turn_compares\":{d:.1},\"victim_ms\":{d:.3},\"victim_compares\":{d},\"fork_ms\":{d:.3},\"fork_compares\":{d},\"find_ms\":{d:.3},\"find_compares\":{d}}}\n", .{ prefix.gated(), n, s.entries.items.len, convs, times[0], times[times.len / 2], @as(f64, @floatFromInt(compares)) / @as(f64, @floatFromInt(times.len)), walk[0], walk_cmp[0], walk[1], walk_cmp[1], walk[2], walk_cmp[2] }));
}
