//! TF_GLM_OVN_DISK_READERS=N: a state's files read by N threads at once into staging slots, each checked by its reader;
//! the calling thread copies each slot out as it fills, so reads and copies overlap and no part sits whole in RAM.
const std = @import("std");
const platform = @import("sys");
const file = @import("rows_file.zig");
const Stage = file.Stage;
const Allocator = std.mem.Allocator;

/// Production's spill tier (recipe 0128): eight readers keep eight reads in flight on one NVMe.
pub const local_default: u32 = 8;

/// Most readers one restore runs.
pub const most: u32 = 64;

/// TF_GLM_OVN_DISK_READERS: readers a restore uses; unset, empty or 0 is `fallback`, an unreadable value is refused.
pub fn knob(fallback: u32) !u32 {
    const v = platform.c.getenv("TF_GLM_OVN_DISK_READERS") orelse return fallback;
    return fromText(std.mem.span(v), fallback);
}

fn fromText(s: []const u8, fallback: u32) !u32 {
    if (s.len == 0) return fallback;
    const n = std.fmt.parseInt(u32, s, 10) catch return error.BadOvnKnob;
    return if (n == 0) fallback else @min(n, most);
}

/// A family's pinning of staging for device copies (CUDA: cuMemHostRegister); without one staging stays pageable.
pub const Pin = struct {
    ctx: *anyopaque,
    pin: *const fn (ctx: *anyopaque, buf: []u8) bool,
    unpin: *const fn (ctx: *anyopaque, buf: []u8) void,
};

/// Reusable staging: `count` slots of `slot` bytes and a `big` buffer for any wider file; grown, never shrunk.
pub const Staging = struct {
    gpa: Allocator = std.heap.page_allocator, // whole pages for its buffers (and the pool's few words a restore)
    pin: ?Pin = null,
    slots: Stage = &.{},
    slot: usize = 0,
    count: usize = 0,
    big: Stage = &.{},
    pinned: [2]bool = .{ false, false }, // slots, big

    /// At least `count` slots of `slot` bytes and a big buffer of `big` bytes (each rounded to the read alignment).
    pub fn ensure(s: *Staging, slot: u64, count: usize, big: u64) !void {
        const want: usize = @intCast(std.mem.alignForward(u64, @max(slot, 1), file.alignment));
        if (want > s.slot or count > s.count) {
            const n = @max(count, s.count);
            const w = @max(want, s.slot);
            s.drop(0);
            s.slots = try s.make(w * n, 0);
            s.slot = w;
            s.count = n;
        }
        if (big > s.big.len) {
            s.drop(1);
            s.big = try s.make(big, 1);
        }
    }

    fn make(s: *Staging, bytes: u64, which: usize) !Stage {
        const b = try file.stage(s.gpa, bytes);
        if (s.pin) |p| s.pinned[which] = p.pin(p.ctx, b);
        return b;
    }

    fn drop(s: *Staging, which: usize) void {
        const b = if (which == 0) &s.slots else &s.big;
        if (b.len == 0) return;
        if (s.pinned[which]) if (s.pin) |p| p.unpin(p.ctx, b.*);
        s.pinned[which] = false;
        s.gpa.free(b.*);
        b.* = &.{};
        if (which == 0) {
            s.slot = 0;
            s.count = 0;
        }
    }

    pub fn deinit(s: *Staging) void {
        s.drop(0);
        s.drop(1);
    }
};

const free: u8 = 0;
const busy: u8 = 1;
const ready: u8 = 2;

fn doze() void {
    var t: platform.c.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_us };
    _ = platform.c.nanosleep(&t, null);
}

fn Pool(comptime C: type) type {
    return struct {
        const P = @This();
        ctx: C,
        n: usize,
        st: *const Staging,
        next: std.atomic.Value(usize) = .init(0),
        stop: std.atomic.Value(bool) = .init(false),
        lock: std.atomic.Mutex = .unlocked,
        err: ?anyerror = null,
        bad: ?usize = null,
        state: []std.atomic.Value(u8), // each slot's, then the big buffer's
        item: []usize,
        got: [][]const u8,

        fn buf(p: *P, s: usize) Stage {
            if (s == p.st.count) return p.st.big;
            return @alignCast(p.st.slots[s * p.st.slot ..][0..p.st.slot]);
        }

        /// The first failure wins: its error and (a read's) item; every reader stops claiming.
        fn fail(p: *P, i: ?usize, e: anyerror) void {
            while (!p.lock.tryLock()) std.Thread.yield() catch {};
            if (p.err == null) {
                p.err = e;
                p.bad = i;
            }
            p.lock.unlock();
            p.stop.store(true, .release);
        }

        fn acquire(p: *P, big: bool) ?usize {
            while (!p.stop.load(.acquire)) {
                if (big) {
                    if (p.state[p.st.count].cmpxchgStrong(free, busy, .acquire, .monotonic) == null) return p.st.count;
                } else for (0..p.st.count) |s| {
                    if (p.state[s].cmpxchgStrong(free, busy, .acquire, .monotonic) == null) return s;
                }
                doze();
            }
            return null;
        }

        fn reader(p: *P) void {
            while (!p.stop.load(.acquire)) {
                const i = p.next.fetchAdd(1, .acq_rel);
                if (i >= p.n) return;
                const span = p.ctx.span(i);
                const big = span > p.st.slot;
                if (big and span > p.st.big.len) return p.fail(i, error.RowsStage);
                const s = p.acquire(big) orelse return;
                const got = p.ctx.read(i, p.buf(s)) catch |e| {
                    p.state[s].store(free, .release);
                    return p.fail(i, e);
                };
                p.item[s] = i;
                p.got[s] = got;
                p.state[s].store(ready, .release);
            }
        }

        /// The calling thread: each ready slot consumed and freed (`last` held back until it is the only one left).
        fn consume(p: *P, last: ?usize) void {
            var done: usize = 0;
            while (done < p.n and !p.stop.load(.acquire)) {
                var moved = false;
                for (0..p.st.count + 1) |s| {
                    if (p.state[s].load(.acquire) != ready) continue;
                    const i = p.item[s];
                    if (last != null and last.? == i and done + 1 != p.n) continue;
                    p.ctx.consume(i, p.got[s]) catch |e| return p.fail(null, e);
                    p.state[s].store(free, .release);
                    done += 1;
                    moved = true;
                }
                if (!moved) doze();
            }
        }
    };
}

/// Items `0..n` read by `threads` reader threads into `st`'s slots (an item wider than a slot into its big buffer), and
/// each given to `ctx.consume(i, payload)` on the calling thread as it is ready; `last` (when given) is consumed after
/// every other. `ctx` gives `span(i)` (an item's aligned bytes) and `read(i, into)` (its checked payload, a prefix of
/// `into`). The first error stops new reads; `failed` names the item whose read failed (null: a consume or the pool).
pub fn pipe(threads: u32, n: usize, last: ?usize, st: *const Staging, ctx: anytype, failed: *?usize) !void {
    failed.* = null;
    if (n == 0) return;
    if (st.count < 2) return error.RowsStage; // `last` may hold one slot while the others keep the readers going
    const P = Pool(@TypeOf(ctx));
    const gpa = st.gpa;
    const state = try gpa.alloc(std.atomic.Value(u8), st.count + 1);
    defer gpa.free(state);
    for (state) |*x| x.* = .init(free);
    const item = try gpa.alloc(usize, st.count + 1);
    defer gpa.free(item);
    const got = try gpa.alloc([]const u8, st.count + 1);
    defer gpa.free(got);
    var p: P = .{ .ctx = ctx, .n = n, .st = st, .state = state, .item = item, .got = got };
    var ts: [most]?std.Thread = @splat(null);
    const t = @min(@max(threads, 1), most, n);
    var started: usize = 0;
    for (ts[0..t]) |*x| {
        x.* = std.Thread.spawn(.{}, P.reader, .{&p}) catch null;
        if (x.* != null) started += 1;
    }
    if (started == 0) return error.ThreadQuotaExceeded;
    p.consume(last);
    p.stop.store(true, .release); // readers waiting for a slot leave
    for (ts[0..t]) |x| if (x) |th| th.join();
    failed.* = p.bad;
    if (p.err) |e| return e;
}

/// `work(ctx, i)` for every `i` in 0..n on `threads` threads, the caller one of them; the first error stops new items.
pub fn each(threads: u32, n: usize, ctx: anytype, comptime work: fn (@TypeOf(ctx), usize) anyerror!void) !void {
    const E = struct {
        ctx: @TypeOf(ctx),
        n: usize,
        next: std.atomic.Value(usize) = .init(0),
        stop: std.atomic.Value(bool) = .init(false),
        lock: std.atomic.Mutex = .unlocked,
        err: ?anyerror = null,

        fn run(x: *@This()) void {
            while (!x.stop.load(.acquire)) {
                const i = x.next.fetchAdd(1, .acq_rel);
                if (i >= x.n) return;
                work(x.ctx, i) catch |e| {
                    while (!x.lock.tryLock()) std.Thread.yield() catch {};
                    if (x.err == null) x.err = e;
                    x.lock.unlock();
                    x.stop.store(true, .release);
                    return;
                };
            }
        }
    };
    var x: E = .{ .ctx = ctx, .n = n };
    var ts: [most]?std.Thread = @splat(null);
    const t = @min(@max(threads, 1), most, @max(n, 1)) - 1;
    for (ts[0..t]) |*th| th.* = std.Thread.spawn(.{}, E.run, .{&x}) catch null;
    E.run(&x);
    for (ts[0..t]) |th| if (th) |h| h.join();
    if (x.err) |e| return e;
}

const Probe = struct {
    inflight: std.atomic.Value(u32) = .init(0),
    peak: std.atomic.Value(u32) = .init(0),
    seen: [24]std.atomic.Value(u32) = @splat(.init(0)),
    order: [24]usize = undefined,
    consumed: usize = 0,
    fail_at: ?usize = null,

    fn enter(p: *Probe) void {
        const now = p.inflight.fetchAdd(1, .acq_rel) + 1;
        var top = p.peak.load(.acquire);
        while (now > top) top = p.peak.cmpxchgWeak(top, now, .acq_rel, .acquire) orelse break;
        var t: platform.c.timespec = .{ .sec = 0, .nsec = 3 * std.time.ns_per_ms };
        _ = platform.c.nanosleep(&t, null);
        _ = p.inflight.fetchSub(1, .acq_rel);
    }

    pub fn span(_: *Probe, i: usize) u64 {
        return if (i == 0) 3 * file.alignment else 64; // item 0 is wider than a slot: the big buffer takes it
    }

    pub fn read(p: *Probe, i: usize, into: Stage) ![]const u8 {
        p.enter();
        if (p.fail_at == i) return error.RowsHash;
        std.mem.writeInt(u64, into[0..8], i *% 7919 +% 3, .little);
        return into[0..8];
    }

    pub fn consume(p: *Probe, i: usize, payload: []const u8) !void {
        if (std.mem.readInt(u64, payload[0..8], .little) != i *% 7919 +% 3) return error.WrongPayload;
        _ = p.seen[i].fetchAdd(1, .acq_rel);
        p.order[p.consumed] = i;
        p.consumed += 1;
    }

    fn work(p: *Probe, i: usize) !void {
        p.enter();
        _ = p.seen[i].fetchAdd(1, .acq_rel);
    }
};

test "disk_readers pipe: every file once, several reads in flight, the held-back file consumed last" {
    var st: Staging = .{ .gpa = std.testing.allocator };
    defer st.deinit();
    try st.ensure(64, 8, 3 * file.alignment);
    var p: Probe = .{};
    var failed: ?usize = null;
    // item 0 is read first and is ready first: only the hold-back keeps it for last
    try pipe(4, p.seen.len, 0, &st, &p, &failed);
    try std.testing.expectEqual(@as(?usize, null), failed);
    for (&p.seen) |*s| try std.testing.expectEqual(@as(u32, 1), s.load(.acquire));
    try std.testing.expectEqual(p.seen.len, p.consumed);
    try std.testing.expectEqual(@as(usize, 0), p.order[p.seen.len - 1]);
    try std.testing.expect(p.peak.load(.acquire) >= 2); // one reader would never have two reads in flight
}

test "disk_readers pipe: a read that fails stops the pool and names its file" {
    var st: Staging = .{ .gpa = std.testing.allocator };
    defer st.deinit();
    try st.ensure(64, 4, 3 * file.alignment);
    var p: Probe = .{ .fail_at = 5 };
    var failed: ?usize = null;
    try std.testing.expectError(error.RowsHash, pipe(3, p.seen.len, null, &st, &p, &failed));
    try std.testing.expectEqual(@as(?usize, 5), failed);
    try std.testing.expectEqual(@as(u32, 0), p.seen[5].load(.acquire));
}

test "disk_readers each: every item once on several threads; the knob's text" {
    var p: Probe = .{};
    try each(4, p.seen.len, &p, Probe.work);
    for (&p.seen) |*s| try std.testing.expectEqual(@as(u32, 1), s.load(.acquire));
    try std.testing.expect(p.peak.load(.acquire) >= 2);
    try std.testing.expectEqual(@as(u32, 8), try fromText("", 8));
    try std.testing.expectEqual(@as(u32, 1), try fromText("0", 1));
    try std.testing.expectEqual(@as(u32, 4), try fromText("4", 1));
    try std.testing.expectEqual(most, try fromText("1000", 1));
    try std.testing.expectError(error.BadOvnKnob, fromText("x", 1));
}
