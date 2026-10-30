//! TF_GLM_OVN_KEPT_WALK: every kept-state walk decides the same with the gate on random stores, comparing fewer lists.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const prefix = @import("prompt_prefix.zig");
const forks = @import("prompt_cache_fork.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const gpa = std.testing.allocator;

test "kept walk: the helper answers as the plain compares, gated or not" {
    defer prefix.fake(false);
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var a: [48]u32 = undefined;
    var b: [48]u32 = undefined;
    for (0..20000) |_| {
        const na = r.uintLessThan(usize, a.len + 1);
        const nb = r.uintLessThan(usize, b.len + 1);
        for (a[0..na]) |*t| t.* = r.uintLessThan(u32, 3); // a small vocabulary: last ids collide often
        for (b[0..nb], 0..) |*t, i| t.* = if (i < na and r.uintLessThan(u8, 8) != 0) a[i] else r.uintLessThan(u32, 3);
        const x = a[0..na];
        const y = b[0..nb];
        const is_prefix = nb <= na and std.mem.eql(u32, x[0..nb], y);
        const shared = std.mem.indexOfDiff(u32, x, y) orelse @min(na, nb);
        for ([_]bool{ false, true }) |on| {
            prefix.fake(on);
            try std.testing.expectEqual(is_prefix, prefix.starts(x, y));
            try std.testing.expectEqual(is_prefix, prefix.ends(x, y));
            try std.testing.expectEqual(shared, prefix.common(x, y));
            var lazy = prefix.Lazy.of(x, y);
            try std.testing.expectEqual(is_prefix, lazy.get());
            const n = r.uintLessThan(usize, 50);
            if (!prefix.beyond(x, y, n)) try std.testing.expect(shared <= n); // a skip never hides a longer share
        }
    }
}

test "kept walk: common finds the first difference past many blocks" {
    defer prefix.fake(false);
    const x = try gpa.alloc(u32, 5000);
    defer gpa.free(x);
    const y = try gpa.alloc(u32, 4100);
    defer gpa.free(y);
    for (x, 0..) |*t, i| t.* = @intCast(i);
    for (y, 0..) |*t, i| t.* = @intCast(i);
    for ([_]bool{ false, true }) |on| {
        prefix.fake(on);
        try std.testing.expectEqual(@as(usize, 4100), prefix.common(x, y));
        for ([_]usize{ 0, 1023, 1024, 1025, 2047, 3071, 4099 }) |at| {
            y[at] += 1;
            try std.testing.expectEqual(at, prefix.common(x, y));
            try std.testing.expectEqual(at, prefix.common(y, x));
            y[at] -= 1;
        }
    }
}

test "kept walk: a long prefix differing at any one id, past and inside the vector folds" {
    defer prefix.fake(false);
    const x = try gpa.alloc(u32, 700);
    defer gpa.free(x);
    const y = try gpa.alloc(u32, 700);
    defer gpa.free(y);
    for (x, y, 0..) |*p, *q, i| {
        p.* = @intCast(i % 5);
        q.* = p.*;
    }
    for ([_]bool{ false, true }) |on| {
        prefix.fake(on);
        for ([_]usize{ 1, 63, 64, 65, 128, 300, 641, 700 }) |n| {
            try std.testing.expect(prefix.starts(x, y[0..n]));
            for (0..n) |at| { // a difference anywhere: before the last id it takes the whole compare to find
                y[at] +%= 1;
                try std.testing.expect(!prefix.starts(x, y[0..n]));
                try std.testing.expectEqual(at, prefix.common(x, y[0..n]));
                y[at] -%= 1;
            }
        }
    }
}

/// Prompts of agent sessions: conversations behind two shared system blocks, chains, re-sends and last-id twins.
const Prompts = struct {
    all: std.ArrayList([]u32) = .empty,
    fn deinit(p: *Prompts) void {
        for (p.all.items) |x| gpa.free(x);
        p.all.deinit(gpa);
    }
    fn pick(p: *Prompts, r: std.Random) []u32 {
        return p.all.items[r.uintLessThan(usize, p.all.items.len)];
    }
    fn next(p: *Prompts, r: std.Random) ![]const u32 {
        const kind = if (p.all.items.len == 0) 0 else r.uintLessThan(u8, 10);
        const out = switch (kind) {
            0 => blk: { // a new conversation behind one of two system blocks that agree on their first ids
                const sys: usize = 4 + r.uintLessThan(usize, 24);
                const k = @as(u32, 1) + @intFromBool(r.boolean());
                const o = try gpa.alloc(u32, sys + 4 + r.uintLessThan(usize, 40));
                for (o[0..sys], 0..) |*t, i| t.* = 10 + @as(u32, @intCast(i)) * k % 7;
                for (o[sys..]) |*t| t.* = r.uintLessThan(u32, 3);
                break :blk o;
            },
            1 => try gpa.dupe(u32, p.pick(r)), // the same prompt again
            2 => blk: { // the same length and last id, another body: a last-id collision
                const o = try gpa.dupe(u32, p.pick(r));
                if (o.len > 1) o[r.uintLessThan(usize, o.len - 1)] +%= 1;
                break :blk o;
            },
            else => blk: { // a later turn: a stored prompt, maybe cut back, then new ids
                const x = p.pick(r);
                const from = x[0 .. x.len - r.uintLessThan(usize, @min(x.len, @as(usize, if (kind < 5) 30 else 2)))];
                const o = try gpa.alloc(u32, from.len + 2 + r.uintLessThan(usize, 60));
                @memcpy(o[0..from.len], from);
                for (o[from.len..]) |*t| t.* = r.uintLessThan(u32, 3);
                break :blk o;
            },
        };
        try p.all.append(gpa, out);
        return out;
    }
};

/// Everything a walk decided, so far: returns, the entries in order with their links, the counts.
const Trace = struct {
    h: std.ArrayList(u64) = .empty,
    fn add(t: *Trace, v: anytype) !void {
        try t.h.append(gpa, switch (@typeInfo(@TypeOf(v))) {
            .optional => if (v) |x| @as(u64, x) + 1 else 0,
            .bool => @intFromBool(v),
            else => @intCast(v),
        });
    }
    fn ids(t: *Trace, x: []const u32) !void {
        try t.add(x.len);
        try t.add(std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(x)));
    }
    fn store(t: *Trace, s: *const pc.Store) !void {
        try t.add(s.entries.items.len);
        for (s.entries.items) |e| {
            for ([_]u64{ e.at, e.born, e.used, e.bytes, @intFromBool(e.shared), @intFromBool(e.branched), @intFromBool(e.expired), e.borrowers }) |v| try t.add(v);
            try t.ids(e.tokens);
            try t.ids(e.last);
            try t.add(e.ext.items.len);
            for (e.ext.items) |o| try t.add(std.mem.indexOfScalar(*pc.Entry, s.entries.items, o));
        }
        for ([_]u64{ s.held, s.clock, s.active_rewind_at, s.counts.hits, s.counts.misses, s.counts.kept, s.counts.evicted, s.counts.failed }) |v| try t.add(v);
    }
};

const Variant = struct { most: usize, fork: bool = false, lost: bool = false };

/// One random session's trace, and the whole-list compares it ran.
fn session(seed: u64, v: Variant, on: bool, t: *Trace) !u64 {
    prefix.fake(on);
    const before = prefix.full;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 4, .fork = v.fork }, 1 << 40);
    s.most = v.most;
    s.links_lost = v.lost; // links complete but unread: `extended` scans, the eviction asks in the other order
    defer s.deinit();
    var prompts: Prompts = .{};
    defer prompts.deinit();
    for (0..300) |_| {
        const prompt = try prompts.next(r);
        switch (r.uintLessThan(u8, 8)) {
            0, 1, 2 => for (0..1 + r.uintLessThan(usize, 3)) |_| { // a pass keeps a few states, some born short
                const at: u32 = 1 + r.uintLessThan(u32, @intCast(prompt.len - 1));
                f.at = at;
                try t.add(s.keepBorn(prompt, if (r.boolean()) @intCast(prompt.len) else at + 1, at, null, &.{}, &.{}));
            },
            3, 4, 5 => { // a lookup with a rewind state, resumed or failed
                const history: u32 = r.uintLessThan(u32, @intCast(prompt.len));
                const l = try s.lookupRewind(arena.allocator(), prompt, @intCast(prompt.len), history, r.uintLessThan(u32, @intCast(prompt.len)), &.{}, &.{}, &.{});
                try t.add(if (l.entry) |e| std.mem.indexOfScalar(*pc.Entry, s.entries.items, e) else null);
                for (l.marks) |m| try t.add(m);
                if (l.entry) |e| s.resumed(e, prompt, r.uintLessThan(u8, 5) != 0);
            },
            6 => if (s.entries.items.len > 0) { // a state the family drops, or a failed peer state forgotten
                const e = s.entries.items[r.uintLessThan(usize, s.entries.items.len)];
                if (r.boolean()) s.expired(e.saved) else s.forget(prompt, e.at, &.{});
                if (r.boolean()) s.collectExpired();
            },
            else => {},
        }
        try probe(&s, arena.allocator(), r, &prompts, t);
        try t.store(&s);
    }
    return prefix.full - before;
}

/// The walks that change nothing, asked on random prompts: eviction, lookup, the fork, marks and wouldServe.
fn probe(s: *pc.Store, a: std.mem.Allocator, r: std.Random, prompts: *Prompts, t: *Trace) !void {
    for (0..3) |_| {
        const prompt = prompts.pick(r);
        const n = s.entries.items.len;
        const skip: ?*pc.Entry = if (n > 0 and r.boolean()) s.entries.items[r.uintLessThan(usize, n)] else null;
        const saved = s.active_rewind_at;
        if (n > 0 and r.boolean()) s.active_rewind_at = s.entries.items[r.uintLessThan(usize, n)].at;
        try t.add(s.victimBut(skip, prompt, r.uintLessThan(u32, @intCast(prompt.len + 2))));
        s.active_rewind_at = saved;
        try t.add(if (s.find(prompt, &.{}, &.{})) |e| std.mem.indexOfScalar(*pc.Entry, s.entries.items, e) else null);
        try t.ids(forks.longestShared(pc.Entry, s.entries.items, prompt));
        const kept = prompts.pick(r);
        try t.add(s.wouldServe(prompt, &.{}, &.{}, kept, &.{}, r.uintLessThan(u32, @intCast(prompt.len + 1))));
        const previous: []const u32 = if (r.boolean()) kept else &.{};
        const m = try s.marks(a, prompt, r.uintLessThan(u32, 4), r.uintLessThan(u32, @intCast(prompt.len + 1)), &.{}, &.{}, previous);
        for (m) |x| try t.add(x);
        try t.add(m.len);
    }
}

test "kept walk: every walk of random sessions decides the same with the gate" {
    defer prefix.fake(false);
    var compared = [2]u64{ 0, 0 };
    var kept: u64 = 0;
    for (0..24) |seed| {
        const v: Variant = .{ .most = if (seed % 3 == 0) 40 else 8, .fork = seed % 2 == 1, .lost = seed % 4 == 2 };
        var off: Trace = .{};
        defer off.h.deinit(gpa);
        var on: Trace = .{};
        defer on.h.deinit(gpa);
        compared[0] += try session(seed, v, false, &off);
        compared[1] += try session(seed, v, true, &on);
        try std.testing.expectEqualSlices(u64, off.h.items, on.h.items);
        kept += off.h.items.len;
    }
    try std.testing.expect(kept > 100_000); // the sessions kept, evicted, looked up and linked: the walks ran
    try std.testing.expect(compared[1] * 3 < compared[0]); // the gate: far fewer whole compares, the same answers
}

test {
    _ = @import("prompt_prefix_count_test.zig");
    _ = prefix;
}

test "kept walk: a state near or at a mark, of other tokens, is never taken as the prompt's" {
    defer prefix.fake(false);
    var p: [200]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(i);
    var q = p;
    q[50] = 7777; // the same length and ids past 50: only a whole compare tells them apart
    for ([_]bool{ false, true }) |on| {
        prefix.fake(on);
        for ([_]*const [200]u32{ &p, &q }) |prompt| { // p's own lookup, then another prompt's over p's states
            var arena: std.heap.ArenaAllocator = .init(gpa);
            defer arena.deinit();
            var f: Fake = .{ .gpa = gpa };
            var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 8 }, 1 << 40);
            defer s.deinit();
            s.most = 2;
            for ([_]u32{ 100, 150 }) |at| {
                f.at = at;
                try std.testing.expect(s.keep(&p, at, null, &.{}, &.{}));
            }
            const l = try s.lookupRewind(arena.allocator(), prompt, prompt.len, 100, 104, &.{}, &.{}, &.{});
            defer if (l.entry) |e| s.resumed(e, prompt, true);
            if (prompt == &p) { // keptNear: the state at 100, under min_gap before the rewind at 103, is the rewind state
                try std.testing.expectEqual(@as(u32, 100), s.active_rewind_at);
                try std.testing.expectEqual(@as(u32, 150), l.entry.?.at);
                continue;
            }
            try std.testing.expect(l.entry == null); // keptNear: p's state at 100 is not q's prefix
            try std.testing.expectEqual(@as(u32, 103), s.active_rewind_at);
            // reserve: q's mark at 100 needs a slot of its own beside p's state there, so both of p's go first
            try std.testing.expectEqualSlices(u32, &.{ 100, 103 }, l.marks);
            try std.testing.expectEqual(@as(u64, 2), s.counts.evicted);
        }
    }
}
