//! TF_GLM_OVN_KEPT_WALK: the whole-list compares each walk runs, gated and not, on two conversations' kept states.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const prefix = @import("prompt_prefix.zig");
const forks = @import("prompt_cache_fork.zig");
const rows = @import("rows_store.zig");
const select = @import("rows_select.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const RowsFake = @import("rows_fake.zig").Fake;
const t = std.testing;
const gpa = t.allocator;

/// Conversations A and B behind one system block (8 ids); every id past it is its conversation's own.
const Talk = struct {
    a: [400]u32 = undefined,
    b: [400]u32 = undefined,
    fn init(x: *Talk) void {
        for (&x.a, &x.b, 0..) |*p, *q, i| {
            p.* = if (i < 8) @intCast(i) else @intCast(1000 + i);
            q.* = if (i < 8) @intCast(i) else @intCast(5000 + i);
        }
    }
    /// A's turn `k` ends at 40 + 30k ids; B's at 45 + 30k (B's turn 8 outruns A's turn 8).
    fn lenA(k: usize) usize {
        return 40 + 30 * k;
    }
    fn lenB(k: usize) usize {
        return 45 + 30 * k;
    }
};

/// Kept turns 0-7 of A and 0-8 of B, alternating, each state one id before its prompt's end.
fn fill(s: *pc.Store, f: *Fake, x: *const Talk) !void {
    for (0..9) |k| {
        for ([_][]const u32{ x.a[0..Talk.lenA(k)], x.b[0..Talk.lenB(k)] }, 0..) |p, j| {
            if (k == 8 and j == 0) continue;
            f.at = @intCast(p.len - 1);
            try t.expect(s.keep(p, f.at, null, &.{}, &.{}));
        }
    }
}

const Span = @import("cache_modes.zig").Span;
const none: []const u32 = &.{};
const plain: []const Span = &.{};

/// The compares `run` makes with the gate `on`.
fn count(on: bool, run: anytype, args: anytype) u64 {
    prefix.fake(on);
    const before = prefix.full;
    _ = @call(.auto, run, args);
    return prefix.full - before;
}

test "kept walk: each walk's whole compares, gated and not" {
    defer prefix.fake(false);
    var x: Talk = .{};
    x.init();
    var cells: [2][9]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |on, m| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 40);
        defer s.deinit();
        prefix.fake(false);
        try fill(&s, &f, &x);
        const next = x.a[0..Talk.lenA(9)];
        f.at = @intCast(Talk.lenA(8) - 1); // link: A's turn 8 against the seventeen kept (B's turn 8 longer than it)
        cells[m][0] = count(on, pc.Store.keep, .{ &s, x.a[0..Talk.lenA(8)], f.at, @as(?*anyopaque, null), none, plain });
        cells[m][1] = count(on, pc.Store.victimBut, .{ &s, null, next, @as(u32, @intCast(next.len)) }); // eviction
        s.active_rewind_at = @intCast(Talk.lenA(3) - 1); // eviction with a rewind state: only A's turn 3 is asked
        cells[m][2] = count(on, pc.Store.victimBut, .{ &s, null, next, @as(u32, @intCast(next.len)) });
        s.active_rewind_at = 0;
        cells[m][3] = count(on, pc.Store.find, .{ &s, next, none, plain });
        cells[m][4] = count(on, forks.longestShared, .{ pc.Entry, s.entries.items, next }); // a miss's fork
        cells[m][5] = count(on, pc.Store.wouldServe, .{ &s, next, none, plain, @as([]const u32, x.b[0..]), plain, @as(u32, 100) });
        cells[m][6] = count(on, pc.Store.forget, .{ &s, x.a[0..Talk.lenB(2)], @as(u32, @intCast(Talk.lenB(2) - 1)), plain });
        s.links_lost = true; // eviction asking `extendedScan`: off, a superseded state's prefix first, as always
        cells[m][7] = count(on, pc.Store.victimBut, .{ &s, null, next, @as(u32, @intCast(next.len)) });
        s.links_lost = false;
        f.at = 289; // a state born as a short request: link asks nothing of the earlier-born, nor does the proof
        cells[m][8] = count(on, pc.Store.keepBorn, .{ &s, x.a[0..290], @as(u32, 30), f.at, @as(?*anyopaque, null), none, plain });
    }
    // off: as the walks always compared; on: one compare a conversation where a link proves the rest, none for B
    try t.expectEqualSlices(u64, &.{ 17, 18, 18, 1, 18, 1, 1, 18, 0 }, &cells[0]);
    try t.expectEqualSlices(u64, &.{ 1, 0, 1, 1, 9, 0, 0, 1, 0 }, &cells[1]);
}

/// A disk tier beside a RAM store, as rows_select_test's rig: states are any token lists, the prompt is A's.
const Disk = struct {
    family: RowsFake = .{ .gpa = gpa },
    disk: rows.Rows = undefined,
    store: pc.Store = undefined,
    fn init(d: *Disk, prompt: []const u32) void {
        d.family.prompt = prompt;
        const f = d.family.snapshots();
        d.disk = .{ .gpa = gpa, .family = f, .img = f.vtable.rows.?, .sync_writes = true, .root = undefined, .dir = undefined, .keying = .{ .identity = 17, .block = rows.block_tokens, .lookahead = 1 }, .cap = 1 << 30, .daily = 1 << 30, .read_rate = 1000, .prefill_rate = 100 };
        d.store = pc.Store.init(gpa, f, .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 30);
        d.store.disk = &d.disk;
    }
    fn deinit(d: *Disk) void {
        d.store.disk = null;
        d.store.deinit();
        for (d.disk.states.items) |state| gpa.free(state.tokens);
        d.disk.states.deinit(gpa);
    }
    fn add(d: *Disk, tokens: []const u32, cost: u64) !void {
        const own = try gpa.dupe(u32, tokens);
        errdefer gpa.free(own);
        try d.disk.states.append(gpa, .{ .name = @splat(0), .at = @intCast(tokens.len - 1), .tokens = own, .spans = &.{}, .starts = &.{}, .own = cost, .rows = 0, .used = 0, .shared = false });
    }
    /// The RAM state at `at` of the prompt, kept with the disk tier out of the way.
    fn hold(d: *Disk, prompt: []const u32, at: u32) !*pc.Entry {
        d.store.disk = null;
        defer d.store.disk = &d.disk;
        d.family.at = at;
        try t.expect(d.store.keep(prompt, at, null, &.{}, &.{}));
        return d.store.entries.items[d.store.entries.items.len - 1];
    }
};

test "kept walk: a disk restore's choice is the same with the gate, from fewer compares" {
    defer prefix.fake(false);
    var x: Talk = .{};
    x.init();
    const prompt = x.a[0..Talk.lenA(9)];
    var cells: [2]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |on, m| {
        var d: Disk = .{};
        d.init(prompt);
        defer d.deinit();
        var k: usize = 8;
        while (k > 0) : (k -= 1) try d.add(x.a[0..Talk.lenA(k)], 10); // A's turns longest first, then B's
        for (0..8) |j| try d.add(x.b[0..Talk.lenB(j)], 10);
        const have = try d.hold(prompt, @intCast(Talk.lenA(5) - 1)); // RAM resumes at A's turn 5
        prefix.fake(on);
        const before = prefix.full;
        const c = select.choose(&d.store, prompt, &.{}, &.{}, have);
        cells[m] = prefix.full - before;
        try t.expectEqual(@as(?usize, 0), c.restore); // A's turn 8 saves the most prefill
        try t.expectEqual(@as(?usize, 0), c.disk);
    }
    // off: every state; on: A's turns past the RAM state (8, 7, 6), the rest moot or a last-id miss
    try t.expectEqual([2]u64{ 16, 3 }, cells);
}

test "kept walk: random disk states choose the same restore with the gate" {
    defer prefix.fake(false);
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    var prompt: [300]u32 = undefined;
    for (0..300) |round| {
        for (&prompt) |*p| p.* = r.uintLessThan(u32, 3);
        var got: [2]select.Choice = undefined;
        for ([_]bool{ false, true }, 0..) |on, m| {
            var rr = std.Random.DefaultPrng.init(round);
            const q = rr.random();
            var d: Disk = .{};
            d.init(&prompt);
            defer d.deinit();
            for (0..q.uintLessThan(usize, 24)) |_| { // a prefix, or a prefix with one id changed (last ids collide)
                var tokens: [300]u32 = prompt;
                const n = 2 + q.uintLessThan(usize, 290);
                if (q.boolean()) tokens[q.uintLessThan(usize, n)] +%= 1;
                try d.add(tokens[0..n], 1 + q.uintLessThan(u64, 4000));
            }
            const have = if (q.boolean()) try d.hold(&prompt, 1 + q.uintLessThan(u32, 250)) else null;
            prefix.fake(on);
            got[m] = select.choose(&d.store, &prompt, &.{}, &.{}, have);
        }
        try t.expectEqual(got[0].disk, got[1].disk);
        try t.expectEqual(got[0].restore, got[1].restore);
        try t.expectEqual(got[0].read_seconds, got[1].read_seconds);
    }
}
