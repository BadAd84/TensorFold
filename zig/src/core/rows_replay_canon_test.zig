//! TF_GLM_OVN_DISK_CARRY's replay through a family's canon (rows Image.canon): a state whose live copies sit elsewhere.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const carry = @import("rows_carry.zig");
const replay = @import("rows_replay.zig");
const fake = @import("rows_fake.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const image = rows.image;
const Allocator = std.mem.Allocator;

const gpa = std.testing.allocator;
const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const old_identity: u64 = 7;
const new_identity: u64 = 8;
const layers = 3; // the family's double-buffered layers, two copies of `copy` bytes each
const copy = 16;

/// A family of the Fake's image between a word of copy bits and its layers' two copies (the live one by its bit).
const Paired = struct {
    f: fake.Fake = .{ .gpa = gpa },
    cur: u64 = 0, // the copy bits its next state takes
    noise: u8 = 0, // the stale copies' bytes
    bump: ?u64 = null, // a copies' byte (from their first) its next state holds one more of

    const Kept = struct { inner: pc.Saved, at: u32, cur: u64, noise: u8, bump: ?u64 };

    fn snapshots(p: *Paired) pc.Snapshots {
        return .{ .ptr = p, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &.{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .canon = canon, .canon_head = 8 } } };
    }

    fn of(ptr: *anyopaque) *Paired {
        return @ptrCast(@alignCast(ptr));
    }

    fn inner(p: *Paired) pc.Snapshots {
        return p.f.snapshots();
    }

    /// The Fake's image's bytes at `at`: where the copies start, past the word.
    fn fakeBytes(p: *Paired, at: u32) u64 {
        return p.inner().vtable.bytes(&p.f, at);
    }

    fn bytes(ptr: *anyopaque, at: u32) u64 {
        return 8 + of(ptr).fakeBytes(at) + 2 * layers * copy;
    }

    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const p = of(ptr);
        const k = try gpa.create(Kept);
        errdefer gpa.destroy(k);
        k.* = .{ .inner = try p.inner().vtable.save(&p.f, owner, at), .at = at, .cur = p.cur, .noise = p.noise, .bump = p.bump };
        return k;
    }

    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const p = of(ptr);
        try p.inner().vtable.restore(&p.f, owner, @as(*Kept, @ptrCast(@alignCast(saved))).inner);
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const p = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        p.inner().vtable.drop(&p.f, k.inner);
        gpa.destroy(k);
    }

    fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(image.Part), a: Allocator) anyerror!void {
        _ = ptr;
        try out.append(a, .{ .bytes = 8 });
        try fake.Fake.layout(at, out, a);
        try out.append(a, .{ .bytes = 2 * layers * copy });
    }

    /// Byte `i` of the copies: layer li's held bytes in its live copy, the noise in the other (`bump` one more).
    fn copyByte(k: *const Kept, i: u64) u8 {
        const h = i / (layers * copy);
        const li = i % (layers * copy) / copy;
        const v: u8 = if (h != (k.cur >> @intCast(li)) & 1) k.noise else @truncate(li * 31 + i % copy * 7 + 1);
        return if (k.bump == i) v +% 1 else v;
    }

    fn gather(ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, out: []u8) anyerror!void {
        const p = of(ptr);
        const k: *const Kept = @ptrCast(@alignCast(saved));
        const end = 8 + p.fakeBytes(k.at);
        var o: usize = 0;
        for (ranges) |r| for (r.off..r.off + r.len) |i| {
            out[o] = if (i < 8) std.mem.asBytes(&k.cur)[@intCast(i)] else if (i < end) blk: {
                var b: [1]u8 = undefined;
                try p.inner().vtable.rows.?.gather(&p.f, k.inner, &.{.{ .off = i - 8, .len = 1 }}, &b);
                break :blk b[0];
            } else copyByte(k, i - end);
            o += 1;
        };
    }

    fn blank(_: *anyopaque, _: u32) anyerror!pc.Saved {
        return error.Unused;
    }

    fn scatter(_: *anyopaque, _: pc.Saved, _: []const image.Range, _: []const u8) anyerror!void {
        return error.Unused;
    }

    /// The Fake's bytes in place, then each layer's live copy of the stored state against the fresh state's.
    fn canon(ptr: *anyopaque, at: u32, stored: []const u8, fresh: []const u8, out: *std.ArrayList(image.Match), a: Allocator) anyerror!void {
        const n = of(ptr).fakeBytes(at);
        try out.append(a, .{ .off = 8, .len = n, .fresh = 8 });
        const sc = std.mem.readInt(u64, stored[0..8], .little);
        const fc = std.mem.readInt(u64, fresh[0..8], .little);
        for (0..layers) |li| {
            const sb = (sc >> @intCast(li)) & 1;
            const fb = (fc >> @intCast(li)) & 1;
            try out.append(a, .{ .off = 8 + n + (sb * layers + li) * copy, .len = copy, .fresh = 8 + n + (fb * layers + li) * copy });
        }
    }
};

fn want(identity: u64) carry.record.Record {
    return .{ .dir = carry.dirOf(identity), .key = 0xca55, .proof = 1, .block = rows.block_tokens, .lookahead = 1 };
}

/// The old build's store under `root`: one conversation kept at 1400 with the family's copy bits as `p` sets them.
fn seed(a: Allocator, root: []const u8, p: *Paired) !void {
    const r = try rows.Rows.open(gpa, p.snapshots(), .{ .sync_writes = true, .root = root, .identity = old_identity, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 });
    defer r.deinit();
    var s = pc.Store.init(gpa, p.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const prompt = try a.alloc(u32, 1500);
    for (prompt, 0..) |*t, i| t.* = @intCast(50_000 + i * 7 % 997);
    p.f.prompt = prompt;
    const plan = try s.begin(a, prompt, 1400, &.{}, p.f.starts, null, p.f.spans);
    _ = p.f.pass(&s, plan);
    while (rows.write.step(r)) {}
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try carry.record.write(r.dir, want(old_identity));
}

/// This build's replay of the old directory: its fresh state with the family's copy bits as `p` sets them.
fn run(root: []const u8, p: *Paired) replay.Verdict {
    const E = struct {
        fn fresh(ctx: *anyopaque, tokens: []const u32, at: u32) anyerror!pc.Saved {
            const q: *Paired = @ptrCast(@alignCast(ctx));
            q.f.prompt = tokens;
            q.f.at = at;
            return q.snapshots().vtable.save(q, null, at);
        }
    };
    return replay.run(gpa, p.snapshots(), root, carry.dirOf(old_identity), want(new_identity), replay.default_tokens, .{ .ctx = p, .fresh = E.fresh });
}

fn rootOf(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-rows-canon-{s}-{d}", .{ tag, @import("sys").c.getpid() }) catch unreachable;
}

test "rows_carry replay proves a state whose layers' live copies sit in the other copy; byte for byte it would not" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var buf: [128]u8 = undefined;
    const dir = rootOf(&buf, "flip");
    defer rmTree(dir);
    var old: Paired = .{ .cur = 0b101, .noise = 0x11 };
    try seed(arena.allocator(), dir, &old);
    var new: Paired = .{ .cur = 0b011, .noise = 0x77 }; // layers 1 and 2 flipped, every stale byte other
    try std.testing.expectEqual(replay.Verdict{ .same = 1 }, run(dir, &new));
    try std.testing.expectEqual(@as(usize, 0), new.f.live);
}

test "rows_carry replay refuses a byte that differs in a live copy, and not one that differs in a stale copy" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var buf: [128]u8 = undefined;
    const dir = rootOf(&buf, "live");
    defer rmTree(dir);
    // layer 1's live copy is copy 0 in the stored state (0b101): its byte 3 held one more
    var old: Paired = .{ .cur = 0b101, .noise = 0x11, .bump = 1 * copy + 3 };
    try seed(arena.allocator(), dir, &old);
    var new: Paired = .{ .cur = 0b011, .noise = 0x11 };
    const v = run(dir, &new);
    try std.testing.expect(v == .differs);
    try std.testing.expectEqual(@as(u64, 8 + new.fakeBytes(1400) + 1 * copy + 3), v.differs.off);
    rmTree(dir);
    // the same byte of the stored state's other copy of layer 1 (copy 1): stale, so not compared
    var stale: Paired = .{ .cur = 0b101, .noise = 0x11, .bump = (layers + 1) * copy + 3 };
    try seed(arena.allocator(), dir, &stale);
    try std.testing.expectEqual(replay.Verdict{ .same = 1 }, run(dir, &new));
}
