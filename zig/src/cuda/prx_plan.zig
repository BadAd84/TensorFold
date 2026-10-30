//! TF_GLM_OVN_PRX's plan: the pinned region's layout, an exchange cut into calls, the GPU's side of each call.

const std = @import("std");

pub const max_world = 4;
pub const max_peers = max_world - 1;
pub const max_stripes = 4;
pub const max_slots = 8;
pub const line = 64; // every word the NICs or the GPU write sits on its own line
pub const qcap = 8192; // calls the host may post ahead of the proxy

const ctrl_ready = 0; // [slot][link]: the payload for that link staged
const ctrl_consumed = max_slots * max_peers; // [slot]: the GPU read the call's receive blocks
const ctrl_send_done = ctrl_consumed + max_slots; // [slot]: the NICs read the call's send blocks
const ctrl_credit = ctrl_send_done + max_slots; // [rank]: that rank's freed calls
const ctrl_lines = ctrl_credit + max_world;

pub const Layout = struct {
    world: usize,
    slots: usize,
    M: usize, // a block's bytes: the most one call moves to or from one peer
    flags: usize,
    data: usize, // the first data byte: [0, data) is ctrl and flags (never relaxed-ordered)
    slot_bytes: usize,
    total: usize,

    pub fn of(world: usize, slots: usize, M: usize) error{BadGeometry}!Layout {
        if (world < 2 or world > max_world or slots < 2 or slots > max_slots) return error.BadGeometry;
        if (M == 0 or M % 4096 != 0 or M > 1 << 31) return error.BadGeometry;
        const ctrl = std.mem.alignForward(usize, ctrl_lines * line, 4096);
        const flags = std.mem.alignForward(usize, slots * max_world * max_stripes * line, 4096);
        const slot_bytes = (2 * world - 1) * M;
        return .{ .world = world, .slots = slots, .M = M, .flags = ctrl, .data = ctrl + flags, .slot_bytes = slot_bytes, .total = ctrl + flags + slots * slot_bytes };
    }

    pub fn ready(l: *const Layout, s: usize, link: usize) usize {
        _ = l;
        return (ctrl_ready + s * max_peers + link) * line;
    }
    pub fn consumed(l: *const Layout, s: usize) usize {
        _ = l;
        return (ctrl_consumed + s) * line;
    }
    pub fn sendDone(l: *const Layout, s: usize) usize {
        _ = l;
        return (ctrl_send_done + s) * line;
    }
    pub fn credit(l: *const Layout, r: usize) usize {
        _ = l;
        return (ctrl_credit + r) * line;
    }
    pub fn flag(l: *const Layout, s: usize, src: usize, stripe: usize) usize {
        return l.flags + ((s * max_world + src) * max_stripes + stripe) * line;
    }
    pub fn send(l: *const Layout, s: usize, link: usize) usize {
        return l.data + s * l.slot_bytes + link * l.M;
    }
    pub fn recv(l: *const Layout, s: usize, src: usize) usize {
        return l.data + s * l.slot_bytes + (l.world - 1 + src) * l.M;
    }
};

/// Link L of rank `rank`: the other ranks in ascending order.
pub fn linkPeer(rank: usize, link: usize) usize {
    return if (link < rank) link else link + 1;
}

/// CU_STREAM_WAIT_VALUE_GEQ's compare: cyclic, (int32_t)(word - value) >= 0.
pub fn geq(word: u32, value: u32) bool {
    return @as(i32, @bitCast(word -% value)) >= 0;
}

/// A posted call: the bytes for each link, and whether one payload (block 0) goes to every link.
pub const Desc = struct { bytes: [max_peers]u32 = @splat(0), shared: bool = false };

/// One peer's side of an exchange: `out` bytes from `src` to it, `in` bytes from it into `dst` (device addresses).
pub const Part = struct { src: u64 = 0, out: usize = 0, dst: u64 = 0, in: usize = 0 };

/// What NCCL's grouped send/receive with every other rank would move, as one exchange.
pub const Exchange = struct {
    parts: [max_world]Part = @splat(.{}), // by rank; this rank's own is unused
    shared: bool = false, // every peer's `src` and `out` alike (an all-gather's one payload): staged once
    unit: usize, // the most bytes any rank sends any peer: every rank cuts the same calls from it

    /// Whether the parts keep the exchange's rules (every size within `unit`; a shared payload alike for every peer).
    pub fn valid(x: *const Exchange, rank: usize, world: usize) bool {
        if (world < 2 or world > max_world or rank >= world) return false;
        const first = x.parts[linkPeer(rank, 0)];
        for (0..world - 1) |link| {
            const p = x.parts[linkPeer(rank, link)];
            if (p.out > x.unit or p.in > x.unit) return false;
            if (x.shared and (p.src != first.src or p.out != first.out)) return false;
        }
        return true;
    }
};

/// The calls an exchange takes: one per `M` bytes of `unit`, at least one (so empty exchanges still meet).
pub fn callCount(unit: usize, M: usize) usize {
    return @max(1, std.math.divCeil(usize, unit, M) catch unreachable);
}

/// Bytes [lo, lo + M) of `n`: what one call moves of a part.
pub fn piece(n: usize, lo: usize, M: usize) usize {
    return if (n > lo) @min(n - lo, M) else 0;
}

/// The GPU's side of `x`: per call (posted on `ch`) on `ops`: slot wait, staging, ready, flags, copies out, consumed.
pub fn run(ch: anytype, ops: anytype, l: *const Layout, rank: usize, stripes: usize, x: *const Exchange) !void {
    if (!x.valid(rank, l.world) or stripes == 0 or stripes > max_stripes) return error.BadExchange;
    const peers = l.world - 1;
    for (0..callCount(x.unit, l.M)) |k| {
        const lo = k * l.M;
        var d: Desc = .{ .shared = x.shared };
        for (0..peers) |link| d.bytes[link] = @intCast(piece(x.parts[linkPeer(rank, link)].out, lo, l.M));
        const c: u64 = try ch.post(d);
        const s: usize = @intCast(c % l.slots);
        const v: u32 = @truncate(c + 1);
        if (c >= l.slots) try ops.wait(l.sendDone(s), @as(u32, @truncate(c - l.slots + 1))); // the NICs read it out
        for (0..if (x.shared) 1 else peers) |link| {
            if (d.bytes[link] > 0) try ops.stage(l.send(s, link), x.parts[linkPeer(rank, link)].src + lo, d.bytes[link]);
            try ops.write(l.ready(s, link), v);
        }
        for (0..peers) |link| {
            const r = linkPeer(rank, link);
            for (0..stripes) |i| try ops.wait(l.flag(s, r, i), v);
            const n = piece(x.parts[r].in, lo, l.M);
            if (n > 0) try ops.unstage(x.parts[r].dst + lo, l.recv(s, r), n);
        }
        try ops.write(l.consumed(s), v);
    }
}

// -- tests ------------------------------------------------------------------------------------------------------------

test "PRX plan: no two words, flags or blocks of the region overlap, at every geometry" {
    for (2..max_world + 1) |w| for (2..max_slots + 1) |slots| {
        const l = try Layout.of(w, slots, 8192);
        var seen = std.AutoHashMap(usize, void).init(std.testing.allocator);
        defer seen.deinit();
        for (0..slots) |s| {
            for (0..max_peers) |k| try std.testing.expect((try seen.fetchPut(l.ready(s, k), {})) == null);
            try std.testing.expect((try seen.fetchPut(l.consumed(s), {})) == null);
            try std.testing.expect((try seen.fetchPut(l.sendDone(s), {})) == null);
            for (0..max_world) |r| for (0..max_stripes) |i| {
                const f = l.flag(s, r, i);
                try std.testing.expect(f >= l.flags and f + 4 <= l.data);
                try std.testing.expect((try seen.fetchPut(f, {})) == null);
            };
        }
        for (0..max_world) |r| try std.testing.expect((try seen.fetchPut(l.credit(r), {})) == null);
        var it = seen.keyIterator();
        while (it.next()) |k| try std.testing.expect(k.* % line == 0 and k.* < l.data);
        // the data blocks tile each slot: world - 1 send blocks then world receive blocks, M apart, inside the region
        for (0..slots) |s| {
            for (0..w - 1) |k| try std.testing.expectEqual(l.data + s * l.slot_bytes + k * l.M, l.send(s, k));
            for (0..w) |r| try std.testing.expectEqual(l.send(s, 0) + (w - 1 + r) * l.M, l.recv(s, r));
            try std.testing.expect(l.recv(s, w - 1) + l.M <= l.total);
        }
        try std.testing.expectEqual(l.data + slots * (2 * w - 1) * l.M, l.total);
    };
    try std.testing.expectError(error.BadGeometry, Layout.of(5, 4, 4096));
    try std.testing.expectError(error.BadGeometry, Layout.of(3, 1, 4096));
    try std.testing.expectError(error.BadGeometry, Layout.of(3, 4, 4000));
}

test "PRX plan: links name the other ranks once, and the cyclic compare matches CU_STREAM_WAIT_VALUE_GEQ" {
    for (2..max_world + 1) |w| for (0..w) |r| {
        var seen: [max_world]bool = @splat(false);
        for (0..w - 1) |k| {
            const p = linkPeer(r, k);
            try std.testing.expect(p != r and p < w and !seen[p]);
            seen[p] = true;
        }
    };
    try std.testing.expect(geq(5, 5) and geq(6, 5) and !geq(4, 5));
    try std.testing.expect(geq(1, 0xffff_fffe)); // past the wrap
    try std.testing.expect(!geq(0xffff_fffe, 1));
}

/// Records what `run` posts and enqueues, one line a step.
const Tape = struct {
    const Op = union(enum) { post: Desc, wait: [2]u64, write: [2]u64, stage: [3]u64, unstage: [3]u64 };
    ops: [256]Op = undefined,
    n: usize = 0,
    calls: u64 = 0,

    fn add(t: *Tape, o: Op) void {
        t.ops[t.n] = o;
        t.n += 1;
    }
    fn post(t: *Tape, d: Desc) !u64 {
        t.add(.{ .post = d });
        t.calls += 1;
        return t.calls - 1;
    }
    fn wait(t: *Tape, at: usize, v: u32) !void {
        t.add(.{ .wait = .{ at, v } });
    }
    fn write(t: *Tape, at: usize, v: u32) !void {
        t.add(.{ .write = .{ at, v } });
    }
    fn stage(t: *Tape, at: usize, src: u64, n: usize) !void {
        t.add(.{ .stage = .{ at, src, n } });
    }
    fn unstage(t: *Tape, dst: u64, at: usize, n: usize) !void {
        t.add(.{ .unstage = .{ dst, at, n } });
    }
};

test "PRX plan: a call's GPU side waits for its slot, stages then readies each link, waits every stripe, frees last" {
    const l = try Layout.of(3, 2, 4096);
    var t: Tape = .{};
    t.calls = 2; // the third call: slot 0's second use waits for the first's sends
    var x: Exchange = .{ .unit = 3000 };
    x.parts[0] = .{ .src = 0x1000, .out = 3000, .dst = 0x9000, .in = 2000 };
    x.parts[2] = .{ .src = 0x2000, .out = 1000, .dst = 0xa000, .in = 3000 };
    try run(&t, &t, &l, 1, 2, &x);
    const O = Tape.Op;
    const want = [_]O{
        .{ .post = .{ .bytes = .{ 3000, 1000, 0 } } },
        .{ .wait = .{ l.sendDone(0), 1 } },
        .{ .stage = .{ l.send(0, 0), 0x1000, 3000 } },
        .{ .write = .{ l.ready(0, 0), 3 } },
        .{ .stage = .{ l.send(0, 1), 0x2000, 1000 } },
        .{ .write = .{ l.ready(0, 1), 3 } },
        .{ .wait = .{ l.flag(0, 0, 0), 3 } },
        .{ .wait = .{ l.flag(0, 0, 1), 3 } },
        .{ .unstage = .{ 0x9000, l.recv(0, 0), 2000 } },
        .{ .wait = .{ l.flag(0, 2, 0), 3 } },
        .{ .wait = .{ l.flag(0, 2, 1), 3 } },
        .{ .unstage = .{ 0xa000, l.recv(0, 2), 3000 } },
        .{ .write = .{ l.consumed(0), 3 } },
    };
    try std.testing.expectEqual(want.len, t.n);
    for (want, t.ops[0..t.n]) |a, b| try std.testing.expectEqualDeep(a, b);
}

test "PRX plan: parts past a block go as several calls on every rank alike, a shared payload staged once a call" {
    const l = try Layout.of(3, 4, 4096);
    var t: Tape = .{};
    // rank 0 sends 5000 to rank 1 and 9000 to rank 2; the unit (9000: some rank's most) cuts three calls
    var x: Exchange = .{ .unit = 9000 };
    x.parts[1] = .{ .src = 0x10000, .out = 5000, .dst = 0x40000, .in = 9000 };
    x.parts[2] = .{ .src = 0x20000, .out = 9000, .dst = 0x50000, .in = 0 };
    try run(&t, &t, &l, 0, 1, &x);
    var posts: [3]Desc = undefined;
    var np: usize = 0;
    var got_in: [3]u64 = @splat(0);
    for (t.ops[0..t.n]) |o| switch (o) {
        .post => |d| {
            posts[np] = d;
            np += 1;
        },
        .unstage => |u| {
            try std.testing.expectEqual(@as(u64, 0x40000 + (np - 1) * 4096), u[0]); // each call's next 4096 bytes
            got_in[np - 1] = u[2];
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 3), np);
    try std.testing.expectEqual([3]u32{ 4096, 4096, 0 }, posts[0].bytes);
    try std.testing.expectEqual([3]u32{ 904, 4096, 0 }, posts[1].bytes);
    try std.testing.expectEqual([3]u32{ 0, 808, 0 }, posts[2].bytes);
    try std.testing.expectEqual([3]u64{ 4096, 4096, 808 }, got_in);
    // an empty exchange still takes one call, so every rank posts it
    try std.testing.expectEqual(@as(usize, 1), callCount(0, 4096));
    // shared: block 0 staged and readied once a call; every link's bytes the payload's
    var y: Exchange = .{ .unit = 100, .shared = true };
    y.parts[0] = .{ .src = 0x7000, .out = 100, .dst = 0x100, .in = 60 };
    y.parts[2] = .{ .src = 0x7000, .out = 100, .dst = 0x200, .in = 100 };
    var u: Tape = .{};
    try run(&u, &u, &l, 1, 1, &y);
    var stages: usize = 0;
    var readies: usize = 0;
    for (u.ops[0..u.n]) |o| switch (o) {
        .stage => |a| {
            stages += 1;
            try std.testing.expectEqual(@as(u64, l.send(0, 0)), a[0]);
        },
        .write => |a| readies += @intFromBool(a[0] == l.ready(0, 0) or a[0] == l.ready(0, 1)),
        .post => |d| try std.testing.expect(d.shared and d.bytes[0] == 100 and d.bytes[1] == 100),
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 1), stages);
    try std.testing.expectEqual(@as(usize, 1), readies);
}

test "PRX plan: an exchange that breaks its rules is refused before anything is posted" {
    const l = try Layout.of(3, 2, 4096);
    var t: Tape = .{};
    var x: Exchange = .{ .unit = 10 };
    x.parts[1] = .{ .out = 11 };
    try std.testing.expectError(error.BadExchange, run(&t, &t, &l, 0, 1, &x));
    var y: Exchange = .{ .unit = 10, .shared = true };
    y.parts[1] = .{ .src = 1, .out = 10 };
    y.parts[2] = .{ .src = 2, .out = 10 };
    try std.testing.expectError(error.BadExchange, run(&t, &t, &l, 0, 1, &y));
    try std.testing.expectEqual(@as(usize, 0), t.n);
}
