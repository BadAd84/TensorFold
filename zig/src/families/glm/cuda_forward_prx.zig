//! TF_GLM_OVN_PRX in a prompt chunk: which split exchanges take the NICs, as relay's direct NCCL postings move them.

const std = @import("std");
const cuda = @import("cuda");
const relay = @import("cuda_relay.zig");
const Relay = relay.Relay;
const plan = cuda.prx_plan;

/// The channel a split exchange on `s` takes: one is open, the ring is off, `s` is not capturing (every rank alike).
pub fn pick(comptime C: type, ch: ?*C, ring: bool, s: cuda.Stream) !?*C {
    if (ring) return null;
    const c = ch orelse return null;
    if (try c.capturing(s)) return null;
    return c;
}

/// The channel of the forward's exchange (`x`: Exchange.nccl), as `pick` decides.
pub fn of(x: anytype, ring: bool, s: cuda.Stream) !?*cuda.prx.Channel {
    return pick(cuda.prx.Channel, if (x.group) |g| g.prx else null, ring, s);
}

/// The most rows any rank owns of the piece: every rank cuts the same calls from it.
fn most(x: Relay) usize {
    var m: usize = 0;
    for (0..x.world) |r| {
        const p = x.rows(r);
        m = @max(m, p[1] - p[0]);
    }
    return m;
}

/// relay.splitPartials as an exchange: each peer its rows of `part`, from each peer this rank's rows into its slot.
pub fn partials(x: Relay, room: u64, part: u64, lo: usize, dims: usize) plan.Exchange {
    const mine = x.rows(x.rank);
    const own = mine[1] - mine[0];
    var e: plan.Exchange = .{ .unit = most(x) * dims * 4 };
    for (0..x.world) |peer| if (peer != x.rank) {
        const theirs = x.rows(peer);
        e.parts[peer] = .{ .src = part + (lo + theirs[0]) * dims * 4, .out = (theirs[1] - theirs[0]) * dims * 4, .dst = room + peer * own * dims * 4, .in = own * dims * 4 };
    };
    return e;
}

/// relay.splitGlued as an exchange: this rank's own bf16 rows to every peer as one payload, theirs into place.
pub fn glued(x: Relay, rows_of: u64, lo: usize, dims: usize) plan.Exchange {
    const mine = x.rows(x.rank);
    var e: plan.Exchange = .{ .unit = most(x) * dims * 2, .shared = true };
    for (0..x.world) |peer| if (peer != x.rank) {
        const theirs = x.rows(peer);
        e.parts[peer] = .{ .src = rows_of + (lo + mine[0]) * dims * 2, .out = (mine[1] - mine[0]) * dims * 2, .dst = rows_of + (lo + theirs[0]) * dims * 2, .in = (theirs[1] - theirs[0]) * dims * 2 };
    };
    return e;
}

/// The channel a forward's split glue asks for a ZC call (zcTake: null when its partials were copied); null: none.
pub fn channelOf(ex: anytype) ?*cuda.prx.Channel {
    return switch (ex) {
        .local => null,
        .nccl => |x| if (x.group) |g| g.prx else null,
    };
}

/// This rank's own rows for a ZC call, { now, later }: from `part` now, or (PEER_FIRST) the room's slot at the glue.
pub fn zcOwn(x: Relay, room: u64, part: u64, lo: usize, dims: usize, own_later: bool) [2]plan.Part {
    const mine = x.rows(x.rank);
    const bytes = (mine[1] - mine[0]) * dims * 4;
    if (own_later) return .{ .{}, .{ .src = room + x.rank * bytes, .out = bytes } };
    return .{ .{ .src = part + (lo + mine[0]) * dims * 4, .out = bytes }, .{} };
}

/// TF_GLM_OVN_PRX_ZC: the split partials on `s` read in place by the glue (keyed by `room`); false: exchange them.
pub fn partialsZc(ch: anytype, s: cuda.Stream, x: Relay, room: u64, part: u64, lo: usize, dims: usize, own_later: bool) !bool {
    if (!ch.zc) return false;
    const o = zcOwn(x, room, part, lo, dims, own_later);
    return ch.exchangeZc(s, &partials(x, room, part, lo, dims), o[0], o[1], room);
}

/// NCCL that records each send and receive (peer, address, bytes), for comparing the exchanges with relay's.
const Rec = struct {
    const Call = struct { send: bool, peer: usize, ptr: u64, bytes: usize };
    var calls: [16]Call = undefined;
    var n: usize = 0;

    fn size(ty: cuda.nccl.DataType) usize {
        return switch (ty) {
            .f32 => 4,
            .bf16 => 2,
            else => 1,
        };
    }
    fn start() callconv(.c) cuda.nccl.Result {
        return 0;
    }
    fn send(p: cuda.abi.DevicePtr, count: usize, ty: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        calls[n] = .{ .send = true, .peer = @intCast(peer), .ptr = p, .bytes = count * size(ty) };
        n += 1;
        return 0;
    }
    fn recv(p: cuda.abi.DevicePtr, count: usize, ty: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        calls[n] = .{ .send = false, .peer = @intCast(peer), .ptr = p, .bytes = count * size(ty) };
        n += 1;
        return 0;
    }
    fn library() cuda.nccl.Library {
        var api: cuda.nccl.Api = undefined;
        api.ncclGroupStart = &start;
        api.ncclGroupEnd = &start;
        api.ncclSend = &send;
        api.ncclRecv = &recv;
        return .{ .lib = undefined, .api = api };
    }
};

/// Every send and receive of `e` against relay's recorded postings: the same peer, address and bytes, each once.
fn expectSame(e: plan.Exchange, x: Relay) !void {
    var seen: usize = 0;
    for (Rec.calls[0..Rec.n]) |c| {
        const p = e.parts[c.peer];
        if (c.send) {
            try std.testing.expectEqual(c.ptr, p.src);
            try std.testing.expectEqual(c.bytes, p.out);
        } else {
            try std.testing.expectEqual(c.ptr, p.dst);
            try std.testing.expectEqual(c.bytes, p.in);
        }
        seen += 1;
    }
    try std.testing.expectEqual(2 * (x.world - 1), seen);
    try std.testing.expect(e.valid(x.rank, x.world));
}

test "TF_GLM_OVN_PRX: a piece's partials and glued rows move what relay's direct NCCL postings move, where they put it" {
    const lib = Rec.library();
    const side: cuda.Stream = .{ .d = undefined, .handle = null };
    for (2..plan.max_world + 1) |w| for ([_]usize{ w, w + 1, 2 * w + 1, 683, 1366 }) |n| {
        var units: [plan.max_world][2]usize = undefined;
        for (0..w) |r| {
            const x: Relay = .{ .world = w, .rank = r, .n = n };
            const dims = 4096;
            Rec.n = 0;
            try relay.splitPartials(&lib, null, side, x, 0x1000_0000, 0x4000_0000, 64, dims);
            const p = partials(x, 0x1000_0000, 0x4000_0000, 64, dims);
            try expectSame(p, x);
            Rec.n = 0;
            try relay.splitGlued(&lib, null, side, x, 0x8000_0000, 64, dims);
            const g = glued(x, 0x8000_0000, 64, dims);
            try expectSame(g, x);
            try std.testing.expect(g.shared and !p.shared);
            units[r] = .{ p.unit, g.unit };
        }
        for (units[1..w]) |u| try std.testing.expectEqual(units[0], u); // every rank cuts the same calls
    };
}

/// A channel that reports a capture state, for `pick`.
const FakeChannel = struct {
    cap: bool,
    fn capturing(c: *const FakeChannel, _: cuda.Stream) !bool {
        return c.cap;
    }
};

test "TF_GLM_OVN_PRX: a split exchange takes the channel unless none is open, the ring relays or the stream captures" {
    const s: cuda.Stream = .{ .d = undefined, .handle = null };
    var live: FakeChannel = .{ .cap = false };
    var captured: FakeChannel = .{ .cap = true };
    try std.testing.expectEqual(@as(?*FakeChannel, &live), try pick(FakeChannel, &live, false, s));
    try std.testing.expectEqual(@as(?*FakeChannel, null), try pick(FakeChannel, &live, true, s));
    try std.testing.expectEqual(@as(?*FakeChannel, null), try pick(FakeChannel, &captured, false, s));
    try std.testing.expectEqual(@as(?*FakeChannel, null), try pick(FakeChannel, null, false, s));
}

/// A channel that records its ZC call, for partialsZc.
const ZcFake = struct {
    zc: bool,
    got: ?struct { x: plan.Exchange, own: plan.Part, later: plan.Part, key: u64 } = null,
    fn exchangeZc(c: *ZcFake, _: cuda.Stream, x: *const plan.Exchange, own: plan.Part, later: plan.Part, key: u64) !bool {
        c.got = .{ .x = x.*, .own = own, .later = later, .key = key };
        return true;
    }
};

test "TF_GLM_OVN_PRX_ZC: a split piece's partials go read in place with this rank's own rows now, or PEER_FIRST's later" {
    const s: cuda.Stream = .{ .d = undefined, .handle = null };
    const x: Relay = .{ .world = 3, .rank = 1, .n = 1408 };
    const dims = 4096;
    const mine = x.rows(1);
    const bytes = (mine[1] - mine[0]) * dims * 4;
    var off: ZcFake = .{ .zc = false };
    try std.testing.expect(!try partialsZc(&off, s, x, 0x1000_0000, 0x4000_0000, 128, dims, false));
    try std.testing.expect(off.got == null);
    var on: ZcFake = .{ .zc = true };
    try std.testing.expect(try partialsZc(&on, s, x, 0x1000_0000, 0x4000_0000, 128, dims, false));
    const a = on.got.?;
    try std.testing.expectEqual(@as(u64, 0x1000_0000), a.key);
    try std.testing.expectEqualDeep(partials(x, 0x1000_0000, 0x4000_0000, 128, dims), a.x);
    try std.testing.expectEqualDeep(plan.Part{ .src = 0x4000_0000 + (128 + mine[0]) * dims * 4, .out = bytes }, a.own);
    try std.testing.expectEqual(@as(usize, 0), a.later.out);
    try std.testing.expect(try partialsZc(&on, s, x, 0x1000_0000, 0x4000_0000, 128, dims, true));
    const b = on.got.?;
    try std.testing.expectEqual(@as(usize, 0), b.own.out);
    // where PEER_FIRST fills this rank's rows: its slot of the room, as the copying path's partials leave them
    try std.testing.expectEqualDeep(plan.Part{ .src = 0x1000_0000 + 1 * bytes, .out = bytes }, b.later);
}
