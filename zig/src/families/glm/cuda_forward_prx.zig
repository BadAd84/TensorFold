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
