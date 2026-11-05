//! GLM's split exchanges over NCCL: straight to every rank, or passed along the ring (TF_GLM_PREFILL_REDUCE=ring).

const std = @import("std");
const cuda = @import("cuda");

/// The ring's plan, shared with every family's row-split exchange (cuda/relay_plan.zig).
pub const Relay = cuda.relay_plan.Relay;

/// A piece's partials (`part`: this rank's fp32 rows) into `room`'s slots on `side`; NCCL keeps a pair's posting order.
pub fn partials(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, side: cuda.Stream, x: Relay, room: u64, part: u64, lo: usize, dims: usize) !void {
    for (1..x.world) |s| try lib.group(Step{ .lib = lib, .comm = comm, .side = side, .x = x, .s = s, .base = room, .part = part, .lo = lo, .dims = dims }, Step.partials);
}

/// A piece's glued rows (`rows_of`, bf16) to every rank, each step sending on what the last one brought in.
pub fn glued(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, side: cuda.Stream, x: Relay, rows_of: u64, lo: usize, dims: usize) !void {
    for (1..x.world) |s| try lib.group(Step{ .lib = lib, .comm = comm, .side = side, .x = x, .s = s, .base = rows_of, .lo = lo, .dims = dims }, Step.glued);
}

/// Step `s` of an exchange: its postings, which `Library.group` puts in one group.
const Step = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    side: cuda.Stream,
    x: Relay,
    s: usize,
    base: u64, // the piece's room (partials) or the glued rows
    part: u64 = 0, // this rank's partial (partials)
    lo: usize,
    dims: usize,

    fn send(t: Step, src: u64, count: usize, ty: cuda.nccl.DataType) !void {
        try t.lib.check(t.lib.api.ncclSend(src, count, ty, @intCast(t.x.next()), t.comm, t.side.handle), "ncclSend");
    }
    fn recv(t: Step, dst: u64, count: usize, ty: cuda.nccl.DataType) !void {
        try t.lib.check(t.lib.api.ncclRecv(dst, count, ty, @intCast(t.x.prev()), t.comm, t.side.handle), "ncclRecv");
    }
    fn partials(t: Step) anyerror!void {
        const x = t.x;
        const mine = x.rows(x.rank);
        const own = mine[1] - mine[0];
        const from = x.transit((t.s - 1) % 2); // what step s - 1 kept (step 1 sends from this rank's partial)
        const into = x.transit(t.s % 2);
        for (0..x.world - t.s) |i| {
            const p = x.rows(x.owner(i));
            const src = if (t.s == 1) t.part + (t.lo + p[0]) * t.dims * 4 else t.base + (from + x.at(i)) * t.dims * 4;
            try t.send(src, (p[1] - p[0]) * t.dims, .f32);
        }
        try t.recv(t.base + x.takenFrom(t.s) * own * t.dims * 4, own * t.dims, .f32);
        for (0..x.world - t.s - 1) |i| {
            const p = x.rows(x.owner(i));
            try t.recv(t.base + (into + x.at(i)) * t.dims * 4, (p[1] - p[0]) * t.dims, .f32);
        }
    }
    fn glued(t: Step) anyerror!void {
        const out = t.x.rows(t.x.sentFrom(t.s));
        const got = t.x.rows(t.x.takenFrom(t.s));
        try t.send(t.base + (t.lo + out[0]) * t.dims * 2, (out[1] - out[0]) * t.dims, .bf16);
        try t.recv(t.base + (t.lo + got[0]) * t.dims * 2, (got[1] - got[0]) * t.dims, .bf16);
    }
};

/// Split's direct exchange of a piece's partials: each rank's block of every owner's rows straight to that owner.
pub fn splitPartials(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, side: cuda.Stream, x: Relay, room: u64, part: u64, lo: usize, dims: usize) !void {
    try lib.group(Direct{ .lib = lib, .comm = comm, .side = side, .x = x, .base = room, .part = part, .lo = lo, .dims = dims }, Direct.partials);
}

/// Split's direct exchange of a piece's glued rows (`rows_of`, bf16): each rank's own rows straight to every rank.
pub fn splitGlued(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, side: cuda.Stream, x: Relay, rows_of: u64, lo: usize, dims: usize) !void {
    try lib.group(Direct{ .lib = lib, .comm = comm, .side = side, .x = x, .base = rows_of, .lo = lo, .dims = dims }, Direct.glued);
}

/// A direct exchange's postings, every peer's send and receive, which `Library.group` puts in one group.
const Direct = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    side: cuda.Stream,
    x: Relay,
    base: u64, // the piece's room (partials) or the glued rows
    part: u64 = 0, // this rank's partial (partials)
    lo: usize,
    dims: usize,

    /// TF_GLM_OVN_EXCHANGE_KB: a direct exchange's send and receive in pieces of at most this many KiB (0: whole);
    /// a 4,096-row chunk's 11 MB share gathers at a seventh of the rate a 4 MB one does, and the pieces land where the
    /// whole would, byte for byte.
    fn pieceBytes() usize {
        return @as(usize, @intCast(cuda.ovn.int("EXCHANGE_KB", 0) catch 0)) << 10; // ovn caches the read itself
    }
    fn post(t: Direct, peer: usize, src: u64, out: usize, dst: u64, in: usize, ty: cuda.nccl.DataType) !void {
        const width: usize = if (ty == .f32) 4 else 2;
        const piece = pieceBytes() / width; // elements a piece
        if (piece == 0 or (out <= piece and in <= piece)) {
            try t.lib.check(t.lib.api.ncclSend(src, out, ty, @intCast(peer), t.comm, t.side.handle), "ncclSend");
            try t.lib.check(t.lib.api.ncclRecv(dst, in, ty, @intCast(peer), t.comm, t.side.handle), "ncclRecv");
            return;
        }
        var so: usize = 0;
        var ro: usize = 0;
        while (so < out or ro < in) {
            if (so < out) {
                const n = @min(piece, out - so);
                try t.lib.check(t.lib.api.ncclSend(src + so * width, n, ty, @intCast(peer), t.comm, t.side.handle), "ncclSend");
                so += n;
            }
            if (ro < in) {
                const n = @min(piece, in - ro);
                try t.lib.check(t.lib.api.ncclRecv(dst + ro * width, n, ty, @intCast(peer), t.comm, t.side.handle), "ncclRecv");
                ro += n;
            }
        }
    }
    fn partials(t: Direct) anyerror!void {
        const x = t.x;
        const mine = x.rows(x.rank);
        const own = mine[1] - mine[0];
        for (0..x.world) |peer| if (peer != x.rank) {
            const theirs = x.rows(peer);
            try t.post(peer, t.part + (t.lo + theirs[0]) * t.dims * 4, (theirs[1] - theirs[0]) * t.dims, t.base + peer * own * t.dims * 4, own * t.dims, .f32);
        };
    }
    fn glued(t: Direct) anyerror!void {
        const x = t.x;
        const mine = x.rows(x.rank);
        for (0..x.world) |peer| if (peer != x.rank) {
            const theirs = x.rows(peer);
            try t.post(peer, t.base + (t.lo + mine[0]) * t.dims * 2, (mine[1] - mine[0]) * t.dims, t.base + (t.lo + theirs[0]) * t.dims * 2, (theirs[1] - theirs[0]) * t.dims, .bf16);
        };
    }
};

/// NCCL for the relay's postings: each call in order, and the call numbered `fail` returns an error.
const Fake = struct {
    const Kind = enum { start, end, send, recv };
    const Call = struct { kind: Kind, peer: c_int = -1, ptr: u64 = 0, count: usize = 0 };
    var calls: [256]Call = undefined;
    var n: usize = 0;
    var fail: ?usize = null;

    fn note(c: Call) cuda.nccl.Result {
        if (n < calls.len) calls[n] = c;
        n += 1;
        return if (fail == n - 1) 1 else 0;
    }
    fn start() callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .start });
    }
    fn end() callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .end });
    }
    fn send(ptr: cuda.abi.DevicePtr, count: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .send, .peer = peer, .ptr = ptr, .count = count });
    }
    fn recv(ptr: cuda.abi.DevicePtr, count: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .recv, .peer = peer, .ptr = ptr, .count = count });
    }
    fn library() cuda.nccl.Library {
        var api: cuda.nccl.Api = undefined;
        api.ncclGroupStart = &start;
        api.ncclGroupEnd = &end;
        api.ncclSend = &send;
        api.ncclRecv = &recv;
        return .{ .lib = undefined, .api = api };
    }
    const Mode = enum { ring_partials, ring_glued, split_partials, split_glued };
    /// One exchange's whole schedule on `lib`: a piece's partials or glued rows, along the ring or direct.
    fn run(lib: *const cuda.nccl.Library, x: Relay, mode: Mode) !void {
        const side: cuda.Stream = .{ .d = undefined, .handle = null };
        switch (mode) {
            .ring_partials => try partials(lib, null, side, x, 0, 0, 0, 8),
            .ring_glued => try glued(lib, null, side, x, 0, 0, 8),
            .split_partials => try splitPartials(lib, null, side, x, 0, 0, 0, 8),
            .split_glued => try splitGlued(lib, null, side, x, 0, 0, 8),
        }
    }
};

test "a posting that fails, at each posting in turn, ends its group and posts nothing after it, ring or direct" {
    const lib = Fake.library();
    for (2..7) |w| for (0..w) |r| for ([_]Fake.Mode{ .ring_partials, .ring_glued, .split_partials, .split_glued }) |mode| {
        const x: Relay = .{ .world = w, .rank = r, .n = 2 * w + 1 };
        const ring = mode == .ring_partials or mode == .ring_glued;
        Fake.n = 0;
        Fake.fail = null;
        try Fake.run(&lib, x, mode);
        const total = Fake.n;
        const clean = Fake.calls; // each failing run below writes over the calls
        var peers: usize = 0; // a direct exchange sends to and takes from every other rank once
        for (clean[0..total]) |c| switch (c.kind) {
            .send => if (ring) try std.testing.expectEqual(@as(c_int, @intCast(x.next())), c.peer) else {
                try std.testing.expect(c.peer >= 0 and c.peer < w and c.peer != r);
                peers += 1;
            },
            .recv => if (ring) try std.testing.expectEqual(@as(c_int, @intCast(x.prev())), c.peer),
            else => {},
        };
        if (!ring) try std.testing.expectEqual(w - 1, peers);
        for (0..total) |k| {
            const failed = clean[k].kind;
            Fake.n = 0;
            Fake.fail = k;
            try std.testing.expectError(error.NcclFailed, Fake.run(&lib, x, mode));
            var opened: usize = 0; // a failed GroupStart opens nothing
            var ended: usize = 0;
            for (Fake.calls[0..Fake.n], 0..) |c, i| switch (c.kind) {
                .start => opened += @intFromBool(i != k),
                .end => ended += 1,
                else => {},
            };
            try std.testing.expectEqual(opened, ended);
            try std.testing.expectEqual(if (failed == .send or failed == .recv) k + 2 else k + 1, Fake.n);
        }
    };
}

test "TF_GLM_OVN_EXCHANGE_KB: a direct exchange in pieces covers each whole posting's bytes, per peer and kind, in order" {
    const lib = Fake.library();
    const x: Relay = .{ .world = 3, .rank = 1, .n = 4096 };
    cuda.ovn.fake("EXCHANGE_KB", null);
    Fake.n = 0;
    Fake.fail = null;
    try Fake.run(&lib, x, .split_partials);
    const whole_n = Fake.n;
    var whole: [256]Fake.Call = undefined;
    @memcpy(whole[0..whole_n], Fake.calls[0..whole_n]);
    cuda.ovn.fake("EXCHANGE_KB", "16"); // 16 KiB: a 1,366-row share of 8 dims at 4 bytes is 43 KiB, so three pieces
    defer cuda.ovn.fake("EXCHANGE_KB", null);
    Fake.n = 0;
    try Fake.run(&lib, x, .split_partials);
    try std.testing.expect(Fake.n > whole_n);
    try std.testing.expectEqual(Fake.Kind.start, Fake.calls[0].kind);
    try std.testing.expectEqual(Fake.Kind.end, Fake.calls[Fake.n - 1].kind);
    // NCCL pairs postings per peer and kind in order: for each whole send (or receive) to a peer, the pieces of that
    // kind to that peer, taken in order, start where it starts, stay within the piece size, and total its count
    for (whole[0..whole_n]) |w| {
        if (w.kind != .send and w.kind != .recv) continue;
        var covered: usize = 0;
        var at: u64 = w.ptr;
        var seen_first = false;
        for (Fake.calls[0..Fake.n]) |c| {
            if (c.kind != w.kind or c.peer != w.peer) continue;
            if (!seen_first) {
                if (c.ptr != w.ptr) continue; // pieces of an earlier whole posting to the same peer
                seen_first = true;
            }
            if (covered >= w.count) break;
            try std.testing.expectEqual(at, c.ptr);
            try std.testing.expect(c.count <= (16 << 10) / 4);
            covered += c.count;
            at += c.count * 4;
        }
        try std.testing.expectEqual(w.count, covered);
    }
}
