//! A GLM kept state's disk image across ranks: rank 0's engine thread leases, fills, reads and seals each rank's copy.
const std = @import("std");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const page = @import("cuda_snapshot_pages.zig").page;
const Engine = @import("cuda_engine.zig").Engine;

/// A disk step's op (Header.flags); its id is `a`, its image offset `seed`, its bytes `depth`, its rank `top_k`.
/// `local_` ops: TF_GLM_OVN_DISK_LOCAL's (cuda_disk_local); `carry_` ops: TF_GLM_OVN_DISK_CARRY's (cuda_disk_carry).
pub const Op = enum(u32) { blank = 1, put, take, seal, prefixes, bcast, local_open, local_file, local_unlink, local_other, local_have, local_sweep, local_read, local_status, local_verify, carry_proof, carry_check, carry_record };
const kind = @backingInt(follow.Kind.disk); // each op crosses the follow link as a disk step, its bytes as the ids

/// This rank's image shape (every sequence of the engine has the same; TF_GLM_OVN_DROP_MTP's hold no head caches).
pub fn shape(e: *const Engine) !image.Shape {
    return image.Shape.of(&e.cur.st, &e.c, e.capacity, e.mtp and !@import("cuda_state.zig").drop_mtp);
}

/// The words of a kept state on this rank: its position, head rows, ring bases and KDA copy bits.
pub fn wordsOf(e: *const Engine, id: u64) !image.Words {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    if (s.cur.len > 64) return error.SnapshotShape;
    var cur: u64 = 0;
    for (s.cur, 0..) |b, i| cur |= @as(u64, b) << @intCast(i);
    return .{ .at = s.pos, .mtp_len = s.mtp_len, .base = s.bases[0], .mtp_base = s.bases[1], .cur = cur };
}

/// Bytes [off, off + host.len) of this rank's image of `id`, out (lease to host) or in; unsealed: before its head only.
pub fn copy(e: *Engine, id: u64, off: u64, host: []u8, out: bool) !void {
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    var w: [6]u64 = undefined;
    try (try copier(e, id, off + host.len, &segs, &w)).run(off, host, out);
}

/// `id`'s image on this rank for bytes up to `end` (unsealed: before its head only); `segs` and `w` hold its parts.
pub fn copier(e: *Engine, id: u64, end: u64, segs: *std.ArrayList(image.Seg), w: *[6]u64) !image.Copy {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const sh = try shape(e);
    const at: u32 = @intCast(s.pos);
    if (s.len == 0 and end > image.Words.bytes + image.front(sh, at)) return error.DiskOrder;
    try image.segments(sh, at, s.mtp_len, segs, e.gpa);
    if (@import("cuda_lineage.zig").on) try @import("cuda_lineage_copy.zig").remap(e, s, segs); // a padded lease
    w.* = (try wordsOf(e, id)).encode();
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    return .{ .d = e.d, .pool = pool, .lease = &s.lease, .segs = segs.items, .words = w };
}

/// Rank 0's link at TP, else null (one rank: every op is local).
pub fn leader(e: *Engine) ?follow.Leader {
    const l = e.link orelse return null;
    return .{ .e = e, .link = @ptrCast(@alignCast(l)) };
}

/// Rank 0: whether the other ranks sleep on their lines (no stream), and putting them back to sleep after disk steps.
pub fn asleep(e: *Engine) bool {
    return if (leader(e)) |x| x.link.asleep else false;
}

pub fn rest(e: *Engine) !void {
    if (leader(e)) |x| try x.idle();
}

/// Rank 0: a new id leased at `at` on every rank (rank 0 first: the ranks' pools match, so theirs cannot fail apart).
pub fn blank(e: *Engine, at: u32) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try e.blankAs(id, at);
    e.next_snap += 1;
    if (leader(e)) |x| try x.disk(.{ .kind = kind, .flags = @backingInt(Op.blank), .a = id, .seed = at }, &.{});
    return id;
}

/// Rank 0: `bytes` into rank `rank`'s image of `id` at `off` (rank 0's own lease, or through the link in pieces).
pub fn put(e: *Engine, rank: u32, id: u64, off: u64, bytes: []const u8) !void {
    if (rank == 0) return copy(e, id, off, @constCast(bytes), false);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4;
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(room, bytes.len - done);
        const words = (n + 3) / 4;
        @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words])[0..n], bytes[done..][0..n]);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.put), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words]);
        done += n;
    }
}

/// Rank 0: rank `rank`'s image bytes of `id` from `off` into `out`.
pub fn take(e: *Engine, rank: u32, id: u64, off: u64, out: []u8) !void {
    if (rank == 0) return copy(e, id, off, out, true);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4;
    var done: usize = 0;
    while (done < out.len) {
        const n = @min(room, out.len - done);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
        try x.link.from(rank, n);
        try x.link.buf.download(64, out[done..][0..n]);
        done += n;
    }
}

/// Rank 0: every rank's blank `id` sealed with `w` (its head's rows, ring bases and KDA copy bits).
pub fn seal(e: *Engine, id: u64, w: image.Words) !void {
    try e.sealState(id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    if (leader(e)) |x| {
        const words = w.encode();
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.seal), .a = id }, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&words)));
    }
}

/// Most lease runs one bcast step carries (each inside one snapshot page on every rank).
const bcast_runs = 64;

/// TF_GLM_OVN_DISK_BCAST, rank 0: shared bytes at `off` of its image into its lease, then every rank's by broadcast.
pub fn share(e: *Engine, id: u64, off: u64, bytes: []const u8, pre: []const u64) !void {
    try copy(e, id, off, @constCast(bytes), false);
    const x = leader(e) orelse return;
    try e.d.synchronize(); // the synchronous uploads into rank 0's lease have landed before a broadcast reads them
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    var w: [6]u64 = undefined;
    const c = try copier(e, id, off + bytes.len, &segs, &w);
    var ids: [2 + 3 * bcast_runs]u32 = undefined;
    ids[0..2].* = halves(pre[0]);
    var k: usize = 0;
    const end = off + bytes.len;
    for (c.segs) |s| {
        const base = switch (s.from) {
            .lease => |l| l,
            else => continue, // the words never are shared bytes; zeros are not in any lease
        };
        var at = @max(off, s.img);
        const hi = @min(end, s.img + s.len);
        while (at < hi) {
            const past = base + (at - s.img) - pre[0]; // lease bytes past rank 0's prefix: rank r's sit at its own
            var n = hi - at;
            for (pre) |q| n = @min(n, page - (past + q) % page);
            ids[2 + 3 * k ..][0..3].* = .{ halves(at)[0], halves(at)[1], @intCast(n) };
            k += 1;
            at += n;
            if (k == bcast_runs) {
                try bcastStep(x, e, id, ids[0 .. 2 + 3 * k]);
                k = 0;
            }
        }
    }
    if (k > 0) try bcastStep(x, e, id, ids[0 .. 2 + 3 * k]);
}

fn halves(v: u64) [2]u32 {
    return .{ @truncate(v), @intCast(v >> 32) };
}

fn bcastStep(x: follow.Leader, e: *Engine, id: u64, ids: []const u32) !void {
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.bcast), .a = id }, ids);
    try runs(x.link, e, id, ids);
}

/// One broadcast of a bcast step: this rank's lease address and its bytes.
const Run = struct { ptr: u64, n: usize };

/// Every rank: a bcast step's runs (rank 0's prefix, then offset and length each) from rank 0's lease into this rank's.
fn runs(l: *follow.Link, e: *Engine, id: u64, ids: []const u32) !void {
    if (ids.len < 5 or (ids.len - 2) % 3 != 0) return error.BadStep;
    const k = (ids.len - 2) / 3;
    if (k > bcast_runs) return error.BadStep;
    const p0 = @as(u64, ids[0]) | @as(u64, ids[1]) << 32;
    const mine = (try shape(e)).prefix;
    var end: u64 = 0;
    var at: [bcast_runs]Run = undefined;
    for (0..k) |i| {
        const t = ids[2 + 3 * i ..][0..3];
        const off = (@as(u64, t[0]) | @as(u64, t[1]) << 32) - p0 + mine;
        if (t[2] == 0) return error.BadStep;
        end = @max(end, off + t[2]);
        at[i] = .{ .ptr = off, .n = t[2] };
    }
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    var w: [6]u64 = undefined;
    const c = try copier(e, id, end, &segs, &w);
    for (at[0..k]) |*r| r.ptr = try c.device(r.ptr, r.n); // every address resolved before the first broadcast
    const Post = struct {
        l: *follow.Link,
        at: []const Run,
        fn post(q: @This()) anyerror!void {
            for (q.at) |r| try q.l.lib.check(q.l.lib.api.ncclBroadcast(r.ptr, r.ptr, r.n, .u8, 0, q.l.comm, q.l.stream.handle), "ncclBroadcast");
        }
    };
    try l.lib.group(Post{ .l = l, .at = at[0..k] }, Post.post);
    try l.stream.synchronize();
}

/// Rank 0: every rank's prefix bytes in rank order (the part of a state each rank holds alone).
pub fn prefixes(e: *Engine, out: []u64) !void {
    const mine = (try shape(e)).prefix;
    const x = leader(e) orelse {
        if (out.len != 1) return error.NoRank;
        out[0] = mine;
        return;
    };
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.prefixes) }, &.{});
    try gather(x.link, e, mine, out);
}

fn gather(l: *follow.Link, e: *const Engine, mine: u64, out: []u64) !void {
    try @import("cuda_ranks.zig").gather(u64, l.d, l.lib, l.comm, l.stream, e.share.rank, mine, out);
}

/// A rank other than 0: rank 0's disk step on this rank's copy of its state.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32) !void {
    if (h.flags < @backingInt(Op.blank) or h.flags > @backingInt(Op.carry_record)) return error.BadStep;
    const op: Op = @fromBackingInt(h.flags);
    const rank: u32 = @intCast(e.share.rank);
    switch (op) {
        .blank => try e.blankAs(h.a, @intCast(h.seed)),
        .seal => {
            if (ids.len != 12) return error.BadStep;
            var x: [6]u64 = undefined;
            @memcpy(std.mem.sliceAsBytes(&x), std.mem.sliceAsBytes(ids));
            const w = try image.Words.decode(x, @intCast((e.snaps.get(h.a) orelse return error.NoKeptState).pos));
            try e.sealState(h.a, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
        },
        .put => if (h.top_k == rank) {
            if (h.depth > ids.len * 4) return error.BadStep;
            try copy(e, h.a, h.seed, @constCast(std.mem.sliceAsBytes(ids))[0..h.depth], false);
        },
        .take => {
            if (h.depth > l.ids.len * 4) return error.BadStep;
            if (h.top_k == rank) {
                const host = std.mem.sliceAsBytes(l.ids)[0..h.depth];
                try copy(e, h.a, h.seed, host, true);
                try l.buf.uploadAsync(64, host, l.stream.handle);
            }
            try l.from(h.top_k, h.depth);
        },
        .prefixes => {
            var out: [64]u64 = undefined;
            if (e.share.world > out.len) return error.BadStep;
            try gather(l, e, (try shape(e)).prefix, out[0..e.share.world]);
        },
        .bcast => try runs(l, e, h.a, ids),
        .carry_proof, .carry_check, .carry_record => try @import("cuda_disk_carry.zig").step(e, l, h, ids, op),
        else => try @import("cuda_disk_local.zig").step(e, l, h, ids, op),
    }
}
