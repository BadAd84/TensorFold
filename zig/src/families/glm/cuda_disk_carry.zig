//! TF_GLM_OVN_DISK_CARRY's disk steps: each rank's hashes of its probe images, and (DISK_LOCAL) its record on its disk.
const std = @import("std");
const lanes = @import("lanes");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const ranks = @import("cuda_disk_ranks.zig");
const Path = @import("cuda_disk_local.zig").Path;
const state = @import("cuda_state.zig");
const probe_rows = @import("cuda_native.zig").snap_min_rows; // the identity's probe state (cuda_glm_disk.probe)
const Engine = @import("cuda_engine.zig").Engine;
const record = lanes.carry_record;
const Op = ranks.Op;

const kind = @backingInt(follow.Kind.disk);

/// Rows the carry probe's prompt runs past its last mark, as the identity's probe does (the head reads one on).
pub const tail: u32 = 40;

/// Rows a mark is a multiple of: TF_GLM_OVN_KDA_CHUNKED's 32-row grid and the index's 4-row pools both divide it.
const grid: u32 = 64;
comptime {
    std.debug.assert(grid % @import("cuda_kda_chunk.zig").sub_rows == 0 and grid % @import("cuda_sparse.zig").pool == 0);
}

/// The carry probe's marks within `capacity`: past top-k keys and a chunk, past the ring; else the longest state's row.
pub fn marks(capacity: usize, out: *[2]u32) []const u32 {
    const want = [_]u32{ state.prefill_rows + state.prefill_rows / 2 + grid, state.index_ring + state.prefill_rows / 2 + 2 * grid };
    var n: usize = 0;
    for (want) |m| if (m + tail <= capacity) {
        out[n] = m;
        n += 1;
    };
    if (n < want.len and capacity > tail) {
        const top: u32 = @intCast(@min((capacity - tail) / grid * grid, std.math.maxInt(u32)));
        if (top > (if (n > 0) out[n - 1] else probe_rows)) {
            out[n] = top;
            n += 1;
        }
    }
    return out[0..n];
}

/// This rank's proof: its own image of each probe state hashed and folded from the probe's first state on.
var proof: u64 = 0;
/// Every state of the last probe hashed on this rank: without it the rank answers no and writes no record.
var proven = false;

fn fold(v: u64, first: bool) void {
    if (first) {
        proof = 0;
        proven = true;
    }
    proof = record.fold(proof, v);
    if (v == record.unhashed) proven = false;
}

/// The hash of this rank's own image of kept state `id`: its words, own fields, shared rows and head, 16 MiB a piece.
pub fn ownHash(e: *Engine, id: u64) !u64 {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const total = image.imageBytes(try ranks.shape(e), @intCast(s.pos));
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    var w: [6]u64 = undefined;
    const c = try ranks.copier(e, id, total, &segs, &w);
    const piece: u64 = 16 << 20;
    const buf = try std.heap.page_allocator.alloc(u8, @intCast(@min(piece, total)));
    defer std.heap.page_allocator.free(buf);
    var h = std.hash.Wyhash.init(0x63617272);
    var off: u64 = 0;
    while (off < total) : (off += piece) {
        const n: usize = @intCast(@min(piece, total - off));
        try c.run(off, buf[0..n], true);
        h.update(buf[0..n]);
    }
    const v = h.final();
    return if (v == record.unhashed) v - 1 else v; // never the word that says a rank could not hash
}

/// Rank 0: every rank's hash of its own image of probe state `id` in rank order (`first`: each rank's fold starts).
pub fn hashes(e: *Engine, id: u64, first: bool, out: []u64) ![]const u64 {
    const world = e.share.world;
    if (world > out.len) return error.TooManyRanks;
    const x = ranks.leader(e) orelse {
        out[0] = ownHash(e, id) catch record.unhashed;
        return out[0..1];
    };
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.carry_proof), .a = id, .seed = @intFromBool(first) }, &.{});
    const v = ownHash(e, id) catch record.unhashed; // rank 0 joins the gather whatever its own hash did
    try gather(x.link, e, v, out[0..world]);
    return out[0..world];
}

/// Rank 0: whether every other rank's own directory `dir` (id `id`) records carry key `key` and that rank's proof.
pub fn check(e: *Engine, dir: []const u8, id: u64, key: u64) !bool {
    const x = ranks.leader(e) orelse return true;
    var p: Path = .{};
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.carry_check), .a = key, .seed = id, .depth = @intCast(dir.len) }, try p.of(dir));
    var out: [64]u64 = undefined;
    if (e.share.world > out.len) return error.TooManyRanks;
    try gather(x.link, e, 1, out[0..e.share.world]); // rank 0's own record was checked before it asked (rows_carry)
    return record.all(out[0..e.share.world]);
}

/// Rank 0: every other rank writes its record of the store's directory `dir` (id `id`) under `key`, on its own disk.
pub fn records(e: *Engine, dir: []const u8, id: u64, key: u64) !void {
    const x = ranks.leader(e) orelse return;
    var p: Path = .{};
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.carry_record), .a = key, .seed = id, .depth = @intCast(dir.len) }, try p.of(dir));
}

fn gather(l: *follow.Link, e: *const Engine, v: u64, out: []u64) !void {
    try @import("cuda_ranks.zig").gather(u64, l.d, l.lib, l.comm, l.stream, e.share.rank, v, out);
}

/// A rank other than 0: rank 0's carry step. Its own failures (a hash, a read, a write) make it answer no, never stop.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32, op: Op) !void {
    var out: [64]u64 = undefined;
    if (e.share.world > out.len) return error.BadStep;
    const rank = e.share.rank;
    switch (op) {
        .carry_proof => {
            const v = ownHash(e, h.a) catch |err| blk: {
                std.log.warn("glm rank {d}: hashing its image of a carry probe state failed ({s})", .{ rank, @errorName(err) });
                break :blk record.unhashed;
            };
            fold(v, h.seed != 0);
            try gather(l, e, v, out[0..e.share.world]);
        },
        .carry_check => {
            const dir = try Path.read(h, ids);
            const why = if (proven) record.refusal(record.read(dir), .{ .dir = h.seed, .key = h.a, .proof = proof }) else "its probe states were not all hashed";
            if (why) |w| std.log.info("glm rank {d}: no carry from {s}: {s}", .{ rank, dir, w });
            try gather(l, e, @intFromBool(why == null), out[0..e.share.world]);
        },
        .carry_record => {
            const dir = try Path.read(h, ids);
            const want: record.Record = .{ .dir = h.seed, .key = h.a, .proof = proof };
            if (!proven or record.refusal(record.read(dir), want) == null) return;
            record.write(dir, want) catch |err| std.log.warn("glm rank {d}: its carry record in {s} was not written ({s})", .{ rank, dir, @errorName(err) });
        },
        else => return error.BadStep,
    }
}

test "glm disk carry: the probe's marks pass the top-k keys, a chunk and the ring, or reach the longest state" {
    var buf: [2]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 6208, 10368 }, marks(1 << 20, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 6208, 10368 }, marks(10368 + tail, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 6208, 10304 }, marks(10368 + tail - 1, &buf));
    try std.testing.expectEqualSlices(u32, &.{4032}, marks(4096, &buf));
    try std.testing.expectEqualSlices(u32, &.{}, marks(probe_rows + tail + grid - 1, &buf));
    try std.testing.expectEqualSlices(u32, &.{}, marks(0, &buf));
    for ([_]usize{ 4096, 8192, 1 << 20 }) |cap| for (marks(cap, &buf)) |m| {
        try std.testing.expect(m % grid == 0 and m + tail <= cap and m > probe_rows);
        try std.testing.expect(m % state.prefill_rows != 0); // a cut inside a chunk, not where one ends anyway
    };
}
