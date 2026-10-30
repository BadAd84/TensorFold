//! TF_GLM_OVN_PRX's prompt schedules (pieced, paired; copied or ZC) on emulated ranks: bytes exact, slots measured.

const std = @import("std");
const plan = @import("prx_plan.zig");
const emu = @import("prx_emu.zig");
const pattern = emu.pattern;

const D = 16; // a row's floats: GLM's 4096 scaled by 1/256, so M = slot_mb * 4096 bytes keeps the calls a block
const piece_rows = 1408; // a 4,096-row chunk in TF_GLM_OVERLAP_PIECES=3 pieces of whole 128-row steps

const Mode = enum { pieced, pair };

const Opts = struct {
    mode: Mode,
    zc: bool,
    slots: usize,
    slot_mb: usize = 8,
    pieces: usize = 3,
    sites: usize = 4, // two layers' mixer and ffn sites
    chunks: usize = 2,
    own_later: bool = false, // TF_GLM_OVN_PEER_FIRST's own rows, copied into the block when the glue takes the call

    pub fn adjust(_: *const Opts, _: usize, _: *emu.Proxy, _: *emu.Nic) void {}
};

pub const Stats = struct { bad_bytes: usize = 0, hung: bool = false, in_use_max: u64 = 0, credit_stalls: u64 = 0, slot_waits: u64 = 0, zc_calls: usize = 0 };

/// One exchange's buffers on one rank: what it sends each peer (or its own rows) and where each rank's lands.
const Bufs = struct { src: [plan.max_world][]u8, dst: [plan.max_world][]u8 };

fn rowsOf(r: usize) usize {
    return std.math.divCeil(usize, piece_rows - r, 3) catch unreachable; // rank_scatter.share's 470, 469, 469
}

fn run(o: Opts) !Stats {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const world = 3;
    const w = try a.create(emu.World);
    w.* = .{ .l = try plan.Layout.of(world, o.slots, o.slot_mb * 4096), .world = world };
    try w.start(a, 1, &o);
    var stopped = false;
    defer if (!stopped) w.stop();
    const n_ex = o.chunks * o.sites * o.pieces * 2; // a piece's partials then its glued rows
    var bufs: [world][]Bufs = undefined;
    var main: [world]emu.Stream = undefined;
    var side: [world]emu.Stream = undefined;
    var calls: [world]usize = @splat(0);
    for (0..world) |r| {
        main[r] = .{ .gpa = a, .w = w, .rank = r };
        side[r] = .{ .gpa = a, .w = w, .rank = r };
        bufs[r] = try a.alloc(Bufs, n_ex);
    }
    // every rank's host, alike, in cuda_forward_pieces' order (pieced) or cuda_forward_pair's (pair)
    var ev: usize = 0;
    var zc: [16]?u64 = undefined;
    for (0..o.chunks) |ch| for (0..o.sites) |site| {
        const base = (ch * o.sites + site) * o.pieces;
        for (0..world) |r| {
            const s = &side[r];
            const glue = if (o.mode == .pair) s else &main[r];
            const e = struct {
                fn of(b: usize, q: usize) usize {
                    return 2 * (b + q);
                }
            }.of;
            if (o.mode == .pieced) for (0..o.pieces) |q| { // exchangePieces: every piece's partials, then the glue
                zc[q] = try partials(a, w, r, s, &bufs[r][e(base, q)], e(base, q), o, &calls[r]);
                try s.signal(ev + q);
            };
            for (0..o.pieces) |q| {
                if (o.mode == .pair) zc[q] = try partials(a, w, r, s, &bufs[r][e(base, q)], e(base, q), o, &calls[r]);
                if (glue != s) try glue.await_(ev + q);
                if (zc[q]) |c| try readZc(w, r, glue, &bufs[r][e(base, q)], c, o);
                if (glue != s) {
                    try glue.signal(ev + o.pieces + q);
                    try s.await_(ev + o.pieces + q);
                }
                try glued(a, w, r, s, &bufs[r][e(base, q) + 1], e(base, q) + 1);
                if (glue != s) try s.signal(ev + 2 * o.pieces + q);
            }
            if (glue != s) for (0..o.pieces) |q| try glue.await_(ev + 2 * o.pieces + q);
        }
        ev += 3 * o.pieces;
    };
    var threads: [world][2]std.Thread = undefined;
    for (0..world) |r| {
        threads[r][0] = try std.Thread.spawn(.{}, emu.Stream.execute, .{&side[r]});
        threads[r][1] = try std.Thread.spawn(.{}, emu.Stream.execute, .{&main[r]});
    }
    var out: Stats = .{};
    for (0..world) |r| {
        for (&threads[r]) |*t| t.join();
        out.hung = out.hung or side[r].hung or main[r].hung;
        out.slot_waits += side[r].slot_waits + main[r].slot_waits;
        out.in_use_max = @max(out.in_use_max, w.proxies[r].in_use_max);
        out.credit_stalls += w.proxies[r].credit_stalls;
        out.zc_calls += calls[r];
    }
    w.stop();
    stopped = true;
    for (0..world) |r| for (bufs[r], 0..) |b, e| for (0..world) |k| {
        const key: usize = if (k == r) 0xfe else if (e % 2 == 1) 0xff else r;
        for (b.dst[k], 0..) |x, i| out.bad_bytes += @intFromBool(x != pattern(k, e, key, i));
    };
    return out;
}

fn fill(a: std.mem.Allocator, from: usize, e: usize, to: usize, n: usize) ![]u8 {
    const s = try a.alloc(u8, n);
    for (s, 0..) |*x, i| x.* = pattern(from, e, to, i);
    return s;
}

/// A piece's partials on `side`: each peer its rows' fp32 block; the ZC call when read in place, else null.
fn partials(a: std.mem.Allocator, w: *emu.World, r: usize, side: *emu.Stream, b: *Bufs, e: usize, o: Opts, zc: *usize) !?u64 {
    var x: plan.Exchange = .{ .unit = rowsOf(0) * D * 4 };
    for (0..3) |k| {
        b.dst[k] = try a.alloc(u8, rowsOf(r) * D * 4);
        b.src[k] = try fill(a, r, e, if (k == r) 0xfe else k, rowsOf(if (k == r) r else k) * D * 4);
        if (k != r) x.parts[k] = .{ .src = @intFromPtr(b.src[k].ptr), .out = b.src[k].len, .dst = @intFromPtr(b.dst[k].ptr), .in = b.dst[k].len };
    }
    const own: plan.Part = .{ .src = @intFromPtr(b.src[r].ptr), .out = b.src[r].len };
    if (o.zc) if (try plan.runZc(w.proxies[r], side, &w.l, r, 1, &x, if (o.own_later) .{} else own)) |c| {
        zc.* += 1;
        return c;
    };
    try plan.run(w.proxies[r], side, &w.l, r, 1, &x);
    @memcpy(b.dst[r], b.src[r]); // the copying path's own rows: a plain copy
    return null;
}

/// The glue's read of a ZC call on `glue`: every rank's block (this rank's own copied in now if PEER_FIRST), then done.
fn readZc(w: *emu.World, r: usize, glue: *emu.Stream, b: *Bufs, c: u64, o: Opts) !void {
    if (o.own_later) try glue.stage(w.l.recv(@intCast(c % w.l.slots), r), @intFromPtr(b.src[r].ptr), b.src[r].len);
    for (0..3) |k| try glue.unstage(@intFromPtr(b.dst[k].ptr), plan.zcBlocks(&w.l, c) + k * w.l.M, b.dst[k].len);
    try plan.zcDone(glue, &w.l, c);
}

/// A piece's glued rows on `side`: this rank's bf16 rows to every peer as one payload, theirs into place.
fn glued(a: std.mem.Allocator, w: *emu.World, r: usize, side: *emu.Stream, b: *Bufs, e: usize) !void {
    var x: plan.Exchange = .{ .unit = rowsOf(0) * D * 2, .shared = true };
    const mine = try fill(a, r, e, 0xff, rowsOf(r) * D * 2);
    for (0..3) |k| {
        b.dst[k] = if (k == r) &.{} else try a.alloc(u8, rowsOf(k) * D * 2);
        b.src[k] = mine;
        if (k != r) x.parts[k] = .{ .src = @intFromPtr(mine.ptr), .out = mine.len, .dst = @intFromPtr(b.dst[k].ptr), .in = b.dst[k].len };
    }
    try plan.run(w.proxies[r], side, &w.l, r, 1, &x);
}

fn expectExact(o: Opts) !Stats {
    const s = try run(o);
    if (s.hung or s.bad_bytes != 0) {
        std.debug.print("PRX schedule {t} zc {} slots {d}: hung {}, {d} bad bytes\n", .{ o.mode, o.zc, o.slots, s.hung, s.bad_bytes });
        return error.TestUnexpectedResult;
    }
    return s;
}

test "PRX schedule: pieced and paired chunks, copied and read in place, every byte exact at two and four slots" {
    for ([_]Mode{ .pieced, .pair }) |m| for ([_]bool{ false, true }) |zc| for ([_]usize{ 2, 4 }) |slots| {
        const s = try expectExact(.{ .mode = m, .zc = zc, .slots = slots });
        if (zc) try std.testing.expect(s.zc_calls > 0) else try std.testing.expectEqual(@as(usize, 0), s.zc_calls);
    };
    _ = try expectExact(.{ .mode = .pieced, .zc = true, .slots = 4, .own_later = true });
}

test "PRX schedule: the slots in use at once (pair: two, pieced read in place: pieces + 1), measured at eight" {
    for ([_]Mode{ .pieced, .pair }) |m| for ([_]bool{ false, true }) |zc| {
        const most = try expectExact(.{ .mode = m, .zc = zc, .slots = 8 });
        const two = try expectExact(.{ .mode = m, .zc = zc, .slots = 2 });
        const four = try expectExact(.{ .mode = m, .zc = zc, .slots = 4 });
        std.debug.print("PRX schedule {t} zc {}: at most {d} calls in a peer's slots (8 slots); credit stalls at 2 slots {d}, at 4 {d}; slot waits at 2 {d}, at 4 {d}\n", .{ m, zc, most.in_use_max, two.credit_stalls, four.credit_stalls, two.slot_waits, four.slot_waits });
        const bound: u64 = if (zc and m == .pieced) 3 + 1 else 2;
        try std.testing.expect(most.in_use_max <= bound);
        if (bound <= 4) try std.testing.expectEqual(@as(u64, 0), four.credit_stalls);
    };
}
