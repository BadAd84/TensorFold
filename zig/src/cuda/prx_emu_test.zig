//! TF_GLM_OVN_PRX on the host (prx_emu): random exchanges' bytes land as NCCL's, and the channel fails stop.

const std = @import("std");
const plan = @import("prx_plan.zig");
const emu = @import("prx_emu.zig");
const pattern = emu.pattern;

/// The exchanges every rank runs, alike: sizes[e][from][to] bytes (a shared exchange: alike for every `to`).
const Script = struct {
    shared: []bool,
    sizes: [][plan.max_world][plan.max_world]usize,

    fn unit(s: *const Script, e: usize, world: usize) usize {
        var u: usize = 0;
        for (0..world) |a| for (0..world) |b| if (a != b) {
            u = @max(u, s.sizes[e][a][b]);
        };
        return u;
    }
};

const guard = 64;

const Opts = struct {
    world: usize,
    slots: usize = 2,
    stripes: usize = 2,
    exchanges: usize = 40,
    streams: usize = 1,
    seed: u64 = 1,
    slow_gpu_rank: ?usize = null,
    slow_nic_rank: ?usize = null,
    fail: ?[2]u64 = null, // rank, call: that rank's NIC fails the call's first stripe
    freeze: ?[2]u64 = null, // rank, poll: that rank's proxy stops there (a hung process), until the others failed
    stall: ?usize = null, // that rank's GPU starts only once the others failed (its proxy alive, its calls never staged)
    dead: ?[2]u64 = null, // rank, call: from that call on, every write of that rank's NIC fails (a dead link)
    timeout_ns: u64 = 0,
    peer_ns: u64 = 0,
    exit_ns: u64 = 0,
    phases: usize = 1, // the exchanges posted and run in this many batches, every rank caught up between
    gap_ns: u64 = 0, // the idle time between batches

    pub fn adjust(o: *const Opts, r: usize, p: *emu.Proxy, n: *emu.Nic) void {
        p.timeout_ns = o.timeout_ns;
        p.peer_ns = o.peer_ns;
        p.exit_ns = o.exit_ns;
        p.exit = countExit;
        if (o.slow_nic_rank == r) n.slow = 7;
        if (o.fail) |f| if (f[0] == r) {
            n.fail_call = f[1];
        };
        if (o.freeze) |f| if (f[0] == r) {
            n.freeze_after = f[1];
        };
        if (o.dead) |f| if (f[0] == r) {
            n.dead_after = f[1];
        };
    }
};

var exits: std.atomic.Value(usize) = .init(0);
fn countExit() void {
    _ = exits.fetchAdd(1, .acq_rel);
}

const Why = enum { none, abort, silent, other };

const Outcome = struct {
    bad_bytes: usize = 0,
    bad_guards: usize = 0,
    hung: bool = false,
    failures: usize = 0,
    refused: usize = 0,
    aborted: usize = 0, // ranks failed by a peer's abort
    silent: usize = 0, // ranks failed by a peer's silent heartbeat
    why: [plan.max_world]Why = @splat(.none),
    silent_of: [plan.max_world]usize = @splat(0), // the ranks the silent-heartbeat failures name
    detect_ms: u64 = 0, // a frozen rank: from the streams' start until every other rank failed
    exits: usize = 0,
};

fn now() u64 {
    return @intCast(std.Io.Clock.awake.now(std.testing.io).toNanoseconds());
}

fn runWorld(o: Opts) !Outcome {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rnd = prng.random();
    const script: Script = .{ .shared = try a.alloc(bool, o.exchanges), .sizes = try a.alloc([plan.max_world][plan.max_world]usize, o.exchanges) };
    const lens = [_]usize{ 0, 4, 100, 4095, 4096, 4097, 5000, 8192, 9001, 12290 };
    for (0..o.exchanges) |e| {
        script.shared[e] = rnd.uintLessThan(u8, 4) == 0;
        for (0..o.world) |f| {
            const one = lens[rnd.uintLessThan(usize, lens.len)];
            for (0..o.world) |t| script.sizes[e][f][t] = if (script.shared[e]) one else lens[rnd.uintLessThan(usize, lens.len)];
        }
    }
    const w = try a.create(emu.World);
    w.* = .{ .l = try plan.Layout.of(o.world, o.slots, 4096), .world = o.world };
    exits.store(0, .release);
    try w.start(a, o.stripes, &o);
    var stopped = false;
    defer if (!stopped) w.stop();
    var src: [plan.max_world][][plan.max_world][]u8 = undefined;
    var dst: [plan.max_world][][plan.max_world][]u8 = undefined;
    for (0..o.world) |r| {
        src[r] = try a.alloc([plan.max_world][]u8, o.exchanges);
        dst[r] = try a.alloc([plan.max_world][]u8, o.exchanges);
        for (0..o.exchanges) |e| for (0..o.world) |p| if (p != r) {
            const key = if (script.shared[e]) 0xff else p;
            src[r][e][p] = try a.alloc(u8, script.sizes[e][r][p]);
            for (src[r][e][p], 0..) |*b, i| b.* = pattern(r, e, key, i);
            dst[r][e][p] = try a.alloc(u8, script.sizes[e][p][r] + 2 * guard);
            @memset(dst[r][e][p], 0xa5);
        };
    }
    var out: Outcome = .{};
    const t0 = now();
    for (0..o.phases) |ph| {
        if (ph > 0) std.Io.sleep(std.testing.io, .fromNanoseconds(o.gap_ns), .awake) catch {}; // every rank caught up
        var streams: [plan.max_world][2]emu.Stream = undefined;
        for (0..o.world) |r| for (&streams[r]) |*s| {
            s.* = .{ .gpa = a, .w = w, .rank = r, .pause_ns = if (o.slow_gpu_rank == r) 200_000 else 0 };
        };
        // the host: every rank posts every exchange in one order, each on its stream (alternating with two)
        for (0..o.world) |r| for (ph * o.exchanges / o.phases..(ph + 1) * o.exchanges / o.phases) |e| {
            var x: plan.Exchange = .{ .unit = script.unit(e, o.world), .shared = script.shared[e] };
            for (0..o.world) |p| if (p != r) {
                const s = if (script.shared[e]) src[r][e][plan.linkPeer(r, 0)] else src[r][e][p];
                x.parts[p] = .{ .src = @intFromPtr(s.ptr), .out = s.len, .dst = @intFromPtr(dst[r][e][p].ptr) + guard, .in = script.sizes[e][p][r] };
            };
            plan.run(w.proxies[r], &streams[r][e % o.streams], &w.l, r, o.stripes, &x) catch |err| switch (err) {
                error.ProxyFailed => break,
                else => return err,
            };
        };
        var threads: [plan.max_world][2]std.Thread = undefined;
        for (0..o.world) |r| for (0..o.streams) |k| if (o.stall != r) {
            threads[r][k] = try std.Thread.spawn(.{}, emu.Stream.execute, .{&streams[r][k]});
        };
        if (o.freeze orelse if (o.stall) |r| [2]u64{ r, 0 } else null) |f| { // the others fail while it is stopped
            while (now() - t0 < 10 * std.time.ns_per_s) {
                var all = true;
                for (0..o.world) |r| if (r != f[0] and w.proxies[r].failure() == null) {
                    all = false;
                };
                if (all) break;
                std.Thread.yield() catch {};
            }
            out.detect_ms = (now() - t0) / std.time.ns_per_ms;
            w.thawed.store(true, .release);
            if (o.stall) |r| for (0..o.streams) |k| { // then its GPU runs, released
                threads[r][k] = try std.Thread.spawn(.{}, emu.Stream.execute, .{&streams[r][k]});
            };
        }
        for (0..o.world) |r| for (0..o.streams) |k| {
            threads[r][k].join();
            out.hung = out.hung or streams[r][k].hung;
            if (streams[r][k].hung) {
                const h = streams[r][k].hung_at;
                std.debug.print("rank {d} stream {d} hung at offset {d} for {d} (word {d}); failure {?s}\n", .{ r, k, h[0], h[1], h[2], w.proxies[r].failure() });
            }
        };
    }
    if (o.exit_ns > 0) while (exits.load(.acquire) < o.world and now() - t0 < 10 * std.time.ns_per_s) std.Thread.yield() catch {};
    out.exits = exits.load(.acquire);
    for (0..o.world) |r| {
        const p = w.proxies[r];
        if (p.failure()) |m| {
            out.failures += 1;
            out.why[r] = if (std.mem.indexOf(u8, m, "abort") != null) .abort else if (std.mem.indexOf(u8, m, "no heartbeat") != null) .silent else .other;
            out.aborted += @intFromBool(out.why[r] == .abort);
            out.silent += @intFromBool(out.why[r] == .silent);
            if (std.mem.indexOf(u8, m, "no heartbeat from rank ")) |at| out.silent_of[m[at + 23] - '0'] += 1;
        }
        if (p.post(.{})) |_| {} else |_| out.refused += 1; // a failed channel takes no more calls
        for (0..o.exchanges) |e| for (0..o.world) |q| if (q != r) {
            const d = dst[r][e][q];
            const n = d.len - 2 * guard;
            const key = if (script.shared[e]) 0xff else r;
            for (d[guard .. guard + n], 0..) |b, i| out.bad_bytes += @intFromBool(b != pattern(q, e, key, i));
            for (d[0..guard]) |b| out.bad_guards += @intFromBool(b != 0xa5);
            for (d[guard + n ..]) |b| out.bad_guards += @intFromBool(b != 0xa5);
        };
    }
    w.stop();
    stopped = true;
    return out;
}

fn expectClean(o: Opts) !void {
    const got = try runWorld(o);
    if (got.hung or got.bad_bytes != 0 or got.bad_guards != 0 or got.failures != 0) {
        std.debug.print("PRX emulation world {d} slots {d} stripes {d} streams {d}: hung {}, {d} bad bytes, {d} bad guard bytes, {d} failed proxies\n", .{ o.world, o.slots, o.stripes, o.streams, got.hung, got.bad_bytes, got.bad_guards, got.failures });
        return error.TestUnexpectedResult;
    }
}

test "PRX emulation: every byte of every exchange lands where NCCL's send/receive puts it, worlds 2 to 4" {
    try expectClean(.{ .world = 2, .stripes = 1, .seed = 11 });
    try expectClean(.{ .world = 3, .stripes = 2, .seed = 12 });
    try expectClean(.{ .world = 3, .stripes = 1, .slots = 4, .seed = 13 });
    try expectClean(.{ .world = 4, .stripes = 2, .slots = 3, .seed = 14 });
}

test "PRX emulation: two streams sharing a channel with two slots, one rank's GPU or NIC slower, stay exact" {
    try expectClean(.{ .world = 3, .streams = 2, .slow_gpu_rank = 0, .exchanges = 24, .seed = 21 });
    try expectClean(.{ .world = 3, .streams = 2, .slow_nic_rank = 0, .exchanges = 24, .seed = 22 });
    try expectClean(.{ .world = 3, .streams = 1, .slow_gpu_rank = 2, .exchanges = 24, .seed = 23 });
}

test "PRX emulation: a failed write fails that rank and its abort the others at once, with no timeout, and nothing hangs" {
    const got = try runWorld(.{ .world = 3, .exchanges = 12, .fail = .{ 1, 5 }, .seed = 31 });
    try std.testing.expect(!got.hung);
    try std.testing.expectEqual(@as(usize, 3), got.failures);
    try std.testing.expectEqual(@as(usize, 2), got.aborted); // ranks 0 and 2 learnt of it from rank 1's abort
    try std.testing.expectEqual(@as(usize, 3), got.refused);
}

test "PRX emulation: a peer that stops while its NIC still completes fails every other rank by its silent heartbeat" {
    const got = try runWorld(.{ .world = 3, .exchanges = 16, .freeze = .{ 2, 40 }, .peer_ns = 300 * std.time.ns_per_ms, .seed = 41 });
    try std.testing.expect(!got.hung);
    try std.testing.expectEqual(@as(usize, 3), got.failures);
    try std.testing.expect(got.silent >= 1); // the first to notice; the other may learn of it by that one's abort
    try std.testing.expectEqual(got.silent, got.silent_of[2]); // and only the stopped rank is ever named
    for (got.why[0..2]) |y| try std.testing.expect(y == .silent or y == .abort);
    try std.testing.expect(got.detect_ms < 3000); // TF_GLM_OVN_PRX_PEER_S, not the 300 s timeout
}

test "PRX emulation: every failed rank exits after TF_GLM_OVN_PRX_EXIT_S, so the serve's watchdog restarts the ranks" {
    const got = try runWorld(.{ .world = 3, .exchanges = 12, .fail = .{ 0, 3 }, .exit_ns = 100 * std.time.ns_per_ms, .seed = 51 });
    try std.testing.expect(!got.hung);
    try std.testing.expectEqual(@as(usize, 3), got.failures);
    try std.testing.expectEqual(@as(usize, 3), got.exits);
}

test "PRX emulation: with the heartbeat and progress watched, a slow live world and an idle gap stay clean" {
    try expectClean(.{ .world = 3, .streams = 2, .slow_gpu_rank = 1, .exchanges = 24, .peer_ns = 100 * std.time.ns_per_ms, .seed = 61 });
    // caught up for longer than both limits between two batches: the watch restarts, nothing fails
    try expectClean(.{ .world = 3, .exchanges = 16, .phases = 2, .gap_ns = 400 * std.time.ns_per_ms, .timeout_ns = 200 * std.time.ns_per_ms, .peer_ns = 200 * std.time.ns_per_ms, .seed = 62 });
}

test "PRX emulation: a peer whose GPU stops while its proxy lives fails every rank by TF_GLM_OVN_PRX_TIMEOUT_S" {
    const got = try runWorld(.{ .world = 3, .exchanges = 12, .stall = 2, .timeout_ns = 300 * std.time.ns_per_ms, .peer_ns = 300 * std.time.ns_per_ms, .seed = 71 });
    try std.testing.expect(!got.hung);
    try std.testing.expectEqual(@as(usize, 3), got.failures);
    try std.testing.expect(got.detect_ms < 3000);
}

test "PRX emulation: a dead link whose abort cannot leave still fails the other end, by its writes into the halted rank" {
    const got = try runWorld(.{ .world = 2, .stripes = 1, .exchanges = 12, .dead = .{ 1, 4 }, .seed = 81 });
    try std.testing.expect(!got.hung);
    try std.testing.expectEqual(@as(usize, 2), got.failures);
}
