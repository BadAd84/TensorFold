//! TF_GLM_OVN_PRX on the host: real plans and proxies, emulated NICs and streams; bytes must land as NCCL's.

const std = @import("std");
const plan = @import("prx_plan.zig");
const proxy_mod = @import("prx_proxy.zig");

const max_q = 512;

/// A queued work request on one queue pair.
const Wr = struct { to_rank: usize, from: usize, to: usize, n: usize, flag_at: ?usize, value: u32, id: u64 };

const World = struct {
    regions: [plan.max_world][]align(64) u8 = undefined,
    nics: [plan.max_world]Nic = undefined,
    proxies: [plan.max_world]*Proxy = undefined,
    l: plan.Layout,
    world: usize,
};

const Nic = struct {
    w: *World,
    rank: usize,
    q: [plan.max_peers][plan.max_stripes][max_q]Wr = undefined,
    head: [plan.max_peers][plan.max_stripes]usize = @splat(@splat(0)),
    tail: [plan.max_peers][plan.max_stripes]usize = @splat(@splat(0)),
    polls: u64 = 0,
    lag: u64 = 3, // the second stripe moves one write every `lag` polls
    slow: u64 = 1, // this rank's NIC moves anything every `slow` polls
    fail_call: ?u64 = null, // the completion of this call's first stripe reports an error

    fn push(n: *Nic, link: usize, i: usize, wr: Wr) !void {
        if (n.tail[link][i] - n.head[link][i] == max_q) return error.QueueFull;
        n.q[link][i][n.tail[link][i] % max_q] = wr;
        n.tail[link][i] += 1;
    }
    pub fn data(n: *Nic, link: usize, i: usize, from: usize, to: usize, len: usize, flag_at: usize, v: u32, id: u64) !void {
        try n.push(link, i, .{ .to_rank = plan.linkPeer(n.rank, link), .from = from, .to = to, .n = len, .flag_at = flag_at, .value = v, .id = id });
    }
    pub fn word(n: *Nic, link: usize, i: usize, at: usize, v: u32, id: u64) !void {
        try n.push(link, i, .{ .to_rank = plan.linkPeer(n.rank, link), .from = 0, .to = at, .n = 0, .flag_at = null, .value = v, .id = id });
    }
    pub fn poll(n: *Nic, out: []proxy_mod.Done) !usize {
        n.polls += 1;
        var k: usize = 0;
        if (n.polls % n.slow != 0) return 0;
        for (0..n.w.world - 1) |link| for (0..plan.max_stripes) |i| {
            if (n.head[link][i] == n.tail[link][i] or k == out.len) continue;
            if (i == 1 and n.polls % n.lag != 0) continue;
            const wr = n.q[link][i][n.head[link][i] % max_q];
            n.head[link][i] += 1;
            const dst = n.w.regions[wr.to_rank];
            if (wr.n > 0) @memcpy(dst[wr.to .. wr.to + wr.n], n.w.regions[n.rank][wr.from .. wr.from + wr.n]);
            const at = wr.flag_at orelse wr.to;
            @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(dst.ptr + at))), wr.value, .release);
            const bad = n.fail_call != null and wr.id & 0xf == 1 and wr.id >> 11 == n.fail_call.? and i == 0;
            out[k] = .{ .id = wr.id, .ok = !bad, .status = if (bad) 12 else 0 };
            k += 1;
        };
        if (k == 0) std.Thread.yield() catch {};
        return k;
    }
};

const Proxy = proxy_mod.Proxy(Nic);

/// A stream's recorded GPU work, executed later in order (the host posts every call before the GPU reaches it).
const Op = union(enum) { wait: [2]u64, write: [2]u64, stage: [3]u64, unstage: [3]u64 };

const Stream = struct {
    ops: std.ArrayList(Op) = .empty,
    gpa: std.mem.Allocator,
    region: []align(64) u8 = undefined,
    pause_ns: u64 = 0, // a slow GPU: this long after every operation
    hung: bool = false,
    hung_at: [3]u64 = @splat(0), // the wait that never passed: its offset, its value and the word's last value

    pub fn wait(s: *Stream, at: usize, v: u32) !void {
        try s.ops.append(s.gpa, .{ .wait = .{ at, v } });
    }
    pub fn write(s: *Stream, at: usize, v: u32) !void {
        try s.ops.append(s.gpa, .{ .write = .{ at, v } });
    }
    pub fn stage(s: *Stream, at: usize, src: u64, n: usize) !void {
        try s.ops.append(s.gpa, .{ .stage = .{ at, src, n } });
    }
    pub fn unstage(s: *Stream, dst: u64, at: usize, n: usize) !void {
        try s.ops.append(s.gpa, .{ .unstage = .{ dst, at, n } });
    }

    fn execute(s: *Stream) void {
        const r = s.region;
        for (s.ops.items) |o| {
            switch (o) {
                .wait => |a| {
                    const w: *u32 = @ptrCast(@alignCast(r.ptr + a[0]));
                    const io = std.testing.io;
                    const give_up = std.Io.Clock.awake.now(io).toNanoseconds() + 10 * std.time.ns_per_s;
                    var spins: u64 = 0;
                    while (!plan.geq(@atomicLoad(u32, w, .acquire), @truncate(a[1]))) : (spins += 1) {
                        if (spins % 64 == 0 and std.Io.Clock.awake.now(io).toNanoseconds() > give_up) {
                            s.hung = true; // a wait nothing satisfies: the GPU would hang here
                            s.hung_at = .{ a[0], a[1], @atomicLoad(u32, w, .acquire) };
                            return;
                        }
                        std.Thread.yield() catch {};
                    }
                },
                .write => |a| @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(r.ptr + a[0]))), @truncate(a[1]), .release),
                .stage => |a| @memcpy(r[a[0] .. a[0] + a[2]], @as([*]const u8, @ptrFromInt(a[1]))[0..a[2]]),
                .unstage => |a| @memcpy(@as([*]u8, @ptrFromInt(a[0]))[0..a[2]], r[a[1] .. a[1] + a[2]]),
            }
            if (s.pause_ns > 0) std.Io.sleep(std.testing.io, .fromNanoseconds(s.pause_ns), .awake) catch {};
        }
    }
};

/// The exchanges every rank runs, alike: sizes[e][from][to] bytes (a shared exchange: alike for every `to`).
const Script = struct {
    n: usize,
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

fn pattern(from: usize, e: usize, to: usize, i: usize) u8 {
    const h = std.hash.Wyhash.hash(from << 40 | e << 20 | to, std.mem.asBytes(&i));
    return @truncate(h);
}

const guard = 64;

const Bufs = struct {
    src: [][plan.max_world][]u8, // [e][peer] (shared: [e][0])
    dst: [][plan.max_world][]u8, // [e][peer], guard bytes either side
};

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
    timeout_ns: u64 = 0,
};

const Outcome = struct { bad_bytes: usize = 0, bad_guards: usize = 0, hung: bool = false, failures: usize = 0, refused: usize = 0 };

fn runWorld(o: Opts) !Outcome {
    const gpa = std.heap.page_allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const M = 4096;
    var w: World = .{ .l = try plan.Layout.of(o.world, o.slots, M), .world = o.world };
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rnd = prng.random();
    const script: Script = .{ .n = o.exchanges, .shared = try a.alloc(bool, o.exchanges), .sizes = try a.alloc([plan.max_world][plan.max_world]usize, o.exchanges) };
    const lens = [_]usize{ 0, 4, 100, 4095, 4096, 4097, 5000, 8192, 9001, 12290 };
    for (0..o.exchanges) |e| {
        script.shared[e] = rnd.uintLessThan(u8, 4) == 0;
        for (0..o.world) |f| {
            const one = lens[rnd.uintLessThan(usize, lens.len)];
            for (0..o.world) |t| script.sizes[e][f][t] = if (script.shared[e]) one else lens[rnd.uintLessThan(usize, lens.len)];
        }
    }
    var bufs: [plan.max_world]Bufs = undefined;
    var streams: [plan.max_world][2]Stream = undefined;
    for (0..o.world) |r| {
        w.regions[r] = try a.alignedAlloc(u8, .@"64", w.l.total);
        @memset(w.regions[r], 0);
        w.nics[r] = .{ .w = &w, .rank = r, .slow = if (o.slow_nic_rank == r) 7 else 1 };
        if (o.fail) |f| if (f[0] == r) {
            w.nics[r].fail_call = f[1];
        };
        w.proxies[r] = try a.create(Proxy);
        w.proxies[r].* = .{ .nic = &w.nics[r], .region = w.regions[r].ptr, .l = w.l, .rank = r, .stripes = o.stripes, .timeout_ns = o.timeout_ns };
        for (&streams[r]) |*s| s.* = .{ .gpa = a, .region = w.regions[r], .pause_ns = if (o.slow_gpu_rank == r) 200_000 else 0 };
        bufs[r] = .{ .src = try a.alloc([plan.max_world][]u8, o.exchanges), .dst = try a.alloc([plan.max_world][]u8, o.exchanges) };
        for (0..o.exchanges) |e| for (0..o.world) |p| if (p != r) {
            const out = script.sizes[e][r][p];
            const key = if (script.shared[e]) 0xff else p;
            bufs[r].src[e][p] = try a.alloc(u8, out);
            for (bufs[r].src[e][p], 0..) |*b, i| b.* = pattern(r, e, key, i);
            bufs[r].dst[e][p] = try a.alloc(u8, script.sizes[e][p][r] + 2 * guard);
            @memset(bufs[r].dst[e][p], 0xa5);
        };
    }
    var proxy_threads: [plan.max_world]std.Thread = undefined;
    for (0..o.world) |r| proxy_threads[r] = try std.Thread.spawn(.{}, Proxy.loop, .{ w.proxies[r], std.testing.io });
    defer for (0..o.world) |r| {
        w.proxies[r].running.store(false, .release);
        proxy_threads[r].join();
    };
    // the host: every rank posts every exchange in one order, each on its stream (alternating with two)
    for (0..o.world) |r| for (0..o.exchanges) |e| {
        var x: plan.Exchange = .{ .unit = script.unit(e, o.world), .shared = script.shared[e] };
        for (0..o.world) |p| if (p != r) {
            const shared_src = bufs[r].src[e][plan.linkPeer(r, 0)];
            const src = if (script.shared[e]) shared_src else bufs[r].src[e][p];
            x.parts[p] = .{ .src = @intFromPtr(src.ptr), .out = src.len, .dst = @intFromPtr(bufs[r].dst[e][p].ptr) + guard, .in = script.sizes[e][p][r] };
        };
        const s = &streams[r][e % o.streams];
        plan.run(w.proxies[r], s, &w.l, r, o.stripes, &x) catch |err| switch (err) {
            error.ProxyFailed => break,
            else => return err,
        };
    };
    var threads: [plan.max_world][2]std.Thread = undefined;
    for (0..o.world) |r| for (0..o.streams) |k| {
        threads[r][k] = try std.Thread.spawn(.{}, Stream.execute, .{&streams[r][k]});
    };
    var out: Outcome = .{};
    for (0..o.world) |r| for (0..o.streams) |k| {
        threads[r][k].join();
        out.hung = out.hung or streams[r][k].hung;
        if (streams[r][k].hung) {
            const p = w.proxies[r];
            const h = streams[r][k].hung_at;
            std.debug.print("rank {d} stream {d} hung at offset {d} for {d} (word {d}); proxy: {d} posted, {d} sent, {d} freed, failure {?s}\n", .{ r, k, h[0], h[1], h[2], p.posted.load(.acquire), p.next_send, p.freed, p.failure() });
        }
    };
    for (0..o.world) |r| {
        if (w.proxies[r].failure() != null) out.failures += 1;
        if (w.proxies[r].post(.{})) |_| {} else |_| out.refused += 1; // a failed channel takes no more calls
        for (0..o.exchanges) |e| for (0..o.world) |p| if (p != r) {
            const d = bufs[r].dst[e][p];
            const n = d.len - 2 * guard;
            const key = if (script.shared[e]) 0xff else r;
            for (d[guard .. guard + n], 0..) |b, i| out.bad_bytes += @intFromBool(b != pattern(p, e, key, i));
            for (d[0..guard]) |b| out.bad_guards += @intFromBool(b != 0xa5);
            for (d[guard + n ..]) |b| out.bad_guards += @intFromBool(b != 0xa5);
        };
    }
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

test "PRX emulation: a failed write fails that rank, the others by the timeout, and no GPU wait hangs" {
    const got = try runWorld(.{ .world = 3, .exchanges = 12, .fail = .{ 1, 5 }, .timeout_ns = 300 * std.time.ns_per_ms, .seed = 31 });
    try std.testing.expect(!got.hung);
    try std.testing.expectEqual(@as(usize, 3), got.failures);
    try std.testing.expectEqual(@as(usize, 3), got.refused);
}
