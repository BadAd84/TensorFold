//! TF_GLM_OVN_PRX's emulated ranks for host tests: real proxies, NICs that write late, halt or freeze, streams.

const std = @import("std");
const plan = @import("prx_plan.zig");
const proxy_mod = @import("prx_proxy.zig");

const max_q = 512;
pub const max_events = 4096;

/// A queued work request on one queue pair.
const Wr = struct { to_rank: usize, from: usize, to: usize, n: usize, flag_at: ?usize, value: u32, id: u64 };

pub const World = struct {
    regions: [plan.max_world][]align(64) u8 = undefined,
    nics: [plan.max_world]Nic = undefined,
    proxies: [plan.max_world]*Proxy = undefined,
    threads: [plan.max_world]std.Thread = undefined,
    l: plan.Layout,
    world: usize,
    thawed: std.atomic.Value(bool) = .init(false),
    events: [plan.max_world][max_events]std.atomic.Value(bool) = @splat(@splat(.init(false))),

    /// Regions, NICs and proxies of every rank (`set` adjusts each proxy and NIC), the proxies running.
    pub fn start(w: *World, a: std.mem.Allocator, stripes: usize, set: anytype) !void {
        for (0..w.world) |r| {
            w.regions[r] = try a.alignedAlloc(u8, .@"64", w.l.total);
            @memset(w.regions[r], 0);
            w.nics[r] = .{ .w = w, .rank = r };
            w.proxies[r] = try a.create(Proxy);
            w.proxies[r].* = .{ .nic = &w.nics[r], .region = w.regions[r].ptr, .l = w.l, .rank = r, .stripes = stripes, .timeout_ns = 0 };
            set.adjust(r, w.proxies[r], &w.nics[r]);
        }
        for (0..w.world) |r| w.threads[r] = try std.Thread.spawn(.{}, Proxy.loop, .{ w.proxies[r], std.testing.io });
    }

    pub fn stop(w: *World) void {
        w.thawed.store(true, .release);
        for (0..w.world) |r| w.proxies[r].running.store(false, .release);
        for (0..w.world) |r| w.threads[r].join();
    }
};

pub const Nic = struct {
    w: *World,
    rank: usize,
    q: [plan.max_peers][plan.max_stripes][max_q]Wr = undefined,
    head: [plan.max_peers][plan.max_stripes]usize = @splat(@splat(0)),
    tail: [plan.max_peers][plan.max_stripes]usize = @splat(@splat(0)),
    polls: u64 = 0,
    lag: u64 = 3, // the second stripe moves one write every `lag` polls
    slow: u64 = 1, // this rank's NIC moves anything every `slow` polls
    fail_call: ?u64 = null, // the completion of this call's first stripe reports an error
    freeze_after: ?u64 = null, // a hung process: from this poll its proxy stops until the world thaws
    dead_after: ?u64 = null, // a dead link: from this call's first write on, every write of this NIC fails unwritten
    dead: bool = false,
    halted: std.atomic.Value(bool) = .init(false),

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
    pub fn halt(n: *Nic) void {
        n.halted.store(true, .release);
    }
    pub fn poll(n: *Nic, out: []proxy_mod.Done) !usize {
        n.polls += 1;
        if (n.freeze_after) |f| if (n.polls >= f) while (!n.w.thawed.load(.acquire)) std.Thread.yield() catch {};
        var k: usize = 0;
        if (n.halted.load(.acquire) or n.polls % n.slow != 0) return 0;
        for (0..n.w.world - 1) |link| for (0..plan.max_stripes) |i| {
            if (n.head[link][i] == n.tail[link][i] or k == out.len) continue;
            if (i == 1 and n.polls % n.lag != 0) continue;
            const wr = n.q[link][i][n.head[link][i] % max_q];
            n.head[link][i] += 1;
            if (n.dead_after) |d| if (wr.id & 0xf == 1 and wr.id >> 11 >= d) {
                n.dead = true;
            };
            const refused = n.dead or n.w.nics[wr.to_rank].halted.load(.acquire); // or its queue pair is in error
            const bad = refused or (n.fail_call != null and wr.id & 0xf == 1 and wr.id >> 11 == n.fail_call.? and i == 0);
            if (!refused) {
                const dst = n.w.regions[wr.to_rank];
                if (wr.n > 0) @memcpy(dst[wr.to .. wr.to + wr.n], n.w.regions[n.rank][wr.from .. wr.from + wr.n]);
                @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(dst.ptr + (wr.flag_at orelse wr.to)))), wr.value, .release);
            }
            out[k] = .{ .id = wr.id, .ok = !bad, .status = if (bad) 12 else 0 };
            k += 1;
        };
        if (k == 0) std.Thread.yield() catch {};
        return k;
    }
};

pub const Proxy = proxy_mod.Proxy(Nic);

/// A stream's recorded GPU work, run later in order; `signal`/`await_`: events between one rank's streams.
pub const Op = union(enum) { wait: [2]u64, write: [2]u64, stage: [3]u64, unstage: [3]u64, signal: usize, await_: usize };

pub const Stream = struct {
    ops: std.ArrayList(Op) = .empty,
    gpa: std.mem.Allocator,
    w: *World,
    rank: usize,
    pause_ns: u64 = 0, // a slow GPU: this long after every operation
    hung: bool = false,
    hung_at: [3]u64 = @splat(0), // the wait that never passed: its offset, its value and the word's last value
    slot_waits: u64 = 0, // send_done waits that had to wait: a slot restaged before the NICs had read it out

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
    pub fn signal(s: *Stream, e: usize) !void {
        try s.ops.append(s.gpa, .{ .signal = e });
    }
    pub fn await_(s: *Stream, e: usize) !void {
        try s.ops.append(s.gpa, .{ .await_ = e });
    }

    fn until(s: *Stream, w: *const u32, v: u32, at: usize) bool {
        const io = std.testing.io;
        const give_up = std.Io.Clock.awake.now(io).toNanoseconds() + 10 * std.time.ns_per_s;
        var spins: u64 = 0;
        while (!plan.geq(@atomicLoad(u32, w, .acquire), v)) : (spins += 1) {
            if (spins % 64 == 0 and std.Io.Clock.awake.now(io).toNanoseconds() > give_up) {
                s.hung = true; // a wait nothing satisfies: the GPU would hang here
                s.hung_at = .{ at, v, @atomicLoad(u32, w, .acquire) };
                return false;
            }
            std.Thread.yield() catch {};
        }
        return true;
    }

    pub fn execute(s: *Stream) void {
        const r = s.w.regions[s.rank];
        for (s.ops.items) |o| {
            switch (o) {
                .wait => |a| {
                    const w: *u32 = @ptrCast(@alignCast(r.ptr + a[0]));
                    const slot_word = a[0] >= s.w.l.sendDone(0) and a[0] <= s.w.l.sendDone(s.w.l.slots - 1);
                    if (slot_word and !plan.geq(@atomicLoad(u32, w, .acquire), @truncate(a[1]))) s.slot_waits += 1;
                    if (!s.until(w, @truncate(a[1]), a[0])) return;
                },
                .write => |a| @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(r.ptr + a[0]))), @truncate(a[1]), .release),
                .stage => |a| @memcpy(r[a[0] .. a[0] + a[2]], @as([*]const u8, @ptrFromInt(a[1]))[0..a[2]]),
                .unstage => |a| @memcpy(@as([*]u8, @ptrFromInt(a[0]))[0..a[2]], r[a[1] .. a[1] + a[2]]),
                .signal => |e| s.w.events[s.rank][e].store(true, .release),
                .await_ => |e| {
                    const io = std.testing.io;
                    const give_up = std.Io.Clock.awake.now(io).toNanoseconds() + 10 * std.time.ns_per_s;
                    var spins: u64 = 0;
                    while (!s.w.events[s.rank][e].load(.acquire)) : (spins += 1) {
                        if (spins % 64 == 0 and std.Io.Clock.awake.now(io).toNanoseconds() > give_up) {
                            s.hung = true;
                            s.hung_at = .{ 1 << 40 | e, 0, 0 };
                            return;
                        }
                        std.Thread.yield() catch {};
                    }
                },
            }
            if (s.pause_ns > 0) std.Io.sleep(std.testing.io, .fromNanoseconds(s.pause_ns), .awake) catch {};
        }
    }
};

/// Byte i of what rank `from` sends rank `to` (0xff: a shared payload; 0xfe: its own block) in exchange e.
pub fn pattern(from: usize, e: usize, to: usize, i: usize) u8 {
    const h = std.hash.Wyhash.hash(from << 40 | e << 20 | to, std.mem.asBytes(&i));
    return @truncate(h);
}
