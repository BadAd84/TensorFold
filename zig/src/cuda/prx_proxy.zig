//! TF_GLM_OVN_PRX's proxy: posts each call's blocks once staged and credited; fails stop with every wait released.

const std = @import("std");
const plan = @import("prx_plan.zig");
const nap = @import("roce_nap.zig");

/// A completed work request: its id, and whether it succeeded (status: the NIC's code otherwise).
pub const Done = struct { id: u64, ok: bool = true, status: u32 = 0 };

pub const depth = 256; // sends a queue pair holds
const kind_send = 1;
const kind_credit = 2;
const idle_spins = 1 << 20; // polls with nothing outstanding before the proxy naps between them
const clock_every = 64; // idle polls between reads of the clock (a poll can yield a scheduler quantum)

/// wr_id: kind (4 bits), link (2), stripe (2), whether the flag also covers a data write (bit 10), the call (11 up).
fn wrId(kind: u64, link: usize, stripe: usize, call: u64) u64 {
    return call << 11 | @as(u64, stripe) << 6 | @as(u64, link) << 4 | kind;
}

/// The proxy over `Nic` (`data`: a write then its flag, one completion; `word`; `poll`), offsets in the regions.
pub fn Proxy(comptime Nic: type) type {
    return struct {
        const P = @This();

        nic: *Nic,
        region: [*]u8,
        l: plan.Layout,
        rank: usize,
        stripes: usize,
        timeout_ns: u64, // 0: no timeout
        desc: [plan.qcap]plan.Desc = undefined,
        pending: [plan.qcap]u32 = undefined, // a posted call's stripes not completed yet
        posted: std.atomic.Value(u64) = .init(0),
        next_send: u64 = 0,
        send_mask: u32 = 0, // the links the head call went out on so far
        send_done: std.atomic.Value(u64) = .init(0),
        freed: u64 = 0,
        reported: u64 = 0,
        outstanding: [plan.max_peers][plan.max_stripes]u32 = @splat(@splat(0)),
        running: std.atomic.Value(bool) = .init(true),
        failed: std.atomic.Value(bool) = .init(false),
        err: [256]u8 = @splat(0),
        sent: u64 = 0, // bytes written to the peers

        fn word(p: *const P, at: usize) *u32 {
            return @ptrCast(@alignCast(p.region + at));
        }
        fn load(p: *const P, at: usize) u32 {
            return @atomicLoad(u32, p.word(at), .acquire);
        }
        fn store(p: *const P, at: usize, v: u32) void {
            @atomicStore(u32, p.word(at), v, .release);
        }

        fn fail(p: *P, comptime fmt: []const u8, args: anytype) void {
            if (p.failed.load(.acquire)) return;
            _ = std.fmt.bufPrint(p.err[0 .. p.err.len - 1], fmt, args) catch {};
            p.failed.store(true, .release);
        }

        /// The host's call: queue `d` and return its number (every rank numbers alike); refused after a failure.
        pub fn post(p: *P, d: plan.Desc) !u64 {
            const c = p.posted.raw; // the host thread alone posts
            while (c - p.send_done.load(.acquire) >= plan.qcap) {
                if (p.failed.load(.acquire)) return error.ProxyFailed;
                std.Thread.yield() catch {};
            }
            if (p.failed.load(.acquire)) return error.ProxyFailed;
            p.desc[c % plan.qcap] = d;
            p.posted.store(c + 1, .release);
            return c;
        }

        fn peers(p: *const P) usize {
            return p.l.world - 1;
        }

        fn creditOk(p: *const P, link: usize, call: u64) bool {
            if (call < p.l.slots) return true;
            return plan.geq(p.load(p.l.credit(plan.linkPeer(p.rank, link))), @truncate(call - p.l.slots + 1));
        }

        /// The head calls' links, each as soon as its payload is staged and the peer freed the slot.
        fn stepSends(p: *P) !bool {
            const posted = p.posted.load(.acquire);
            const all: u32 = (@as(u32, 1) << @intCast(p.peers())) - 1;
            var did = false;
            while (p.next_send < posted) {
                const call = p.next_send;
                const s: usize = @intCast(call % p.l.slots);
                const v: u32 = @truncate(call + 1);
                const d = p.desc[call % plan.qcap];
                if (p.send_mask == 0) p.pending[call % plan.qcap] = 0;
                for (0..p.peers()) |link| {
                    const blk = if (d.shared) 0 else link;
                    if (p.send_mask >> @intCast(link) & 1 == 1 or !p.creditOk(link, call) or !plan.geq(p.load(p.l.ready(s, blk)), v)) continue;
                    try p.postPart(link, p.l.send(s, blk), p.l.recv(s, p.rank), d.bytes[link], s, v, call);
                    p.send_mask |= @as(u32, 1) << @intCast(link);
                    did = true;
                }
                if (p.send_mask != all) break;
                p.send_mask = 0;
                p.next_send += 1;
            }
            return did;
        }

        /// One block on `link`, striped in 16-byte cuts over its queue pairs, each stripe's flag after its data.
        fn postPart(p: *P, link: usize, from: usize, to: usize, n: usize, s: usize, v: u32, call: u64) !void {
            var lo: usize = 0;
            for (0..p.stripes) |i| {
                const hi = if (i + 1 == p.stripes) n else @min(n, std.mem.alignForward(usize, n * (i + 1) / p.stripes, 16));
                try p.room(link, i, 2);
                const two: u64 = @intFromBool(hi > lo);
                try p.nic.data(link, i, from + lo, to + lo, hi - lo, p.l.flag(s, p.rank, i), v, wrId(kind_send, link, i, call) | two << 10);
                p.outstanding[link][i] += @intCast(1 + two);
                p.pending[call % plan.qcap] += 1;
                p.sent += hi - lo;
                lo = hi;
            }
        }

        fn room(p: *P, link: usize, i: usize, need: u32) !void {
            while (p.outstanding[link][i] + need > depth - 4) {
                _ = try p.drain();
                if (!p.running.load(.monotonic)) return error.ProxyFailed;
            }
        }

        fn drain(p: *P) !usize {
            var buf: [64]Done = undefined;
            const n = p.nic.poll(&buf) catch |e| {
                p.fail("polling the completions failed ({t})", .{e});
                return error.ProxyFailed;
            };
            for (buf[0..n]) |w| {
                const link: usize = @intCast(w.id >> 4 & 3);
                const i: usize = @intCast(w.id >> 6 & 3);
                const call = w.id >> 11;
                if (!w.ok) {
                    p.fail("an RDMA write to rank {d} (kind {d}, stripe {d}, call {d}) failed: status {d}", .{ plan.linkPeer(p.rank, link), w.id & 0xf, i, call, w.status });
                    return error.ProxyFailed;
                }
                p.outstanding[link][i] -= @intCast(1 + (w.id >> 10 & 1));
                if (w.id & 0xf == kind_send) p.pending[call % plan.qcap] -= 1;
            }
            return n;
        }

        /// Calls whose sends completed (send_done) and calls this rank's GPU freed (to every peer as its credit).
        fn stepDone(p: *P) !bool {
            var did = false;
            var sd = p.send_done.raw;
            while (sd < p.next_send and p.pending[sd % plan.qcap] == 0) : (sd += 1) {
                p.store(p.l.sendDone(@intCast(sd % p.l.slots)), @truncate(sd + 1));
                did = true;
            }
            p.send_done.store(sd, .release);
            const posted = p.posted.load(.acquire);
            while (p.freed < posted and plan.geq(p.load(p.l.consumed(@intCast(p.freed % p.l.slots))), @truncate(p.freed + 1))) {
                p.freed += 1;
                did = true;
            }
            if (p.freed != p.reported) {
                for (0..p.peers()) |link| {
                    try p.room(link, 0, 1);
                    try p.nic.word(link, 0, p.l.credit(p.rank), @truncate(p.freed), wrId(kind_credit, link, 0, 0));
                    p.outstanding[link][0] += 1;
                }
                p.reported = p.freed;
            }
            return did;
        }

        /// One pass: sends, completions, done and freed calls. Whether anything moved.
        pub fn step(p: *P) !bool {
            var did = try p.stepSends();
            if (try p.drain() > 0) did = true;
            if (try p.stepDone()) did = true;
            return did;
        }

        /// Whether calls are outstanding: posted but not sent, or not yet freed by this rank's GPU.
        pub fn busy(p: *const P) bool {
            const posted = p.posted.load(.acquire);
            return p.next_send < posted or p.freed < posted;
        }

        /// Every word the GPU can wait on (send_done, flags) past its pending waits (cyclic compare, posted + 2^30).
        pub fn release(p: *P) void {
            const past: u32 = @truncate(p.posted.load(.acquire) +% (1 << 30));
            for (0..p.l.slots) |s| {
                p.store(p.l.sendDone(s), past);
                for (0..plan.max_world) |r| for (0..plan.max_stripes) |i| p.store(p.l.flag(s, r, i), past);
            }
        }

        /// The thread: steps until stopped; after a failure it keeps re-releasing the GPU's words (late flags).
        pub fn loop(p: *P, io: std.Io) void {
            var idle: u64 = 0;
            var last = std.Io.Clock.awake.now(io).toNanoseconds();
            while (p.running.load(.monotonic)) {
                if (p.failed.load(.acquire)) {
                    p.release();
                    nap.pause();
                    continue;
                }
                const did = p.step() catch |e| {
                    p.fail("the proxy stopped: {t}", .{e});
                    continue;
                };
                if (did) {
                    idle = 0;
                    last = std.Io.Clock.awake.now(io).toNanoseconds();
                    continue;
                }
                idle += 1;
                const outstanding = p.busy();
                if (outstanding and p.timeout_ns > 0 and idle % clock_every == 0) {
                    const now = std.Io.Clock.awake.now(io).toNanoseconds();
                    if (now - last > p.timeout_ns) p.fail("no progress for {d} ms with calls outstanding (call {d} of {d} sent, {d} freed)", .{ @divTrunc(now - last, std.time.ns_per_ms), p.next_send, p.posted.load(.acquire), p.freed });
                }
                if (!outstanding and idle >= idle_spins) nap.pause();
            }
        }

        /// The failure's message, or null.
        pub fn failure(p: *const P) ?[]const u8 {
            if (!p.failed.load(.acquire)) return null;
            return std.mem.sliceTo(&p.err, 0);
        }
    };
}
