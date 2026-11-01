//! TF_GLM_OVN_PRX's proxy: posts each call's blocks once staged and credited; fails stop with every wait released.

const std = @import("std");
const plan = @import("prx_plan.zig");
const nap = @import("roce_nap.zig");

/// A completed work request: its id, and whether it succeeded (status: the NIC's code otherwise).
pub const Done = struct { id: u64, ok: bool = true, status: u32 = 0 };

pub const depth = 256; // sends a queue pair holds
const kind_send = 1;
const kind_credit = 2;
const kind_beat = 3;
const kind_abort = 4;
const idle_spins = 1 << 20; // polls with nothing outstanding before the proxy naps between them
const beat_most_ns = 500 * std.time.ns_per_ms; // the heartbeat's period, or a quarter of the peer limit if shorter
const abort_wait_ns = 200 * std.time.ns_per_ms; // how long a failed proxy waits for its aborts to leave

fn now(io: std.Io) u64 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

/// The default after TF_GLM_OVN_PRX_EXIT_S: the ranks cannot agree any more, so the serve's watchdog restarts them.
fn exit70() void {
    std.c._exit(70);
}

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
        timeout_ns: u64, // TF_GLM_OVN_PRX_TIMEOUT_S: no progress this long with calls outstanding fails (0: never)
        peer_ns: u64 = 0, // TF_GLM_OVN_PRX_PEER_S: a peer's heartbeat still this long with calls outstanding fails
        exit_ns: u64 = 0, // TF_GLM_OVN_PRX_EXIT_S: the process exits 70 this long after a failure (0: it stays)
        exit: *const fn () void = exit70,
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
        origin: u32 = 0, // the rank where a failure started, plus one (the abort's value)
        beats: u32 = 0,
        beat_out: [plan.max_peers]bool = @splat(false), // a heartbeat to that link not completed yet
        seen: [plan.max_peers]u32 = @splat(0), // each peer's last heartbeat, and when it changed
        seen_ns: [plan.max_peers]u64 = @splat(0),
        in_use_max: u64 = 0, // the most calls in a peer's slots by its credit as read here (lags its frees: an upper bound)
        credit_stalls: u64 = 0, // links whose payload was staged while the peer's credit, as read here, held the slot
        stalled_mask: u32 = 0,

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
                    if (p.send_mask >> @intCast(link) & 1 == 1 or !plan.geq(p.load(p.l.ready(s, blk)), v)) continue;
                    if (!p.creditOk(link, call)) {
                        if (p.stalled_mask >> @intCast(link) & 1 == 0) p.credit_stalls += 1;
                        p.stalled_mask |= @as(u32, 1) << @intCast(link);
                        continue;
                    }
                    const held = call + 1 - @min(call + 1, p.load(p.l.credit(plan.linkPeer(p.rank, link))));
                    p.in_use_max = @max(p.in_use_max, held);
                    try p.postPart(link, p.l.send(s, blk), p.l.recv(s, p.rank), d.bytes[link], s, v, call);
                    p.send_mask |= @as(u32, 1) << @intCast(link);
                    did = true;
                }
                if (p.send_mask != all) break;
                p.send_mask = 0;
                p.stalled_mask = 0;
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

        /// The completions, failing on an error; returns the exchange's (a heartbeat's is not progress).
        fn drain(p: *P) !usize {
            var buf: [64]Done = undefined;
            const n = p.nic.poll(&buf) catch |e| {
                p.fail("polling the completions failed ({t})", .{e});
                return error.ProxyFailed;
            };
            var moved: usize = 0;
            for (buf[0..n]) |w| {
                const link: usize = @intCast(w.id >> 4 & 3);
                const i: usize = @intCast(w.id >> 6 & 3);
                const call = w.id >> 11;
                if (!w.ok) {
                    p.fail("an RDMA write to rank {d} (kind {d}, stripe {d}, call {d}) failed: status {d}", .{ plan.linkPeer(p.rank, link), w.id & 0xf, i, call, w.status });
                    return error.ProxyFailed;
                }
                p.outstanding[link][i] -= @intCast(1 + (w.id >> 10 & 1));
                switch (w.id & 0xf) {
                    kind_send => p.pending[call % plan.qcap] -= 1,
                    kind_beat => p.beat_out[link] = false,
                    else => {},
                }
                moved += @intFromBool(w.id & 0xf != kind_beat);
            }
            return moved;
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

        /// Every heartbeat period, busy or idle: a counter to every peer, which watches it while calls are outstanding.
        fn beat(p: *P, t: u64, last: *u64) !void {
            const period = if (p.peer_ns > 0) @min(beat_most_ns, p.peer_ns / 4) else beat_most_ns;
            if (t - last.* < period) return;
            last.* = t;
            p.beats +%= 1;
            for (0..p.peers()) |link| {
                if (p.beat_out[link] or p.outstanding[link][0] + 1 > depth - 4) continue;
                try p.nic.word(link, 0, p.l.beat(p.rank), p.beats, wrId(kind_beat, link, 0, 0));
                p.outstanding[link][0] += 1;
                p.beat_out[link] = true;
            }
        }

        /// With calls outstanding: every peer's heartbeat moving within peer_ns, and progress within timeout_ns.
        fn watch(p: *P, t: u64, progress: u64) bool {
            for (0..p.peers()) |link| {
                const h = p.load(p.l.beat(plan.linkPeer(p.rank, link)));
                if (h != p.seen[link]) {
                    p.seen[link] = h;
                    p.seen_ns[link] = t;
                } else if (p.peer_ns > 0 and t - p.seen_ns[link] > p.peer_ns) {
                    p.fail("no heartbeat from rank {d} for {d} ms (TF_GLM_OVN_PRX_PEER_S): it is stopped or gone", .{ plan.linkPeer(p.rank, link), (t - p.seen_ns[link]) / std.time.ns_per_ms });
                    return true;
                }
            }
            if (p.timeout_ns > 0 and t - progress > p.timeout_ns) {
                p.fail("no progress for {d} ms with calls outstanding (call {d} of {d} sent, {d} freed)", .{ (t - progress) / std.time.ns_per_ms, p.next_send, p.posted.load(.acquire), p.freed });
                return true;
            }
            return false;
        }

        /// The thread: steps until stopped, or failed (a peer's abort, a failed write, a silent peer, no progress).
        pub fn loop(p: *P, io: std.Io) void {
            var t = now(io);
            var progress = t;
            var beat_at = t;
            var idle: u64 = 0;
            for (&p.seen_ns) |*s| s.* = t;
            while (p.running.load(.monotonic)) {
                const origin = p.load(p.l.abort()); // first: a peer's abort lands before its errored queue pairs fail us
                if (origin != 0) {
                    p.origin = origin;
                    p.fail("rank {d} failed and its abort reached this rank", .{origin - 1});
                    break;
                }
                const did = p.step() catch |e| {
                    p.fail("the proxy stopped: {t}", .{e});
                    break;
                };
                idle = if (did) 0 else idle + 1;
                t = now(io); // every pass, as prx_proxy.c: a poll can yield, so a count of polls is no clock
                p.beat(t, &beat_at) catch |e| {
                    p.fail("a heartbeat could not be posted: {t}", .{e});
                    break;
                };
                if (did) {
                    progress = t;
                } else if (p.busy()) {
                    if (p.watch(t, progress)) break;
                } else { // caught up: nothing can be late, so the watch restarts from here
                    progress = t;
                    for (0..p.peers()) |link| {
                        p.seen_ns[link] = t;
                        p.seen[link] = p.load(p.l.beat(plan.linkPeer(p.rank, link)));
                    }
                    if (idle >= idle_spins) nap.pause();
                }
            }
            if (p.failed.load(.acquire)) p.hold(io);
        }

        /// After a failure: waits released, aborts sent, queue pairs to error, and after exit_ns the process exits.
        fn hold(p: *P, io: std.Io) void {
            p.release();
            const t0 = now(io);
            std.log.warn("TF_GLM_OVN_PRX failed on rank {d}: {s}; the GPU's waits are released", .{ p.rank, std.mem.sliceTo(&p.err, 0) });
            const origin: u32 = if (p.origin != 0) p.origin else @intCast(p.rank + 1);
            var aborts: usize = 0;
            for (0..p.peers()) |link| for (0..p.stripes) |i| {
                if (p.nic.word(link, i, p.l.abort(), origin, wrId(kind_abort, link, i, 0))) |_| {
                    aborts += 1;
                } else |_| {}
            };
            var buf: [64]Done = undefined;
            while (aborts > 0 and now(io) - t0 < abort_wait_ns) {
                const n = p.nic.poll(&buf) catch break;
                for (buf[0..n]) |w| aborts -|= @intFromBool(w.id & 0xf == kind_abort);
            }
            p.nic.halt();
            while (p.running.load(.monotonic)) {
                p.release();
                _ = p.nic.poll(&buf) catch {};
                if (p.exit_ns > 0 and now(io) - t0 >= p.exit_ns) {
                    std.log.warn("TF_GLM_OVN_PRX failed on rank {d} {d} ms ago: exiting (TF_GLM_OVN_PRX_EXIT_S)", .{ p.rank, (now(io) - t0) / std.time.ns_per_ms });
                    p.exit();
                    p.exit_ns = 0; // a test's exit returns: once
                }
                nap.pause();
            }
        }

        /// The failure's message, or null.
        pub fn failure(p: *const P) ?[]const u8 {
            if (!p.failed.load(.acquire)) return null;
            return std.mem.sliceTo(&p.err, 0);
        }
    };
}
