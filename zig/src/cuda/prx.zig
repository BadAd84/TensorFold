//! TF_GLM_OVN_PRX=1: split prompt exchanges by the NICs (copy-engine staging, proxy RDMA writes, stream waits).

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const Stream = @import("stream.zig").Stream;
const tp_group = @import("tp_group.zig");
const ovn = @import("ovn.zig");
const rails = @import("roce_rails.zig");
const verbs = @import("roce_verbs.zig");
const plan = @import("prx_plan.zig");
const proxy_mod = @import("prx_proxy.zig");

const max_hcas = rails.max_hcas;
const cq_depth = 4 * proxy_mod.depth;
pub const max_world = plan.max_world;

pub const Settings = struct {
    slots: usize = 4,
    slot_mb: usize = 16,
    timeout_s: u64 = 300,
    relaxed: bool = true,

    /// The TF_GLM_OVN_PRX_* geometry and limits; a value out of range is refused.
    pub fn read() error{BadOvnKnob}!Settings {
        const s: Settings = .{ .slots = try ovn.int("PRX_SLOTS", 4), .slot_mb = try ovn.int("PRX_SLOT_MB", 16), .timeout_s = try ovn.int("PRX_TIMEOUT_S", 300), .relaxed = !std.mem.eql(u8, ovn.text("PRX_RO") orelse "1", "0") };
        if (s.slots < 2 or s.slots > plan.max_slots or s.slot_mb == 0 or s.slot_mb > 2048 or s.timeout_s > 86400) return error.BadOvnKnob;
        return s;
    }
};

/// What a rank tells every peer: its geometry, region and keys, and per HCA its port and its queue pair to each rank.
const Blob = extern struct {
    ok: u32,
    n_hca: u32,
    world: u32,
    slots: u32,
    M: u64,
    addr: u64,
    rkey_sync: [max_hcas]u32,
    rkey_data: [max_hcas]u32,
    mtu: [max_hcas]u32,
    lid: [max_hcas]u32,
    gid: [max_hcas][16]u8,
    qpn: [plan.max_world][max_hcas]u32, // this rank's queue pair toward rank r on its HCA h (0: none)
};

const Hca = struct {
    ctx: ?*verbs.Context = null,
    pd: ?*verbs.Pd = null,
    sync: ?*verbs.Mr = null, // ctrl and flags, [0, data): never relaxed, so a flag cannot pass its stripe's data
    data: ?*verbs.Mr = null, // the blocks, [data, total)
    cq: ?*verbs.Cq = null,
    qps: [plan.max_world]?*verbs.Qp = @splat(null), // one queue pair per other rank
    gid: [16]u8 = @splat(0),
    gid_index: c_int = 0,
    port: verbs.Port = .{ .mtu = 0, .lid = 0 },
};

/// One peer's stripes: my HCA and its queue pair on each, the peer's keys, and its region's address.
const Link = struct {
    hca: [plan.max_stripes]u8 = @splat(0),
    qp: [plan.max_stripes]?*verbs.Qp = @splat(null),
    rkey_sync: [plan.max_stripes]u32 = @splat(0),
    rkey_data: [plan.max_stripes]u32 = @splat(0),
    addr: u64 = 0,
};

/// prx_proxy's NIC on verbs: each stripe's write then its inline flag on one queue pair, one signaled completion.
const Nic = struct {
    api: *const verbs.Api,
    hcas: []Hca,
    links: *[plan.max_peers]Link,
    base: [*]u8,

    pub fn data(n: *Nic, link: usize, i: usize, from: usize, to: usize, len: usize, flag_at: usize, v: u32, id: u64) !void {
        const k = &n.links[link];
        const x = &n.hcas[k.hca[i]];
        var val = v;
        var fs: verbs.Sge = .{ .addr = @intFromPtr(&val), .length = 4, .lkey = 0 };
        var flag: verbs.SendWr = .{ .wr_id = id, .sg_list = &fs, .num_sge = 1, .opcode = verbs.wr_rdma_write, .send_flags = verbs.send_signaled | verbs.send_inline, .remote_addr = k.addr + flag_at, .rkey = k.rkey_sync[i] };
        var ds: verbs.Sge = .{ .addr = @intFromPtr(n.base + from), .length = @intCast(len), .lkey = x.data.?.lkey };
        var dw: verbs.SendWr = .{ .wr_id = id, .next = &flag, .sg_list = &ds, .num_sge = 1, .opcode = verbs.wr_rdma_write, .remote_addr = k.addr + to, .rkey = k.rkey_data[i] };
        try n.send(x, k.qp[i].?, if (len > 0) &dw else &flag);
    }

    pub fn word(n: *Nic, link: usize, i: usize, at: usize, v: u32, id: u64) !void {
        const k = &n.links[link];
        var val = v;
        var fs: verbs.Sge = .{ .addr = @intFromPtr(&val), .length = 4, .lkey = 0 };
        var wr: verbs.SendWr = .{ .wr_id = id, .sg_list = &fs, .num_sge = 1, .opcode = verbs.wr_rdma_write, .send_flags = verbs.send_signaled | verbs.send_inline, .remote_addr = k.addr + at, .rkey = k.rkey_sync[i] };
        try n.send(&n.hcas[k.hca[i]], k.qp[i].?, &wr);
    }

    fn send(n: *Nic, x: *Hca, qp: *verbs.Qp, first: *verbs.SendWr) !void {
        _ = n;
        var bad: ?*verbs.SendWr = null;
        if (verbs.opsCall(verbs.PostSend, x.ctx.?, verbs.ops_post_send)(qp, first, &bad) != 0) return error.PostFailed;
    }

    pub fn poll(n: *Nic, out: []proxy_mod.Done) !usize {
        var k: usize = 0;
        for (n.hcas) |*x| {
            if (k == out.len) break;
            var wc: [64]verbs.Wc = undefined;
            const room = @min(wc.len, out.len - k);
            const got = verbs.opsCall(verbs.PollCq, x.ctx.?, verbs.ops_poll_cq)(x.cq.?, @intCast(room), &wc);
            if (got < 0) return error.PollFailed;
            for (wc[0..@intCast(got)]) |w| {
                out[k] = .{ .id = w.wr_id, .ok = w.status == 0, .status = w.status };
                k += 1;
            }
        }
        return k;
    }
};

const Proxy = proxy_mod.Proxy(Nic);
const MemOp = *const fn (abi.Stream, abi.DevicePtr, u32, c_uint) callconv(.c) abi.Result;

/// The GPU's side of a call on one stream: stream memory operations on the region's words and copy-engine copies.
const Ops = struct {
    d: *const Driver,
    s: abi.Stream,
    dev: abi.DevicePtr,
    host: [*]u8,
    wait_fn: MemOp,
    write_fn: MemOp,

    pub fn wait(o: *const Ops, at: usize, v: u32) !void {
        try o.d.check(o.wait_fn(o.s, o.dev + at, v, 0), "cuStreamWaitValue32"); // 0: CU_STREAM_WAIT_VALUE_GEQ
    }
    pub fn write(o: *const Ops, at: usize, v: u32) !void {
        try o.d.check(o.write_fn(o.s, o.dev + at, v, 0), "cuStreamWriteValue32"); // after the stream's earlier work
    }
    pub fn stage(o: *const Ops, at: usize, src: u64, n: usize) !void {
        try o.d.check(o.d.api.cuMemcpyDtoHAsync_v2(o.host + at, src, n, o.s), "cuMemcpyDtoHAsync");
    }
    pub fn unstage(o: *const Ops, dst: u64, at: usize, n: usize) !void {
        try o.d.check(o.d.api.cuMemcpyHtoDAsync_v2(dst, o.host + at, n, o.s), "cuMemcpyHtoDAsync");
    }
};

pub const Channel = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    lib: platform.DynLib = undefined,
    api: verbs.Api = undefined,
    cu: platform.DynLib = undefined,
    wait_fn: MemOp = undefined,
    write_fn: MemOp = undefined,
    l: plan.Layout,
    rank: usize,
    stripes: usize = 0,
    set: Settings,
    region: ?memory.HostBuffer = null,
    region_dev: abi.DevicePtr = 0,
    hcas: [max_hcas]Hca = @splat(.{}),
    n_hca: usize = 0,
    links: [plan.max_peers]Link = @splat(.{}),
    nic: Nic = undefined,
    proxy: ?*Proxy = null,
    thread: ?std.Thread = null,
    opened: u8 = 0, // 1: verbs open, 2: and the driver's memory operations
    calls: u64 = 0, // exchanges this channel carried
    err: [256]u8 = @splat(0),

    fn fail(ch: *Channel, comptime fmt: []const u8, args: anytype) void {
        _ = std.fmt.bufPrint(ch.err[0 .. ch.err.len - 1], fmt, args) catch {};
    }

    /// Every rank's channel or null on all (a warning says why); setup crosses `group`'s communicator on `s`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, group: *const tp_group.Group, s: Stream, set: Settings) !?*Channel {
        const l = plan.Layout.of(group.world, set.slots, set.slot_mb << 20) catch return error.BadOvnKnob;
        const ch = try gpa.create(Channel);
        ch.* = .{ .gpa = gpa, .d = d, .l = l, .rank = group.rank, .set = set };
        var mine = std.mem.zeroes(Blob);
        if (ch.setup(io)) {
            ch.blob(&mine);
        } else |e| std.log.warn("TF_GLM_OVN_PRX: {t} ({s}); NCCL carries the prompt exchanges", .{ e, std.mem.sliceTo(&ch.err, 0) });
        const all = try gpa.alloc(Blob, group.world);
        defer gpa.free(all);
        group.gather(Blob, d, s, mine, all) catch |e| {
            ch.deinit();
            return e;
        };
        var usable = true;
        for (all) |b| usable = usable and b.ok == 1 and b.world == mine.world and b.slots == mine.slots and b.M == mine.M;
        const connected = usable and blk: {
            ch.connect(all) catch |e| {
                std.log.warn("TF_GLM_OVN_PRX: {t} ({s}); NCCL carries the prompt exchanges", .{ e, std.mem.sliceTo(&ch.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        const bits = try gpa.alloc(u32, group.world);
        defer gpa.free(bits);
        group.gather(u32, d, s, @intFromBool(connected), bits) catch |e| {
            ch.deinit();
            return e;
        };
        for (bits) |b| if (b == 0) {
            if (mine.ok == 1 and connected == usable) std.log.warn("TF_GLM_OVN_PRX: another rank could not set up or connect (or its geometry differs); NCCL carries the prompt exchanges", .{});
            ch.deinit();
            return null;
        };
        const p = try gpa.create(Proxy);
        p.* = .{ .nic = &ch.nic, .region = ch.region.?.bytes.ptr, .l = ch.l, .rank = ch.rank, .stripes = ch.stripes, .timeout_ns = set.timeout_s * std.time.ns_per_s };
        ch.proxy = p;
        ch.thread = std.Thread.spawn(.{}, Proxy.loop, .{ p, io }) catch |e| {
            ch.deinit();
            return e;
        };
        std.log.info("TF_GLM_OVN_PRX connected: {d} ranks, {d} stripe(s) a peer over {d} HCA(s), {d} slots of {d} MiB ({d} MiB pinned)", .{ group.world, ch.stripes, ch.n_hca, set.slots, set.slot_mb, ch.l.total >> 20 });
        return ch;
    }

    fn setup(ch: *Channel, io: std.Io) !void {
        ch.lib = platform.DynLib.open("libibverbs.so.1") catch return platform.optional_libraries.unavailable(.verbs);
        ch.opened = 1;
        inline for (@typeInfo(verbs.Api).@"struct".field_names, @typeInfo(verbs.Api).@"struct".field_types) |name, T| {
            @field(ch.api, name) = ch.lib.lookup(T, name) orelse return platform.optional_libraries.unavailable(.verbs);
        }
        ch.cu = platform.DynLib.open(platform.cuda_driver) catch return error.DriverUnavailable;
        ch.opened = 2;
        ch.wait_fn = ch.cu.lookup(MemOp, "cuStreamWaitValue32_v2") orelse return error.NoStreamMemOps;
        ch.write_fn = ch.cu.lookup(MemOp, "cuStreamWriteValue32_v2") orelse return error.NoStreamMemOps;
        ch.region = try memory.HostBuffer.allocMapped(ch.d, ch.l.total);
        @memset(ch.region.?.bytes, 0);
        ch.region_dev = try ch.region.?.device();
        var names: [max_hcas][64]u8 = undefined;
        var lens: [max_hcas]usize = undefined;
        const cap: usize = if (try rails.wanted()) max_hcas else rails.default_hcas; // TF_GLM_OVN_ROCE_RAILS, as the gather
        ch.n_hca = try rails.hcaNames(io, ch.gpa, &names, &lens, cap);
        for (0..ch.n_hca) |h| try ch.openHca(io, h, names[h][0..lens[h]]);
        ch.nic = .{ .api = &ch.api, .hcas = ch.hcas[0..ch.n_hca], .links = &ch.links, .base = ch.region.?.bytes.ptr };
    }

    fn openHca(ch: *Channel, io: std.Io, h: usize, name: []const u8) !void {
        const a = &ch.api;
        const x = &ch.hcas[h];
        x.ctx = verbs.openDevice(a, name) catch |e| {
            ch.fail("RDMA device {s}: {t}", .{ name, e });
            return e;
        };
        x.port = verbs.activePort(a, x.ctx.?) catch |e| {
            ch.fail("{s} port 1: {t}", .{ name, e });
            return e;
        };
        x.gid_index = try rails.gidIndex(io, ch.gpa, name);
        if (a.ibv_query_gid(x.ctx.?, verbs.port_num, x.gid_index, &x.gid) != 0) return error.HcaFailed;
        x.pd = a.ibv_alloc_pd(x.ctx.?) orelse return error.HcaFailed;
        const bytes = ch.region.?.bytes;
        const access: c_int = verbs.access_local_write | verbs.access_remote_write;
        const relaxed: c_int = if (ch.set.relaxed) verbs.access_relaxed else 0;
        x.sync = a.ibv_reg_mr(x.pd.?, bytes.ptr, ch.l.data, access) orelse {
            ch.fail("{s}: registering the flags failed", .{name});
            return error.HcaFailed;
        };
        x.data = a.ibv_reg_mr(x.pd.?, bytes.ptr + ch.l.data, ch.l.total - ch.l.data, access | relaxed) orelse {
            ch.fail("{s}: registering the {d} MiB of blocks failed", .{ name, (ch.l.total - ch.l.data) >> 20 });
            return error.HcaFailed;
        };
        x.cq = a.ibv_create_cq(x.ctx.?, cq_depth, null, null, 0) orelse return error.HcaFailed;
        for (0..ch.l.world) |r| if (r != ch.rank) {
            x.qps[r] = try verbs.makeQp(a, x.pd.?, x.cq.?, proxy_mod.depth);
        };
    }

    fn blob(ch: *const Channel, b: *Blob) void {
        b.ok = 1;
        b.n_hca = @intCast(ch.n_hca);
        b.world = @intCast(ch.l.world);
        b.slots = @intCast(ch.l.slots);
        b.M = ch.l.M;
        b.addr = @intFromPtr(ch.region.?.bytes.ptr);
        for (ch.hcas[0..ch.n_hca], 0..) |x, h| {
            b.rkey_sync[h] = x.sync.?.rkey;
            b.rkey_data[h] = x.data.?.rkey;
            b.mtu[h] = x.port.mtu;
            b.lid[h] = x.port.lid;
            b.gid[h] = x.gid;
            for (0..ch.l.world) |r| if (x.qps[r]) |q| {
                b.qpn[r][h] = q.qp_num;
            };
        }
    }

    /// Each peer's stripes (HCAs paired by a shared /24, cut to the ranks' fewest, as the gather), moved to RTS.
    fn connect(ch: *Channel, all: []const Blob) !void {
        const least = rails.leastLinks(all, max_hcas);
        ch.stripes = @min(least, plan.max_stripes);
        if (ch.stripes == 0) return error.PeerRefused;
        for (0..ch.l.world - 1) |link| {
            const r = plan.linkPeer(ch.rank, link);
            const peer = &all[r];
            var mine: [max_hcas]u8 = undefined;
            var theirs: [max_hcas]u8 = undefined;
            const n = rails.keep(&mine, &theirs, rails.pairs(&all[ch.rank], peer, max_hcas, &mine, &theirs), ch.stripes, r < ch.rank);
            if (n != ch.stripes) return error.PeerRefused;
            const k = &ch.links[link];
            k.addr = peer.addr;
            for (0..n) |i| {
                const h = mine[i];
                const q = theirs[i];
                const x = &ch.hcas[h];
                k.hca[i] = h;
                k.qp[i] = x.qps[r];
                k.rkey_sync[i] = peer.rkey_sync[q];
                k.rkey_data[i] = peer.rkey_data[q];
                verbs.connectQp(&ch.api, x.qps[r].?, x.port.mtu, x.gid_index, .{ .mtu = peer.mtu[q], .lid = @intCast(peer.lid[q]), .gid = peer.gid[q], .qpn = peer.qpn[ch.rank][q] }) catch |e| {
                    ch.fail("HCA {d} to rank {d} HCA {d}: {t}", .{ h, r, q, e });
                    return error.PeerRefused;
                };
            }
        }
    }

    /// Whether an exchange on `s` can take this channel: a stream capture cannot replay the host's posts.
    pub fn capturing(ch: *const Channel, s: Stream) !bool {
        var st: abi.CaptureStatus = .none;
        try ch.d.check(ch.d.api.cuStreamIsCapturing(s.handle, &st), "cuStreamIsCapturing");
        return st != .none;
    }

    /// Exchange `x` on `s` (enqueued: the stream waits for the peers' bytes, the host does not).
    pub fn exchange(ch: *Channel, s: Stream, x: *const plan.Exchange) !void {
        const p = ch.proxy.?;
        if (p.failure()) |why| {
            std.log.err("TF_GLM_OVN_PRX failed: {s}", .{why});
            return error.ProxyFailed;
        }
        const ops: Ops = .{ .d = ch.d, .s = s.handle, .dev = ch.region_dev, .host = ch.region.?.bytes.ptr, .wait_fn = ch.wait_fn, .write_fn = ch.write_fn };
        try plan.run(p, &ops, &ch.l, ch.rank, ch.stripes, x);
        ch.calls += 1;
    }

    /// An error once the proxy failed (what the GPU computed since is garbage): after a synchronize, as the gather's.
    pub fn check(ch: *const Channel) !void {
        const p = ch.proxy orelse return;
        const why = p.failure() orelse return;
        std.log.err("TF_GLM_OVN_PRX failed: {s} ({d} bytes sent, call {d} of {d})", .{ why, p.sent, p.next_send, p.posted.load(.acquire) });
        return error.ProxyFailed;
    }

    pub fn deinit(ch: *Channel) void {
        if (ch.proxy) |p| {
            p.running.store(false, .release);
            if (ch.thread) |t| t.join();
            ch.gpa.destroy(p);
        }
        for (&ch.hcas) |*x| {
            for (&x.qps) |*q| if (q.*) |qp| {
                _ = ch.api.ibv_destroy_qp(qp);
            };
            if (x.cq) |q| _ = ch.api.ibv_destroy_cq(q);
            if (x.data) |m| _ = ch.api.ibv_dereg_mr(m);
            if (x.sync) |m| _ = ch.api.ibv_dereg_mr(m);
            if (x.pd) |p| _ = ch.api.ibv_dealloc_pd(p);
            if (x.ctx) |c| _ = ch.api.ibv_close_device(c);
        }
        if (ch.region) |*r| r.free();
        if (ch.opened >= 2) ch.cu.close();
        if (ch.opened >= 1) ch.lib.close();
        ch.gpa.destroy(ch);
    }
};

test "PRX channel: the blob is the shape every rank gathers, and the knobs keep their defaults" {
    try std.testing.expectEqual(@as(usize, 32 + 4 * 4 * 4 + 4 * 16 + plan.max_world * max_hcas * 4), @sizeOf(Blob));
    const s: Settings = .{};
    try std.testing.expectEqual(@as(usize, 4), s.slots);
    const l = try plan.Layout.of(3, s.slots, s.slot_mb << 20);
    try std.testing.expectEqual(@as(usize, 320), (l.total - l.data) >> 20);
}
