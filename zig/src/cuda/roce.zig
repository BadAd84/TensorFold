//! One-shot RoCE all-gathers after b12x's RoCEnante (local-inference-lab/b12x, Apache-2.0); two to four ranks.

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");
const tp_group = @import("tp_group.zig");
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");
const nap = @import("roce_nap.zig");

const region = @import("roce_region.zig");
const Layout = region.Layout;
const Blob = region.Blob;
const peerRank = region.peerRank;
const myIndexAt = region.myIndexAt;
const sameSubnet = region.sameSubnet;
pub const max_world_roce = region.max_world_roce;
const c_seq = region.c_seq;
const c_nbytes = region.c_nbytes;
const c_error = region.c_error;
const c_hca = region.c_hca;
const max_hcas = 2;
const max_peers = max_world_roce - 1;
const slots = 2;
const flag_stride = 128;
const send_depth = 256;
const port_num: u8 = 1;
const threads = 512;
const max_blocks = 8; // the largest grid; smaller shards launch fewer (a power of two)
const classes = 4; // grid size classes 1, 2, 4, 8 (each its own arrival counters)
const idle_spins: u64 = 20_000_000; // doorbell polls before the proxy naps between them (a model's gaps are short)
const spin_limit: u32 = 100_000_000; // the kernel's flag polls before it gives up and poisons the exchange

// -- libibverbs (the ABI of rdma-core's verbs.h; offsets checked against its headers) ---------------------------------
const roce_abi = @import("roce_abi.zig");
const Device = roce_abi.Device;
const Context = roce_abi.Context;
const Pd = roce_abi.Pd;
const Cq = roce_abi.Cq;
const Mr = roce_abi.Mr;
const Qp = roce_abi.Qp;
const QpCap = roce_abi.QpCap;
const QpInitAttr = roce_abi.QpInitAttr;
const GlobalRoute = roce_abi.GlobalRoute;
const AhAttr = roce_abi.AhAttr;
const QpAttr = roce_abi.QpAttr;
const Sge = roce_abi.Sge;
const SendWr = roce_abi.SendWr;
const Wc = roce_abi.Wc;
const PortAttr = roce_abi.PortAttr;
const ops_poll_cq = roce_abi.ops_poll_cq;
const ops_post_send = roce_abi.ops_post_send;
const PollCq = roce_abi.PollCq;
const PostSend = roce_abi.PostSend;
const wr_rdma_write = roce_abi.wr_rdma_write;
const send_signaled = roce_abi.send_signaled;
const send_inline = roce_abi.send_inline;
const qpt_rc = roce_abi.qpt_rc;
const qps_init = roce_abi.qps_init;
const qps_rtr = roce_abi.qps_rtr;
const qps_rts = roce_abi.qps_rts;
const port_active = roce_abi.port_active;
const access_local_write = roce_abi.access_local_write;
const access_remote_write = roce_abi.access_remote_write;
const qp_state = roce_abi.qp_state;
const qp_access_flags = roce_abi.qp_access_flags;
const qp_pkey_index = roce_abi.qp_pkey_index;
const qp_port = roce_abi.qp_port;
const qp_av = roce_abi.qp_av;
const qp_path_mtu = roce_abi.qp_path_mtu;
const qp_timeout = roce_abi.qp_timeout;
const qp_retry_cnt = roce_abi.qp_retry_cnt;
const qp_rnr_retry = roce_abi.qp_rnr_retry;
const qp_rq_psn = roce_abi.qp_rq_psn;
const qp_max_qp_rd_atomic = roce_abi.qp_max_qp_rd_atomic;
const qp_min_rnr_timer = roce_abi.qp_min_rnr_timer;
const qp_sq_psn = roce_abi.qp_sq_psn;
const qp_max_dest_rd_atomic = roce_abi.qp_max_dest_rd_atomic;
const qp_dest_qpn = roce_abi.qp_dest_qpn;
const Api = roce_abi.Api;
fn opsCall(comptime T: type, ctx: *Context, offset: usize) T {
    const slot: *const T = @ptrFromInt(@intFromPtr(ctx) + offset);
    return slot.*;
}

pub const Error = error{ VerbsUnavailable, NoHca, HcaFailed, PeerRefused, ProxyFailed, ExchangeTimedOut, Misaligned };

/// TF_ROCE=0 keeps every gather on NCCL; unset or any other value asks for this path (both ranks must ask).
pub fn wanted() bool {
    const v = std.c.getenv("TF_ROCE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

const Hca = struct {
    ctx: ?*Context = null,
    pd: ?*Pd = null,
    mr: ?*Mr = null,
    cq: ?*Cq = null,
    qps: [max_peers]?*Qp = @splat(null), // one queue pair per peer on this HCA's completion queue
    gid: [16]u8 = @splat(0),
    gid_index: c_int = 0,
    lid: u16 = 0,
    mtu: u32 = 0,
    outstanding: u32 = 0,
};

pub const Gather = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    lib: platform.DynLib,
    api: Api,
    rank: u8,
    world: usize = 2,
    n_peers: usize = 1,
    hcas: [max_hcas]Hca = @splat(.{}),
    n_hca: u32 = 0,
    /// Per peer, the HCAs on the same cable found by shared /24; index-h pairing is wrong on a crossed mesh.
    links: [max_peers][max_hcas]u8 = @splat(@splat(0)),
    peer_links: [max_peers][max_hcas]u8 = @splat(@splat(0)),
    n_links: u32 = 0,
    layout: Layout,
    region: memory.HostBuffer,
    region_dev: abi.DevicePtr,
    counters: memory.DeviceBuffer,
    module: Module,
    f: Function,
    max_bytes: usize,
    peer_addr: [max_peers]u64 = @splat(0),
    peer_rkey: [max_peers][max_hcas]u32 = @splat(@splat(0)),
    last_seq: u32 = 0,
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    err: [256]u8 = @splat(0),
    gathers: u64 = 0,

    /// Every rank's exchange or null on all; setup crosses ``group``'s communicator on ``s``. ``max_bytes``: a shard.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, group: *const tp_group.Group, s: Stream, want: bool, max_bytes: usize) !?*Gather {
        if (group.world < 2 or group.world > max_world_roce) return error.PeerRefused;
        const g = try gpa.create(Gather);
        var mine: Blob = std.mem.zeroes(Blob);
        const ok = want and blk: {
            g.setup(gpa, io, d, group, max_bytes) catch |e| {
                std.log.warn("RoCE all-gather: {s} ({s}); NCCL carries every gather", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        if (ok) g.blob(&mine);
        var all = try gpa.alloc(Blob, group.world);
        defer gpa.free(all);
        group.gather(Blob, d, s, mine, all) catch |e| {
            if (ok) g.teardown();
            gpa.destroy(g);
            return e;
        };
        var usable = ok;
        var cap = mine.max_bytes;
        for (all) |b| {
            if (b.ok == 0 or b.n_hca != mine.n_hca or b.slot_bytes != mine.slot_bytes) usable = false;
            cap = @min(cap, b.max_bytes);
        }
        if (!usable) {
            if (ok) g.teardown();
            gpa.destroy(g);
            return null;
        }
        g.max_bytes = @intCast(cap); // one cap for fits on every rank
        g.selectLinks(all) catch |e| {
            std.log.warn("RoCE all-gather: {s} ({s}); NCCL carries every gather", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
            g.teardown();
            gpa.destroy(g);
            return null;
        };
        const connected = blk: {
            for (0..g.n_peers) |p| g.connect(p, &all[peerRank(g.rank, p, g.world)]) catch |e| {
                std.log.warn("RoCE all-gather: {s} ({s}); NCCL carries every gather", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        const bits = try gpa.alloc(u32, group.world);
        defer gpa.free(bits);
        group.gather(u32, d, s, @intFromBool(connected), bits) catch |e| {
            g.teardown();
            gpa.destroy(g);
            return e;
        };
        for (bits) |b| if (b == 0) {
            g.teardown();
            gpa.destroy(g);
            return null;
        };
        g.running.store(true, .release);
        g.thread = try std.Thread.spawn(.{}, proxy, .{g});
        std.log.info("RoCE all-gather: {d} rank(s), {d} link(s) over {d} HCA(s), shards up to {d} KiB by RDMA writes", .{ g.world, g.n_links, g.n_hca, g.max_bytes >> 10 });
        return g;
    }

    fn fail(g: *Gather, comptime fmt: []const u8, args: anytype) void {
        _ = std.fmt.bufPrint(g.err[0 .. g.err.len - 1], fmt, args) catch {};
    }

    fn setup(g: *Gather, gpa: std.mem.Allocator, io: std.Io, d: *const Driver, group: *const tp_group.Group, max_bytes: usize) !void {
        g.* = .{ .gpa = gpa, .d = d, .lib = undefined, .api = undefined, .rank = @intCast(group.rank), .world = group.world, .n_peers = group.world - 1, .layout = .of(max_bytes, group.world), .region = undefined, .region_dev = 0, .counters = undefined, .module = undefined, .f = undefined, .max_bytes = max_bytes };
        g.lib = platform.DynLib.open("libibverbs.so.1") catch return platform.optional_libraries.unavailable(.verbs);
        errdefer g.lib.close();
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(g.api, name) = g.lib.lookup(T, name) orelse return @import("sys").optional_libraries.unavailable(.verbs);
        }
        var names: [max_hcas][64]u8 = undefined;
        var lens: [max_hcas]usize = undefined;
        g.n_hca = try hcaNames(io, gpa, &names, &lens);
        g.region = try memory.HostBuffer.allocMapped(d, g.layout.total);
        errdefer g.region.free();
        @memset(g.region.bytes, 0);
        g.region_dev = try g.region.device();
        g.counters = try memory.DeviceBuffer.alloc(d, 64 * 4);
        errdefer g.counters.free();
        const zeros: [64 * 4]u8 = @splat(0);
        try g.counters.upload(0, &zeros);
        g.module = try Module.load(d, kernels.roce_gather);
        errdefer g.module.unload();
        g.f = try g.module.function("roce_gather");
        errdefer g.closeHcas();
        for (0..g.n_hca) |h| try g.openHca(io, gpa, h, names[h][0..lens[h]]);
    }

    fn closeHcas(g: *Gather) void {
        for (&g.hcas) |*x| {
            for (&x.qps) |*q| {
                if (q.*) |qp| _ = g.api.ibv_destroy_qp(qp);
            }
            if (x.cq) |q| _ = g.api.ibv_destroy_cq(q);
            if (x.mr) |m| _ = g.api.ibv_dereg_mr(m);
            if (x.pd) |p| _ = g.api.ibv_dealloc_pd(p);
            if (x.ctx) |c| _ = g.api.ibv_close_device(c);
            x.* = .{};
        }
    }

    fn openHca(g: *Gather, io: std.Io, gpa: std.mem.Allocator, h: usize, name: []const u8) !void {
        const a = &g.api;
        var num: c_int = 0;
        const list = a.ibv_get_device_list(&num) orelse return error.NoHca;
        defer a.ibv_free_device_list(list);
        var dev: ?*Device = null;
        for (0..@intCast(num)) |i| {
            const n = a.ibv_get_device_name(list[i].?) orelse continue;
            if (std.mem.eql(u8, std.mem.span(n), name)) dev = list[i];
        }
        const hca = &g.hcas[h];
        hca.ctx = a.ibv_open_device(dev orelse {
            g.fail("RDMA device {s} not found", .{name});
            return error.NoHca;
        }) orelse return error.HcaFailed;
        var port: PortAttr = std.mem.zeroes(PortAttr);
        if (a.ibv_query_port(hca.ctx.?, port_num, &port) != 0) return error.HcaFailed;
        if (std.mem.readInt(u32, port.bytes[0..4], .little) != port_active) {
            g.fail("{s} port 1 is not active", .{name});
            return error.NoHca;
        }
        hca.mtu = std.mem.readInt(u32, port.bytes[8..12], .little);
        hca.lid = std.mem.readInt(u16, port.bytes[34..36], .little);
        hca.gid_index = try gidIndex(io, gpa, name);
        if (a.ibv_query_gid(hca.ctx.?, port_num, hca.gid_index, &hca.gid) != 0) return error.HcaFailed;
        hca.pd = a.ibv_alloc_pd(hca.ctx.?) orelse return error.HcaFailed;
        hca.mr = a.ibv_reg_mr(hca.pd.?, g.region.bytes.ptr, g.region.bytes.len, access_local_write | access_remote_write) orelse {
            g.fail("{s}: registering the pinned region failed", .{name});
            return error.HcaFailed;
        };
        hca.cq = a.ibv_create_cq(hca.ctx.?, send_depth, null, null, 0) orelse return error.HcaFailed;
        var attr: QpInitAttr = .{ .send_cq = hca.cq, .recv_cq = hca.cq, .qp_type = qpt_rc, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 } };
        for (0..g.n_peers) |p| {
            hca.qps[p] = a.ibv_create_qp(hca.pd.?, &attr) orelse return error.HcaFailed;
            var init_attr: QpAttr = .{ .qp_state = qps_init, .port_num = port_num, .qp_access_flags = access_remote_write };
            if (a.ibv_modify_qp(hca.qps[p].?, &init_attr, qp_state | qp_pkey_index | qp_port | qp_access_flags) != 0) return error.HcaFailed;
        }
    }

    fn blob(g: *Gather, b: *Blob) void {
        b.ok = 1;
        b.n_hca = g.n_hca;
        b.slot_bytes = g.layout.slot_bytes;
        b.max_bytes = g.max_bytes;
        b.addr = @intFromPtr(g.region.bytes.ptr);
        for (0..g.n_hca) |h| {
            const x = &g.hcas[h];
            b.rkey[h] = x.mr.?.rkey;
            for (0..g.n_peers) |p| b.qpn[p][h] = x.qps[p].?.qp_num;
            b.mtu[h] = x.mtu;
            b.lid[h] = x.lid;
            b.gid[h] = x.gid;
        }
    }

    /// Pair each peer's HCAs by the /24 their GIDs share; no match falls back to index-h.
    fn selectLinks(g: *Gather, all: []const Blob) !void {
        var count: usize = 0;
        var taken_h: [max_hcas]bool = @splat(false); // a port faces one peer: never lend it to a second
        for (0..g.n_peers) |p| {
            const peer = &all[peerRank(g.rank, p, g.world)];
            var n: usize = 0;
            for (0..g.n_hca) |h| {
                var q: usize = 0;
                var best: ?usize = null;
                while (q < peer.n_hca) : (q += 1) {
                    if (!sameSubnet(g.hcas[h].gid, peer.gid[q])) continue;
                    var taken = false;
                    for (g.peer_links[p][0..n]) |u| {
                        if (@as(usize, u) == q) taken = true;
                    }
                    if (taken) continue;
                    best = q;
                    break;
                }
                if (best) |pq| {
                    taken_h[h] = true;
                    g.links[p][n] = @intCast(h);
                    g.peer_links[p][n] = @intCast(pq);
                    n += 1;
                    if (n == max_hcas) break;
                }
            }
            if (n == 0) { // no shared subnet: index-h among the ports no earlier peer took
                for (0..@min(g.n_hca, max_hcas)) |h| {
                    if (taken_h[h]) continue;
                    taken_h[h] = true;
                    g.links[p][n] = @intCast(h);
                    g.peer_links[p][n] = @intCast(h);
                    n += 1;
                    if (n == max_hcas) break;
                }
                if (n == 0) return error.PeerRefused; // every port already faces another peer
            }
            if (p == 0) {
                count = n;
            } else if (n != count) {
                g.fail("peer {d} has {d} link(s) where peer 0 has {d}", .{ p, n, count });
                return error.PeerRefused;
            }
        }
        g.n_links = @intCast(count);
    }

    fn connect(g: *Gather, p: usize, peer: *const Blob) !void {
        const r = peerRank(g.rank, p, g.world);
        const me_there = myIndexAt(g.rank, r, g.world); // my slot in r's peer list: qpn[me_there][q] targets me
        g.peer_addr[p] = peer.addr;
        for (0..g.n_links) |i| {
            const h = g.links[p][i];
            const q = g.peer_links[p][i];
            const x = &g.hcas[h];
            g.peer_rkey[p][h] = peer.rkey[q];
            var rtr: QpAttr = .{ .qp_state = qps_rtr, .path_mtu = @min(peer.mtu[q], x.mtu), .dest_qp_num = peer.qpn[me_there][q], .max_dest_rd_atomic = 1, .min_rnr_timer = 12, .ah_attr = .{ .is_global = 1, .dlid = peer.lid[q], .port_num = port_num, .grh = .{ .dgid = peer.gid[q], .sgid_index = @intCast(x.gid_index), .hop_limit = 64 } } };
            if (g.api.ibv_modify_qp(x.qps[p].?, &rtr, qp_state | qp_av | qp_path_mtu | qp_dest_qpn | qp_rq_psn | qp_max_dest_rd_atomic | qp_min_rnr_timer) != 0) {
                g.fail("HCA {d} to peer {d} HCA {d}: the queue pair refused RTR (errno {d})", .{ h, p, q, std.c._errno().* });
                return error.PeerRefused;
            }
            var rts: QpAttr = .{ .qp_state = qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .max_rd_atomic = 1 };
            if (g.api.ibv_modify_qp(x.qps[p].?, &rts, qp_state | qp_timeout | qp_retry_cnt | qp_rnr_retry | qp_sq_psn | qp_max_qp_rd_atomic) != 0) {
                g.fail("HCA {d} to peer {d} HCA {d}: the queue pair refused RTS", .{ h, p, q });
                return error.PeerRefused;
            }
        }
    }

    fn ctrl(g: *Gather) [*]u32 {
        return @ptrCast(@alignCast(g.region.bytes.ptr + g.layout.ctrl));
    }

    fn drain(g: *Gather, h: usize) !void {
        const x = &g.hcas[h];
        var wc: [32]Wc = undefined;
        const n = opsCall(PollCq, x.ctx.?, ops_poll_cq)(x.cq.?, wc.len, &wc);
        if (n < 0) return error.ProxyFailed;
        for (wc[0..@intCast(n)]) |w| {
            if (w.status != 0) {
                const why = g.api.ibv_wc_status_str(w.status) orelse "?";
                g.fail("RDMA write on HCA {d} failed: {s} (vendor 0x{x}, seq {d})", .{ h, std.mem.span(why), w.vendor_err, g.last_seq });
                return error.ProxyFailed;
            }
            x.outstanding -= 1;
        }
    }

    /// One sequence's shard to every peer: a stripe per link, flag signalled inline; stripes tile the shard.
    fn post(g: *Gather, seq: u32, nbytes: u32) !void {
        if (nbytes == 0 or nbytes % 16 != 0 or nbytes > g.layout.slot_bytes) {
            g.fail("bad shard size {d} at seq {d}", .{ nbytes, seq });
            return error.ProxyFailed;
        }
        const slot: u64 = seq & 1;
        const base = @intFromPtr(g.region.bytes.ptr);
        const packs = nbytes / 16;
        var seq_copy: u32 = seq;
        for (0..g.n_peers) |p| {
            const r = peerRank(g.rank, p, g.world);
            const at_peer = myIndexAt(g.rank, r, g.world);
            var offset: u64 = 0;
            for (0..g.n_links) |i| {
                const h = g.links[p][i];
                const x = &g.hcas[h];
                while (x.outstanding >= send_depth / 4) {
                    try g.drain(h);
                    if (!g.running.load(.monotonic)) return error.ProxyFailed;
                }
                var n = packs / g.n_links;
                if (i < packs % g.n_links) n += 1;
                const bytes: u32 = n * 16;
                var flag_sge: Sge = .{ .addr = @intFromPtr(&seq_copy), .length = 4, .lkey = 0 };
                var flag: SendWr = .{ .wr_id = h, .sg_list = &flag_sge, .num_sge = 1, .opcode = wr_rdma_write, .send_flags = send_signaled | send_inline, .remote_addr = g.peer_addr[p] + g.layout.flags + at_peer * g.layout.flag_peer_stride + (slot * g.n_links + i) * flag_stride, .rkey = g.peer_rkey[p][h] };
                var data_sge: Sge = .{ .addr = base + g.layout.send + slot * g.layout.slot_bytes + offset, .length = bytes, .lkey = x.mr.?.lkey };
                var data: SendWr = .{ .wr_id = h, .next = &flag, .sg_list = &data_sge, .num_sge = 1, .opcode = wr_rdma_write, .remote_addr = g.peer_addr[p] + g.layout.recv + at_peer * g.layout.recv_peer_stride + slot * g.layout.slot_bytes + offset, .rkey = g.peer_rkey[p][h] };
                var bad: ?*SendWr = null;
                const first: *SendWr = if (bytes != 0) &data else &flag;
                const rc = opsCall(PostSend, x.ctx.?, ops_post_send)(x.qps[p].?, first, &bad);
                if (rc != 0) {
                    g.fail("ibv_post_send: {d}", .{rc});
                    return error.ProxyFailed;
                }
                x.outstanding += 1;
                offset += bytes;
            }
        }
        for (0..g.n_hca) |h| try g.drain(h);
    }

    /// The proxy: spins on the doorbell and posts each sequence in order (a missed one from its slot's size).
    fn proxy(g: *Gather) void {
        const c = g.ctrl();
        var idle: u64 = 0;
        while (g.running.load(.monotonic)) {
            const seq = @atomicLoad(u32, &c[c_seq], .acquire);
            if (seq == g.last_seq) {
                idle += 1;
                if (idle % 64 == 0) for (0..g.n_hca) |h| g.drain(h) catch {
                    g.failed.store(true, .release);
                    return;
                };
                if (idle >= idle_spins) nap.pause();
                continue;
            }
            idle = 0;
            // a rank's next gather can ring before this thread saw the last one; peers stay within a slot's distance
            var pending = seq -% g.last_seq;
            if (pending > slots) {
                g.fail("the doorbell skipped {d} sequences (last {d}, now {d})", .{ pending, g.last_seq, seq });
                g.failed.store(true, .release);
                return;
            }
            var s = g.last_seq +% 1;
            while (pending > 0) : ({
                s +%= 1;
                pending -= 1;
            }) {
                const nbytes = @atomicLoad(u32, &c[c_nbytes + (s & 1)], .acquire);
                g.post(s, nbytes) catch {
                    g.failed.store(true, .release);
                    return;
                };
                g.last_seq = s;
            }
        }
    }

    /// Whether a gather of ``bytes`` a rank goes this way: the size alone decides, so every rank picks the same path.
    pub fn fits(g: *const Gather, bytes: usize) bool {
        return bytes > 0 and bytes <= g.max_bytes and bytes % 16 == 0;
    }

    /// Every rank's ``bytes`` from ``send`` into ``recv`` in rank order (as ncclAllGather); misaligned is an error.
    pub fn gather(g: *Gather, stream: abi.Stream, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) !void {
        if (send % 16 != 0 or recv % 16 != 0) return error.Misaligned;
        const packs = bytes / 16;
        const need = @max(1, std.math.divCeil(usize, packs, 2 * threads) catch unreachable);
        const grid: u32 = @intCast(@min(std.math.ceilPowerOfTwo(usize, need) catch max_blocks, max_blocks));
        const klass: i32 = @intCast(std.math.log2_int(u32, grid));
        const base = g.region_dev;
        var a: launch.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(i32, @intCast(packs)));
        a.add(@as(i32, g.rank));
        a.add(base + g.layout.recv);
        a.add(base + g.layout.flags);
        a.add(base + g.layout.send);
        a.add(base + g.layout.ctrl);
        a.add(@as(i64, @intCast(g.layout.slot_bytes)));
        a.add(g.counters.ptr);
        a.add(klass);
        a.add(spin_limit);
        a.add(@as(i32, @intCast(g.n_links)));
        a.add(@as(i32, @intCast(g.world)));
        a.add(@as(i64, @intCast(g.layout.recv_peer_stride)));
        a.add(@as(i64, @intCast(g.layout.flag_peer_stride)));
        try launch.launch(g.f, .{ .grid = .{ .x = grid }, .block = .{ .x = threads } }, .{ .handle = stream, .d = g.d }, &a);
        g.gathers += 1;
    }

    /// An error if a wait timed out on the GPU or the proxy failed (the host checks after a synchronize).
    pub fn check(g: *Gather) !void {
        const c = g.ctrl();
        const bad = @atomicLoad(u32, &c[c_error], .acquire);
        if (bad == 0 and !g.failed.load(.acquire)) return;
        std.log.err("RoCE all-gather failed (sequence {d}, HCA {d}): {s}", .{ bad, c[c_hca], std.mem.sliceTo(&g.err, 0) });
        return error.ProxyFailed;
    }

    fn teardown(g: *Gather) void {
        if (g.running.swap(false, .acq_rel)) if (g.thread) |t| t.join();
        g.closeHcas();
        g.module.unload();
        g.counters.free();
        g.region.free();
        g.lib.close();
    }

    pub fn deinit(g: *Gather) void {
        g.teardown();
        g.gpa.destroy(g);
    }
};

/// The HCAs to stripe over, at most two: TF_ROCE_HCA, else NCCL_IB_HCA without "=^" prefixes or ":port" suffixes.
fn hcaNames(io: std.Io, gpa: std.mem.Allocator, names: *[max_hcas][64]u8, lens: *[max_hcas]usize) !u32 {
    _ = io;
    _ = gpa;
    const raw = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA") orelse return error.NoHca;
    var n: u32 = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
    while (it.next()) |w| {
        if (n == max_hcas) break;
        var s = std.mem.trimStart(u8, std.mem.trim(u8, w, " "), "=^");
        if (std.mem.indexOfScalar(u8, s, ':')) |c| s = s[0..c];
        if (s.len == 0 or s.len > 63) continue;
        @memcpy(names[n][0..s.len], s);
        lens[n] = s.len;
        n += 1;
    }
    if (n == 0) return error.NoHca;
    return n;
}

/// The HCA's RoCE v2 IPv4 GID (TF_ROCE_GID_INDEX overrides): its index differs between hosts' tables.
fn gidIndex(io: std.Io, gpa: std.mem.Allocator, name: []const u8) !c_int {
    if (std.c.getenv("TF_ROCE_GID_INDEX")) |v| return std.fmt.parseInt(c_int, std.mem.span(v), 10) catch error.NoHca;
    const cwd = std.Io.Dir.cwd();
    var i: c_int = 0;
    while (i < 16) : (i += 1) {
        var pbuf: [160]u8 = undefined;
        const tpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ name, i });
        const t = cwd.readFileAlloc(io, tpath, gpa, .limited(64)) catch continue;
        defer gpa.free(t);
        if (!std.mem.startsWith(u8, t, "RoCE v2")) continue;
        const gpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gids/{d}", .{ name, i });
        const gid = cwd.readFileAlloc(io, gpath, gpa, .limited(64)) catch continue;
        defer gpa.free(gid);
        if (std.mem.startsWith(u8, gid, "0000:0000:0000:0000:0000:ffff:")) return i;
    }
    return error.NoHca;
}
