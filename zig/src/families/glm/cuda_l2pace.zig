//! TF_GLM_OVN_L2PF_PACED: upstream #1580's paced L2 prefetch (cuda_l2pf.zig there) beside our TF_GLM_L2PF.

const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const projection = @import("cuda_proj.zig");

/// #1580's defaults (TF_DSV41_L2PF's): 12 MiB a site, 2 CTAs, 150 GB/s; the stream gate is the knob's own value.
pub const Options = struct {
    streams: u32 = 0, // TF_GLM_OVN_L2PF_PACED: on while a window holds at most this many streams (0: off)
    mb: u32 = 12, // TF_GLM_OVN_L2PF_MB: a site's budget in MiB
    ctas: u32 = 2, // TF_GLM_OVN_L2PF_CTAS: the CTAs a site's prefetch spreads over
    pace_gbps: u32 = 150, // TF_GLM_OVN_L2PF_GBPS: the sites beside a gather read at this rate (0: unpaced)

    pub fn read() !Options {
        var o: Options = .{ .streams = @intCast(@min(try cuda.ovn.int("L2PF_PACED", 0), std.math.maxInt(u32))) };
        if (o.streams == 0) return o;
        o.mb = @intCast(@min(try cuda.ovn.int("L2PF_MB", 12), 1024));
        o.ctas = @intCast(@min(try cuda.ovn.int("L2PF_CTAS", 2), 1024));
        o.pace_gbps = @intCast(@min(try cuda.ovn.int("L2PF_GBPS", 150), 100000));
        if (o.mb == 0 or o.ctas == 0) return error.BadOvnKnob;
        return o;
    }
};

/// A layer's sites: n before gather_b (the next mixer's input), f before gather_a (the ffn's head), o (KDA's chain).
pub const Site = enum(u2) { n, f, o };

/// One site's rows of the table and their bytes.
pub const Span = struct { at: u32 = 0, rows: u32 = 0, bytes: u64 = 0 };

const max_sources = 8;

const Sources = struct {
    ptrs: [max_sources]u64 = undefined,
    len: usize = 0,

    fn add(s: *Sources, p: u64) void {
        if (p != 0) {
            s.ptrs[s.len] = p;
            s.len += 1;
        }
    }

    fn lane(s: *Sources, q: cuda.qmmf.Weight) void {
        s.add(q.scales);
        s.add(q.codes);
    }

    fn proj(s: *Sources, w: projection.Proj) void {
        switch (w) {
            .bf16 => |p| s.add(p),
            .fp8g, .mxfp8 => |q| s.lane(q),
        }
    }
};

/// What site `site` of layer `l` reads first, lane scales before codes; SHARED_FIRST: site f adds the shared expert.
fn sources(l: weights.Layer, site: Site, shared_first: bool) Sources {
    var s: Sources = .{};
    switch (site) {
        .n => switch (l.mixer) {
            .kda => |k| switch (k.proj) {
                .bf16 => |p| s.add(p),
                .split => |sp| {
                    s.lane(sp.qkv);
                    s.add(sp.rest);
                },
            },
            .mla => |a| {
                s.proj(a.proj);
                s.proj(a.q_b);
            },
        },
        .f => switch (l.ffn) {
            .moe => |m| {
                s.add(m.router);
                if (shared_first) { // l2pf.py:252-254's site a, read now before the routed experts evict it
                    s.proj(m.shared_gu);
                    s.proj(m.shared_down);
                }
            },
            .dense => |m| switch (m.w) {
                .nvfp4 => |x| {
                    s.lane(x.gate);
                    s.lane(x.up);
                },
                .bf16 => |x| s.add(x.gu),
                .mxfp8 => |x| s.lane(x.gu),
            },
        },
        .o => switch (l.mixer) {
            .kda => |k| s.proj(k.o),
            .mla => {},
        },
    }
    return s;
}

/// l2pf.take: the bytes of each of `sizes` in order within `budget`, 16-byte multiples, the last one's head, 0 past it.
pub fn take(sizes: []const u64, budget: u64, out: []u64) void {
    var left = budget;
    for (sizes, out) |s, *n| {
        n.* = if (left < 16) 0 else @min(left, s) / 16 * 16;
        left -= n.*;
    }
}

/// The bytes from `p` to the end of the weight allocation holding it (#1580 matched starts only), or null.
fn extent(bufs: []const cuda.DeviceBuffer, p: u64) ?u64 {
    for (bufs) |b| if (p >= b.ptr and p < b.ptr + b.len) return b.ptr + b.len - p;
    return null;
}

/// Every site's rows of (address, bytes) from `bufs`, the weights' allocations; false when a source lies in none.
pub fn table(gpa: std.mem.Allocator, bufs: []const cuda.DeviceBuffer, layers: []const weights.Layer, budget: u64, shared_first: bool, rows: *std.ArrayList([2]i64), spans: []Span) !bool {
    std.debug.assert(spans.len == layers.len * 3);
    for (layers, 0..) |_, j| for ([_]Site{ .n, .f, .o }) |site| {
        const sp = &spans[j * 3 + @backingInt(site)];
        sp.* = .{ .at = @intCast(rows.items.len) };
        if (site == .n and j + 1 == layers.len) continue; // the last layer's next is the head
        const src = sources(layers[if (site == .n) j + 1 else j], site, shared_first);
        var sizes: [max_sources]u64 = undefined;
        for (src.ptrs[0..src.len], sizes[0..src.len]) |p, *n| n.* = extent(bufs, p) orelse {
            std.log.warn("TF_GLM_OVN_L2PF_PACED: layer {d}'s site {t} reads 0x{x}, in no weight buffer; prefetch off", .{ j, site, p });
            return false;
        };
        var got: [max_sources]u64 = undefined;
        take(sizes[0..src.len], budget, got[0..src.len]);
        for (src.ptrs[0..src.len], got[0..src.len]) |p, b| if (b > 0) {
            try rows.append(gpa, .{ @intCast(p), @intCast(b) });
            sp.rows += 1;
            sp.bytes += b;
        };
    };
    return true;
}

/// DeepSeek V4.1's launch: 32 KiB pieces, segments over 128 threads a CTA, paced as one piece per `chunk / rate` ns.
const chunk: u32 = 32 << 10;

/// The prefetch's own stream, events, kernels and table; the main stream forks at each site and joins once a window.
pub const Paced = struct {
    gpa: std.mem.Allocator,
    o: Options,
    side: cuda.Stream,
    fork_ev: cuda.Event,
    join_ev: cuda.Event,
    mods: [2]cuda.Module,
    segments: cuda.Function,
    paced: cuda.Function,
    rows: cuda.DeviceBuffer,
    spans: []Span,

    /// Built after the weights load and tile (tiling moves linears); null when the knob is off or a source is unknown.
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, ours_on: bool) !?Paced {
        const o = Options.read() catch |err| {
            std.log.err("glm: TF_GLM_OVN_L2PF_PACED, _MB, _CTAS or _GBPS: a whole number (MB and CTAS 1 or more)", .{});
            return err;
        };
        if (o.streams == 0) return null;
        if (ours_on) {
            std.log.err("glm: TF_GLM_OVN_L2PF_PACED and TF_GLM_L2PF both prefetch beside decode windows: set one", .{});
            return error.BadOvnKnob;
        }
        const spans = try gpa.alloc(Span, w.layers.len * 3);
        errdefer gpa.free(spans);
        var host: std.ArrayList([2]i64) = .empty;
        defer host.deinit(gpa);
        const first = @import("cuda_moe.zig").Ovn.read().shared_first; // the MoE's knob, read its way
        if (!try table(gpa, w.buffers.items, w.layers, @as(u64, o.mb) << 20, first, &host, spans)) {
            gpa.free(spans);
            return null;
        }
        if (host.items.len == 0) return error.NoPrefetchRows;
        var total: u64 = 0;
        for (spans) |sp| total += sp.bytes;
        std.log.info("TF_GLM_OVN_L2PF_PACED: {d} rows, {d} MiB a decode window of at most {d} streams, {d} CTAs a site, gathers' sites at {d} GB/s", .{ host.items.len, total >> 20, o.streams, o.ctas, o.pace_gbps });
        var rows = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host.items));
        errdefer rows.free();
        var mods: [2]cuda.Module = .{ try cuda.Module.load(d, cuda.kernels.l2pf), undefined };
        errdefer mods[0].unload();
        mods[1] = try cuda.Module.load(d, cuda.kernels.l2pace);
        errdefer mods[1].unload();
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork_ev = try cuda.Event.init(d, false);
        errdefer fork_ev.deinit();
        var join_ev = try cuda.Event.init(d, false);
        errdefer join_ev.deinit();
        const segments = try mods[0].function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij");
        const paced = try mods[1].function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        return .{ .gpa = gpa, .o = o, .side = side, .fork_ev = fork_ev, .join_ev = join_ev, .mods = mods, .segments = segments, .paced = paced, .rows = rows, .spans = spans };
    }

    /// After the main stream synchronized (Engine.deinit): the side stream drains first (Stream.deinit synchronizes).
    pub fn deinit(p: *Paced) void {
        p.side.deinit();
        p.fork_ev.deinit();
        p.join_ev.deinit();
        p.rows.free();
        for (&p.mods) |*m| m.unload();
        p.gpa.free(p.spans);
        p.* = undefined;
    }

    /// The prefetch when this forward is a decode window of this process's one rank with at most `streams` streams.
    pub fn gate(p: *const ?Paced, ranks: usize, prefill: bool, segs: usize) ?*const Paced {
        const x = if (p.*) |*v| v else return null;
        return if (ranks == 1 and !prefill and @max(1, segs) <= x.o.streams) x else null;
    }

    /// Site `site` of layer `j` read into L2 on the side stream from this point of `main` on.
    pub fn fork(p: *const Paced, main: cuda.Stream, j: usize, site: Site) !void {
        if (j * 3 >= p.spans.len) return;
        const sp = p.spans[j * 3 + @backingInt(site)];
        if (sp.rows == 0) return;
        try p.fork_ev.record(main);
        try p.side.wait(p.fork_ev);
        var a: cuda.Args = .{};
        a.add(p.rows.ptr + @as(u64, sp.at) * 16);
        a.add(@as(c_int, @intCast(sp.rows)));
        a.add(chunk);
        if (site != .o and p.o.pace_gbps > 0) {
            a.add(@as(u64, @max(1, (chunk + p.o.pace_gbps / 2) / p.o.pace_gbps))); // GB/s is bytes a ns
            a.add(@as(u64, 0));
            return cuda.launch.launch(p.paced, .{ .grid = .{ .x = p.o.ctas }, .block = .{ .x = 32 } }, p.side, &a);
        }
        const pieces = std.math.divCeil(u64, sp.bytes, chunk) catch unreachable;
        const grid: u32 = @intCast(std.math.clamp(std.math.divCeil(u64, pieces, 128) catch unreachable, 1, p.o.ctas));
        try cuda.launch.launch(p.segments, .{ .grid = .{ .x = grid }, .block = .{ .x = 128 } }, p.side, &a);
    }

    /// A window's end: `main` waits for every site forked on it, so nothing is in flight past the window.
    pub fn join(p: *const Paced, main: cuda.Stream) !void {
        try p.join_ev.record(p.side);
        try main.wait(p.join_ev);
    }
};

/// Site o's place in a KDA block: between its projections and its chain.
pub const At = struct {
    l2: *const Paced,
    layer: usize,

    pub fn fork(at: At, main: cuda.Stream) !void {
        return at.l2.fork(main, at.layer, .o);
    }
};

test "take: each size in order within the budget, in 16-byte multiples, the last one's head and nothing past it" {
    const t = std.testing;
    var out: [3]u64 = undefined;
    take(&.{ 1 << 20, 34 << 20, 2 << 20 }, 12 << 20, &out);
    try t.expectEqual([3]u64{ 1 << 20, 11 << 20, 0 }, out);
    take(&.{ 1000, 50, 7 }, 4096, &out);
    try t.expectEqual([3]u64{ 992, 48, 0 }, out);
    take(&.{ 5 << 20, 1 << 20, 1 << 20 }, (4 << 20) + 8, &out);
    try t.expectEqual([3]u64{ 4 << 20, 0, 0 }, out);
}

test "the table: each site's sources in its matmuls' order, site n from the next layer; an unknown buffer turns it off" {
    const t = std.testing;
    const kda_block = @import("cuda_kda_block.zig");
    const lane: cuda.qmmf.Weight = .{ .mode = .mxfp8, .codes = 0x2000_0000, .scales = 0x1000_0000, .n = 64, .k = 64, .npad = 64 };
    var k: kda_block.Layer = undefined;
    k.proj = .{ .split = .{ .qkv = lane, .rest = 0x3000_0000 } };
    k.o = .{ .bf16 = 0x4000_0000 };
    var d: weights.Layer = undefined; // layer 0: DSA and a dense bf16 MLP
    d.mixer = .{ .mla = undefined };
    d.mixer.mla.proj = .{ .bf16 = 0x6000_0000 };
    d.mixer.mla.q_b = .{ .bf16 = 0x7000_0000 };
    d.ffn = .{ .dense = undefined };
    d.ffn.dense.w = .{ .bf16 = .{ .gu = 0x9000_0000, .down = 0xa000_0000 } };
    var e: weights.Layer = undefined; // layer 1: KDA on the lane and MoE
    e.mixer = .{ .kda = k };
    e.ffn = .{ .moe = undefined };
    e.ffn.moe.router = 0x5000_0000;
    const MiB = 1 << 20;
    var bufs = [_]cuda.DeviceBuffer{
        .{ .d = undefined, .ptr = 0x1000_0000, .len = 1 * MiB },
        .{ .d = undefined, .ptr = 0x2000_0000, .len = 34 * MiB },
        .{ .d = undefined, .ptr = 0x3000_0000, .len = 2 * MiB },
        .{ .d = undefined, .ptr = 0x4000_0000, .len = 23 * MiB },
        .{ .d = undefined, .ptr = 0x5000_0000, .len = 2 * MiB },
        .{ .d = undefined, .ptr = 0x6000_0000, .len = 16 * MiB },
        .{ .d = undefined, .ptr = 0x7000_0000, .len = 17 * MiB },
        .{ .d = undefined, .ptr = 0x9000_0000, .len = 34 * MiB },
    };
    var rows: std.ArrayList([2]i64) = .empty;
    defer rows.deinit(t.allocator);
    var spans: [6]Span = undefined;
    try t.expect(try table(t.allocator, &bufs, &.{ d, e }, 12 * MiB, false, &rows, &spans));
    const want = [_][2]i64{
        .{ 0x1000_0000, 1 * MiB }, // layer 0's n: layer 1's qkv scales whole, then its codes' head
        .{ 0x2000_0000, 11 * MiB },
        .{ 0x9000_0000, 12 * MiB }, // layer 0's f: its gate and up's head
        .{ 0x5000_0000, 2 * MiB }, // layer 1's f: its router; its n would be the head's, its o is KDA's o
        .{ 0x4000_0000, 12 * MiB },
    };
    try t.expectEqualSlices([2]i64, &want, rows.items);
    try t.expectEqualSlices(Span, &.{
        .{ .at = 0, .rows = 2, .bytes = 12 * MiB },
        .{ .at = 2, .rows = 1, .bytes = 12 * MiB },
        .{ .at = 3 }, // DSA: no site o
        .{ .at = 3 },
        .{ .at = 3, .rows = 1, .bytes = 2 * MiB },
        .{ .at = 4, .rows = 1, .bytes = 12 * MiB },
    }, &spans);
    rows.clearRetainingCapacity();
    try t.expect(!try table(t.allocator, bufs[1..], &.{ d, e }, 12 * MiB, false, &rows, &spans));
}

test "TF_GLM_OVN_SHARED_FIRST: site f takes the router, then the shared expert's gate/up and down within the budget" {
    const t = std.testing;
    const MiB = 1 << 20;
    var e: weights.Layer = undefined; // one KDA layer with a MoE: site n is the head's, site o KDA's o
    e.mixer = .{ .kda = undefined };
    e.mixer.kda.o = .{ .bf16 = 0x4000_0000 };
    e.ffn = .{ .moe = undefined };
    e.ffn.moe.router = 0x5000_0000;
    e.ffn.moe.shared_gu = .{ .mxfp8 = .{ .mode = .mxfp8, .codes = 0xc000_0000, .scales = 0xb000_0000, .n = 64, .k = 64, .npad = 64 } };
    e.ffn.moe.shared_down = .{ .bf16 = 0xd000_0000 };
    const bufs = [_]cuda.DeviceBuffer{
        .{ .d = undefined, .ptr = 0x4000_0000, .len = 23 * MiB },
        .{ .d = undefined, .ptr = 0x5000_0000, .len = 2 * MiB },
        .{ .d = undefined, .ptr = 0xb000_0000, .len = MiB / 2 },
        .{ .d = undefined, .ptr = 0xc000_0000, .len = 6 * MiB },
        .{ .d = undefined, .ptr = 0xd000_0000, .len = 6 * MiB },
    };
    var rows: std.ArrayList([2]i64) = .empty;
    defer rows.deinit(t.allocator);
    var spans: [3]Span = undefined;
    try t.expect(try table(t.allocator, &bufs, &.{e}, 12 * MiB, true, &rows, &spans));
    const want = [_][2]i64{
        .{ 0x5000_0000, 2 * MiB }, // site f: the router whole
        .{ 0xb000_0000, MiB / 2 }, // the shared gate/up's scales, then its codes, whole
        .{ 0xc000_0000, 6 * MiB },
        .{ 0xd000_0000, 7 * MiB / 2 }, // the shared down's head: what the budget leaves
        .{ 0x4000_0000, 12 * MiB }, // site o: KDA's o
    };
    try t.expectEqualSlices([2]i64, &want, rows.items);
    try t.expectEqualSlices(Span, &.{ .{ .at = 0 }, .{ .at = 0, .rows = 4, .bytes = 12 * MiB }, .{ .at = 4, .rows = 1, .bytes = 12 * MiB } }, &spans);
    rows.clearRetainingCapacity(); // knob off: the router alone, as before
    try t.expect(try table(t.allocator, &bufs, &.{e}, 12 * MiB, false, &rows, &spans));
    try t.expectEqualSlices([2]i64, &.{ .{ 0x5000_0000, 2 * MiB }, .{ 0x4000_0000, 12 * MiB } }, rows.items);
}

test "a source inside an allocation reads to its end (#1580 matched starts only)" {
    const bufs = [_]cuda.DeviceBuffer{.{ .d = undefined, .ptr = 0x1000, .len = 0x100 }};
    try std.testing.expectEqual(@as(?u64, 0x40), extent(&bufs, 0x10c0));
    try std.testing.expectEqual(@as(?u64, null), extent(&bufs, 0x1100));
}

test "gate: off when unset, on a decode window of at most N streams of one rank" {
    const t = std.testing;
    const none: ?Paced = null;
    try t.expect(Paced.gate(&none, 1, false, 0) == null);
    var v: Paced = undefined;
    v.o = .{ .streams = 2 };
    const p: ?Paced = v;
    try t.expect(Paced.gate(&p, 1, false, 0) != null);
    try t.expect(Paced.gate(&p, 1, false, 2) != null);
    try t.expect(Paced.gate(&p, 1, false, 3) == null);
    try t.expect(Paced.gate(&p, 1, true, 0) == null);
    try t.expect(Paced.gate(&p, 2, false, 0) == null);
}
