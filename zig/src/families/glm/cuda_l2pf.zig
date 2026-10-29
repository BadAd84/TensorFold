//! TF_GLM_L2PF: a decode window's L2 weight prefetch on a side stream (glm_cuda's port of the glm5_next
//! family's l2pf, v1.8). While the main stream waits on an all-gather or runs latency-bound glue, a small
//! kernel asks the memory system to bring the weights the next kernels read into L2. It only loads or
//! prefetches, so no result changes. Three sites a layer: `a` before the mixer's all-gather (this layer's
//! FFN: its hyper-connection weights, norm, the router and shared expert, or a dense MLP), `f` before the
//! FFN's all-gather (the next layer's hyper-connection weights, norm and input projections; after the last
//! layer the final norm and the head), `o` before the mixer (its output projection). Each site takes up to
//! TF_GLM_L2PF_MB (default 8) MiB: the small tensors whole, then the leading bytes of the large ones (the
//! glm5_next original also cut 4-bit chunk heads for its Triton kit; glm_mm's tiled readers have no such
//! one-wave chunk shape, so this port spans only). The side stream forks from the main stream at a site (an
//! event) and is joined before a window ends.
const std = @import("std");
const cuda = @import("cuda");
const kernels = @import("cuda").kernels;
const wt = @import("cuda_weights.zig");
const cfg = @import("config.zig");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");

const align_b = 16; // cp.async.bulk wants 16-byte addresses and sizes
const piece: usize = 32 << 10; // the table's pieces are at most 32 KiB

pub const Settings = struct {
    mode: i32 = -1, // -1: off; 0 bulk, 1 lines, 2 touch
    mb: f64 = 8.0,
    sites: [3]bool = .{ true, true, true }, // a, f, o
    threads: usize = 128,

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        var s: Settings = .{};
        const v = get("TF_GLM_L2PF") orelse return s;
        if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return s;
        s.mode = if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "bulk") or v.len == 0) 0 else if (std.mem.eql(u8, v, "lines")) 1 else if (std.mem.eql(u8, v, "touch")) 2 else {
            problem.* = "TF_GLM_L2PF: 0, off, 1, bulk, lines or touch";
            return error.BadSetting;
        };
        if (get("TF_GLM_L2PF_MB")) |x| {
            s.mb = std.fmt.parseFloat(f64, x) catch {
                problem.* = "TF_GLM_L2PF_MB: MiB, at most 64";
                return error.BadSetting;
            };
            if (!(s.mb > 0 and s.mb <= 64)) {
                problem.* = "TF_GLM_L2PF_MB: MiB, at most 64";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_L2PF_SITES")) |x| {
            s.sites = .{ false, false, false };
            for (x) |ch| switch (ch) {
                'a' => s.sites[0] = true,
                'f' => s.sites[1] = true,
                'o' => s.sites[2] = true,
                else => {
                    problem.* = "TF_GLM_L2PF_SITES: a, f and o";
                    return error.BadSetting;
                },
            };
        }
        return s;
    }

    pub fn on(s: Settings) bool {
        return s.mode >= 0;
    }
};

const T = struct { ptr: u64, bytes: usize };

/// One site's tensors, prefetched small first within `budget`.
const Group = struct {
    ts: [10]T = undefined,
    n: usize = 0,

    fn add(g: *Group, t: ?T) void {
        const x = t orelse return;
        if (x.ptr == 0 or x.bytes == 0) return;
        for (g.ts[0..g.n]) |y| if (y.ptr == x.ptr) return; // each tensor once
        if (g.n == g.ts.len) return; // a site's worth of tensors is all this needs
        g.ts[g.n] = x;
        g.n += 1;
    }
};

pub const Site = struct { first: u32, count: u32, grid: u32 };

pub const Prefetch = struct {
    s: Settings,
    table: cuda.DeviceBuffer,
    sink: cuda.DeviceBuffer,
    mod: cuda.Module,
    f: cuda.Function,
    sites: [cfg.max_layers][3]?Site = @splat(@splat(null)),
    side: cuda.Stream,
    fork: cuda.Event,
    join_ev: cuda.Event,
    forked: bool = false,
    bytes: [3]usize = .{ 0, 0, 0 },
    count: [3]usize = .{ 0, 0, 0 },

    fn named(w: *const wt.Weights, ptr: u64) ?usize {
        for (w.named.items) |n| if (n.ptr == ptr) return n.len;
        return null;
    }

    fn whole(w: *const wt.Weights, g: *Group, ptr: u64) void {
        g.add(.{ .ptr = ptr, .bytes = named(w, ptr) orelse 0 });
    }

    /// A hidden-size norm, bf16 or kept fp32.
    fn norm(w: *const wt.Weights, g: *Group, n: glue.Norm) void {
        switch (n) {
            .bf16 => |x| whole(w, g, x),
            .f32 => |x| whole(w, g, x),
        }
    }
    /// A projection's codes and scale tables (bf16: the one buffer).
    fn proj(w: *const wt.Weights, g: *Group, p: projection.Proj) void {
        switch (p) {
            .bf16 => |x| whole(w, g, x),
            .fp8g, .mxfp8 => |q| {
                whole(w, g, q.codes);
                whole(w, g, q.scales);
            },
        }
    }

    /// Site `a`: this layer's FFN - hyper-connection weights, norm, the router and shared expert, or a dense MLP.
    fn ffnSite(w: *const wt.Weights, l: wt.Layer, gs: *std.ArrayList(Group), gpa: std.mem.Allocator) !void {
        var hc: Group = .{};
        whole(w, &hc, l.ffn_hc.fn_w);
        whole(w, &hc, l.ffn_hc.base);
        whole(w, &hc, l.ffn_hc.scale);
        try gs.append(gpa, hc);
        var n: Group = .{};
        norm(w, &n, l.post_norm);
        try gs.append(gpa, n);
        switch (l.ffn) {
            .moe => |m| {
                var r: Group = .{};
                whole(w, &r, m.router);
                whole(w, &r, m.bias);
                try gs.append(gpa, r);
                var a: Group = .{};
                proj(w, &a, m.shared_gu);
                try gs.append(gpa, a);
                var b: Group = .{};
                proj(w, &b, m.shared_down);
                try gs.append(gpa, b);
            },
            .dense => |m| switch (m.w) {
                .bf16 => |b| {
                    var a: Group = .{};
                    whole(w, &a, b.gu);
                    try gs.append(gpa, a);
                    var d: Group = .{};
                    whole(w, &d, b.down);
                    try gs.append(gpa, d);
                },
                .nvfp4 => |b| {
                    var a: Group = .{};
                    whole(w, &a, b.gate.codes);
                    whole(w, &a, b.gate.scales);
                    whole(w, &a, b.up.codes);
                    whole(w, &a, b.up.scales);
                    try gs.append(gpa, a);
                    var d: Group = .{};
                    whole(w, &d, b.down.codes);
                    whole(w, &d, b.down.scales);
                    try gs.append(gpa, d);
                },
                .mxfp8 => |b| {
                    var a: Group = .{};
                    whole(w, &a, b.gu.codes);
                    whole(w, &a, b.gu.scales);
                    try gs.append(gpa, a);
                    var d: Group = .{};
                    whole(w, &d, b.down.codes);
                    whole(w, &d, b.down.scales);
                    try gs.append(gpa, d);
                },
            },
        }
    }

    /// Site `f`: the next layer's hyper-connection weights, norm and input projections; after the last layer
    /// the final norm and the head.
    fn nextSite(w: *const wt.Weights, nxt: ?wt.Layer, gs: *std.ArrayList(Group), gpa: std.mem.Allocator) !void {
        const n = nxt orelse {
            var a: Group = .{};
            norm(w, &a, w.norm);
            try gs.append(gpa, a);
            var h: Group = .{};
            proj(w, &h, w.head);
            try gs.append(gpa, h);
            return;
        };
        var hc: Group = .{};
        whole(w, &hc, n.attn_hc.fn_w);
        whole(w, &hc, n.attn_hc.base);
        whole(w, &hc, n.attn_hc.scale);
        try gs.append(gpa, hc);
        var nm: Group = .{};
        norm(w, &nm, n.in_norm);
        try gs.append(gpa, nm);
        switch (n.mixer) {
            .kda => |k| {
                var a: Group = .{};
                switch (k.proj) {
                    .bf16 => |x| whole(w, &a, x),
                    .split => |sp| {
                        whole(w, &a, sp.qkv.codes);
                        whole(w, &a, sp.qkv.scales);
                        whole(w, &a, sp.rest);
                    },
                }
                try gs.append(gpa, a);
                var s: Group = .{};
                whole(w, &s, k.fb);
                whole(w, &s, k.gb);
                whole(w, &s, k.conv_w);
                whole(w, &s, k.a_log);
                whole(w, &s, k.dt_bias);
                whole(w, &s, k.norm_w);
                try gs.append(gpa, s);
            },
            .mla => |a| {
                var p: Group = .{};
                proj(w, &p, a.proj);
                try gs.append(gpa, p);
                var s: Group = .{};
                whole(w, &s, a.q_norm);
                whole(w, &s, a.kv_norm);
                whole(w, &s, a.index.gate);
                whole(w, &s, a.index.ln_w);
                whole(w, &s, a.index.ln_b);
                whole(w, &s, a.index.ape);
                try gs.append(gpa, s);
                var ix: Group = .{};
                proj(w, &ix, a.index.kw);
                proj(w, &ix, a.index.qb);
                whole(w, &ix, a.wk);
                try gs.append(gpa, ix);
            },
        }
    }

    /// Site `o`: the mixer's output projection (DSA's fired value half beside it).
    fn outSite(w: *const wt.Weights, l: wt.Layer, gs: *std.ArrayList(Group), gpa: std.mem.Allocator) !void {
        switch (l.mixer) {
            .kda => |k| {
                var g: Group = .{};
                proj(w, &g, k.o);
                try gs.append(gpa, g);
            },
            .mla => |a| {
                var v: Group = .{};
                whole(w, &v, a.wv);
                try gs.append(gpa, v);
                var g: Group = .{};
                proj(w, &g, a.o);
                try gs.append(gpa, g);
            },
        }
    }

    /// l2pf._span: the tensor's leading bytes (at most `left`), the start rounded down and the end up to 16.
    fn span(t: T, left: usize) ?T {
        const lo = t.ptr - t.ptr % align_b;
        var hi = @min(t.ptr + t.bytes, lo + left);
        hi += (align_b - hi % align_b) % align_b;
        return if (hi > lo) .{ .ptr = lo, .bytes = hi - lo } else null;
    }

    /// l2pf.ranges: (address, bytes) ranges of `groups` up to `budget` bytes, appended to `out`.
    fn ranges(gpa: std.mem.Allocator, groups: []const Group, budget: usize, out: *std.ArrayList(T)) !void {
        var left = budget;
        for (groups) |g0| {
            var g = g0;
            // within a group the small tensors first (a stable sort by size)
            std.mem.sort(T, g.ts[0..g.n], {}, struct {
                fn lt(_: void, a: T, b: T) bool {
                    return a.bytes < b.bytes;
                }
            }.lt);
            for (g.ts[0..g.n]) |t| {
                if (left < align_b) return;
                if (span(t, left)) |sp| {
                    try out.append(gpa, sp);
                    left -= sp.bytes;
                }
            }
        }
    }

    pub fn build(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wt.Weights, s: Settings) !Prefetch {
        var p: Prefetch = .{ .s = s, .table = undefined, .sink = undefined, .mod = try cuda.Module.load(d, kernels.glm_l2pf), .f = undefined, .side = undefined, .fork = undefined, .join_ev = undefined };
        errdefer p.mod.unload();
        p.f = try p.mod.function("tf_glm_l2pf");
        const budget: usize = @intFromFloat(s.mb * (1 << 20));
        var rows: std.ArrayList(i64) = .empty;
        errdefer rows.deinit(gpa);
        for (w.layers, 0..) |l, i| {
            const nxt: ?wt.Layer = if (i + 1 < w.layers.len) w.layers[i + 1] else null;
            inline for (0..3) |si| {
                if (s.sites[si]) {
                    var gs: std.ArrayList(Group) = .empty;
                    defer gs.deinit(gpa);
                    switch (si) {
                        0 => try ffnSite(w, l, &gs, gpa),
                        1 => try nextSite(w, nxt, &gs, gpa),
                        else => try outSite(w, l, &gs, gpa),
                    }
                    var rs: std.ArrayList(T) = .empty;
                    defer rs.deinit(gpa);
                    try ranges(gpa, gs.items, budget, &rs);
                    const first = rows.items.len / 2;
                    var total: usize = 0;
                    for (rs.items) |sp| {
                        total += sp.bytes;
                        var o: usize = 0;
                        while (o < sp.bytes) : (o += piece) {
                            try rows.append(gpa, @intCast(sp.ptr + o));
                            try rows.append(gpa, @intCast(@min(piece, sp.bytes - o)));
                        }
                    }
                    const count = rows.items.len / 2 - first;
                    if (count > 0) {
                        const per: usize = if (s.mode == 0) s.threads else s.threads / 32;
                        p.sites[l.index][si] = .{ .first = @intCast(first), .count = @intCast(count), .grid = @intCast(@max(1, @min(48, (count + per - 1) / per))) };
                        p.bytes[si] += total;
                        p.count[si] += 1;
                    }
                }
            }
        }
        if (rows.items.len == 0) try rows.appendSlice(gpa, &.{ 0, 0 });
        p.table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(rows.items));
        errdefer p.table.free();
        p.sink = try cuda.DeviceBuffer.alloc(d, 256);
        errdefer p.sink.free();
        try p.sink.fill8(0, null);
        p.side = try cuda.Stream.init(d, true);
        errdefer p.side.deinit();
        p.fork = try cuda.Event.init(d, false);
        errdefer p.fork.deinit();
        p.join_ev = try cuda.Event.init(d, false);
        return p;
    }

    pub fn deinit(p: *Prefetch) void {
        p.join_ev.deinit();
        p.fork.deinit();
        p.side.deinit();
        p.sink.free();
        p.table.free();
        p.mod.unload();
        p.* = undefined;
    }

    /// l2pf.Prefetch.site: fork the side stream at this point of `main` and prefetch the site's ranges there.
    pub fn site(p: *Prefetch, main: cuda.Stream, layer: u32, which: usize) !void {
        const st = p.sites[layer][which] orelse return;
        try p.fork.record(main);
        try p.side.wait(p.fork);
        var args: cuda.Args = .{};
        args.add(p.table.ptr);
        args.add(@as(c_int, @intCast(st.count)));
        args.add(p.s.mode);
        args.add(p.sink.ptr);
        try cuda.launch.launch(p.f, .{ .grid = .{ .x = st.grid }, .block = .{ .x = @intCast(p.s.threads) } }, p.side, &args);
        p.forked = true;
    }

    /// The main stream waits for the side stream (every fork before a window ends has rejoined).
    pub fn join(p: *Prefetch, main: cuda.Stream) !void {
        if (!p.forked) return;
        try p.join_ev.record(p.side);
        try main.wait(p.join_ev);
        p.forked = false;
    }
};
