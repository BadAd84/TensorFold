//! The EXL3 routed experts (exl3/experts.cu's device code): the group, the grouped GEMMs, the epilogues.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const exl3 = @import("exl3.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const DType = exl3.DType;
pub const Codebook = exl3.Codebook;
pub const prompt = @import("exl3_prompt.zig");

/// One layer's routed experts (experts.py's Exl3RoutedExperts): per-expert trellis pointers, stacked scales, the clamp.
pub const Layer = struct {
    gate: u64, // int64 [E], one trellis pointer per expert
    up: u64,
    down: u64,
    gate_k2: u64, // int32 [E], half-bits a value (2 * bits)
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    experts: usize, // E
    dims: usize, // D (model width)
    width: usize, // I (expert width)
    codebook: Codebook,
    uniform_k2: u32 = 0, // nonzero only when the loader verified this half-bit width for every projection
    shared_input: bool = false, // the loader verified identical gate/up input scales across every expert
    gu_stride: usize = 0, // gate/up trellis row stride in tiles (0: contiguous per projection)
    limit: f32 = 0, // the SwiGLU clamp (0: none)
    act_mode: i32 = 0, // 0: GLM's bf16 roundings (the route RED-SNOW runs), 1: fp32
    gate_base: u64 = 0, // every expert's gate trellis back to back, expert e at e * expertWords (0: not known)
    up_base: u64 = 0,
    down_base: u64 = 0,
};

/// The TF_GLM_OVN_X3_* decode variants (GLM's loader reads them; all off is run() itself). Every one keeps the bits.
pub const Ovn = struct {
    loads: Loads = .off, // X3_NC: 1, 2 (16-byte staged loads 1 or 2 k steps ahead), w (one word a lane plus a shuffle)
    fuse_down: bool = false, // X3_FUSEDOWN: the down kernel writes the slot rows itself (fusedScratch, slots 9)
    xrow: bool = false, // X3_XROW: one gate/up input rotation a row when every expert shares it
    pub const Loads = enum { off, nc1, nc2, word };
    pub fn any(o: Ovn) bool {
        return o.loads != .off or o.fuse_down or o.xrow;
    }
};

/// Words of one expert's trellis of a K x N projection (the loader's contiguous [K/16][N/16][32] layout).
fn expertWords(k: usize, n: usize) i64 {
    return @intCast((k / 16) * (n / 16) * 32);
}

/// What one projection launches (experts.py default_config): the 8-tile 4-warp K splits, shape-fixed; n % 128 required.
pub fn plan(k: usize, n: usize, gateup: bool) !u32 {
    if (forcedSplit(k, n)) |sk| return sk;
    const cands = [_]usize{ if (gateup) 4 else 1, 2, 1 };
    for (cands) |sk| {
        if (k % (16 * sk * 4) == 0 and n % 128 == 0) return @intCast(sk);
    }
    return error.TileNotBuilt;
}

/// TF_EXL3_SPLITK forces both projections' K split when the shape holds (an A/B knob; unset: the built
/// order). The kernel takes `sk` at launch, so any divisor of the K tiles works; the checks mirror plan's.
/// TF_EXL3_DEC_ORDER: 1 launches each k slice's 128-column blocks before the items, for locality in the
/// expert stream. The values are identical either way; only the order the blocks reach the SMs changes.
fn decOrder() u32 {
    const v = std.c.getenv("TF_EXL3_DEC_ORDER") orelse return 0;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    return if (std.mem.eql(u8, t, "1")) 1 else 0;
}

var splitk_env: ?u8 = null; // parsed once: null unset, 0 invalid-and-ignored, else the forced split
fn forcedSplit(k: usize, n: usize) ?u8 {
    if (splitk_env == null) {
        const v = if (std.c.getenv("TF_EXL3_SPLITK")) |p| std.mem.span(p) else "";
        splitk_env = std.fmt.parseInt(u8, v, 10) catch b: {
            if (v.len != 0) std.log.err("TF_EXL3_SPLITK: a K split 1 to 16, not \"{s}\"; ignored", .{v});
            break :b 0;
        };
    }
    const sk = splitk_env orelse return null;
    if (sk == 0 or sk > 16) return null;
    return if (k % (16 * @as(usize, sk) * 4) == 0 and n % 128 == 0) sk else null;
}

/// TF_EXL3_PIPE=1 walks the trellis with the software-pipelined kernel (the next k tile's words load while
/// the current tile's decode and mma run; bit-identical). Unset or 0: the plain walker.
var pipe_env: ?bool = null;
fn pipe() bool {
    if (pipe_env == null) {
        const v = if (std.c.getenv("TF_EXL3_PIPE")) |p| std.mem.span(p) else "";
        if (v.len != 0 and !std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "1") and !std.mem.eql(u8, v, "true"))
            std.log.err("TF_EXL3_PIPE: 0 or 1, not \"{s}\"; ignored", .{v});
        pipe_env = std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true");
    }
    return pipe_env.?;
}

/// One call's buffers (exl3/experts.py Scratch), laid out in scratch()'s order, each 256-byte aligned.
pub const Scratch = struct {
    xg: u64, // fp16 [P, D], the rotated gate inputs
    xu: u64, // fp16 [P, D], the rotated up inputs
    xd: u64, // fp16 [P, I], the rotated gate and up outputs (the down inputs)
    z: u64, // fp32, gate and up write 2 * sk_gu * P * I partials, down sk_d * P * D
    y: u64, // fp32 [P, D], the per-slot outputs
    ids: u64, // int32 [min(P, E)], distinct routed experts in id order
    count: u64, // int32 [1]
    members: u64, // int32 [min(P, E), rows], row * 32 + slot, -1 after the last member
    rows: usize, // the row capacity the buffers hold
    slots: usize, // slots a row, at most 32
};

/// The byte size of the scratch buffers for a layer and row capacity, or error.TileNotBuilt when plan() fails.
pub fn scratchBytes(l: Layer, rows: usize, slots: usize) !usize {
    return layoutBytes(l, rows, slots, l.dims * 4);
}

/// TF_GLM_OVN_X3_FUSEDOWN's scratch: scratchBytes() without y, which the fused down kernel never writes.
pub fn fusedBytes(l: Layer, rows: usize, slots: usize) !usize {
    return layoutBytes(l, rows, slots, 0);
}

/// Whether the down projection takes TF_GLM_OVN_X3_FUSEDOWN: mcg and one K split (TF_EXL3_SPLITK may force more).
pub fn fusable(l: Layer) bool {
    return l.codebook == .mcg and (plan(l.width, l.dims, false) catch return false) == 1;
}

fn layoutBytes(l: Layer, rows: usize, slots: usize, y_pair: usize) !usize {
    const g_sk = try plan(l.dims, l.width, true);
    const d_sk = try plan(l.width, l.dims, false);
    const p = rows * slots;
    const maxu = @min(p, l.experts);
    return align256(p * l.dims * 2) + align256(p * l.dims * 2) + align256(p * l.width * 2) +
        align256(@max(2 * g_sk * l.width, d_sk * l.dims) * p * 4) + align256(p * y_pair) +
        align256(maxu * 4) + 256 + align256(maxu * rows * 4);
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// The pointers into one allocation of scratchBytes(), or error.TileNotBuilt when plan() fails.
pub fn scratch(base: u64, l: Layer, rows: usize, slots: usize) !Scratch {
    return carve(base, l, rows, slots, l.dims * 4);
}

/// The pointers into one allocation of fusedBytes(); y is 0 (the fused down writes the caller's slot rows).
pub fn fusedScratch(base: u64, l: Layer, rows: usize, slots: usize) !Scratch {
    var sc = try carve(base, l, rows, slots, 0);
    sc.y = 0;
    return sc;
}

fn carve(base: u64, l: Layer, rows: usize, slots: usize, y_pair: usize) !Scratch {
    const p = rows * slots;
    const g_sk = try plan(l.dims, l.width, true);
    const d_sk = try plan(l.width, l.dims, false);
    const maxu = @min(p, l.experts);
    var at = base;
    const take = struct {
        fn f(a: *u64, bytes: usize) u64 {
            const r = a.*;
            a.* = r + align256(bytes);
            return r;
        }
    }.f;
    const xg = take(&at, p * l.dims * 2);
    const xu = take(&at, p * l.dims * 2);
    const xd = take(&at, p * l.width * 2);
    const z = take(&at, @max(2 * g_sk * l.width, d_sk * l.dims) * p * 4);
    const y = take(&at, p * y_pair);
    const ids = take(&at, maxu * 4);
    const count = take(&at, 4);
    const members = take(&at, maxu * rows * 4);
    return .{ .xg = xg, .xu = xu, .xd = xd, .z = z, .y = y, .ids = ids, .count = count, .members = members, .rows = rows, .slots = slots };
}

/// The loaded kernels.
pub const Experts = struct {
    mod: Module,
    group: Function,
    rot_in: Function,
    grouped: [3]Function, // one per codebook
    grouped_pipe: [3]Function, // the software-pipelined walkers (TF_EXL3_PIPE=1), bit-identical
    gateup: Function,
    down_combine: Function,
    prompt: prompt.Kernels = undefined,
    plan_rotate: Function = undefined,
    plan_finish: Function = undefined,
    ovn_walk: [2]Function = undefined, // TF_GLM_OVN_X3_*: the mcg walker with the variants' tail; [1] one word a lane
    ovn_nc: [2]Function = undefined, // TF_GLM_OVN_X3_NC=1|2: 16-byte staged trellis loads 1 or 2 k steps ahead
    ovn_rot_rows: Function = undefined, // TF_GLM_OVN_X3_XROW: rot_in once a row

    pub fn load(d: *const Driver) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var e: Experts = .{
            .mod = try Module.load(d, kernels.exl3_experts),
            .group = undefined,
            .rot_in = undefined,
            .grouped = undefined,
            .grouped_pipe = undefined,
            .gateup = undefined,
            .down_combine = undefined,
        };
        errdefer e.mod.unload();
        e.prompt = try prompt.Kernels.resolve(e.mod);
        e.plan_rotate = try e.mod.function("tf_exl3_plan_rotate");
        e.plan_finish = try e.mod.function("tf_exl3_plan_finish");
        e.group = try e.mod.function("tf_exl3_group");
        try e.group.allowDynamicShared(64 * 1024);
        e.rot_in = try e.mod.function("tf_exl3_experts_rot_in");
        e.grouped = .{
            try e.mod.function("tf_exl3_grouped_0"),
            try e.mod.function("tf_exl3_grouped_1"),
            try e.mod.function("tf_exl3_grouped_2"),
        };
        e.grouped_pipe = .{
            try e.mod.function("tf_exl3_grouped_p0"),
            try e.mod.function("tf_exl3_grouped_p1"),
            try e.mod.function("tf_exl3_grouped_p2"),
        };
        e.gateup = try e.mod.function("tf_exl3_gateup_epilogue");
        e.down_combine = try e.mod.function("tf_exl3_down_combine");
        e.ovn_walk = .{ try e.mod.function("tf_exl3_ovn_walk"), try e.mod.function("tf_exl3_ovn_walkw") };
        e.ovn_nc = .{ try e.mod.function("tf_exl3_ovn_nc1"), try e.mod.function("tf_exl3_ovn_nc2") };
        e.ovn_rot_rows = try e.mod.function("tf_exl3_ovn_rot_rows");
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    /// Per-slot FP32 outputs after the same BF16-up contract; the caller owns its routing-weight reduction.
    pub fn runSlots(e: *const Experts, s: Stream, l: Layer, x: u64, stride: usize, dtype: DType, pick: u64, rows: usize, sc: Scratch) !void {
        try e.run(s, l, x, stride, dtype, pick, 0, rows, 0, sc);
    }

    /// The routed experts of `rows` rows: group, rotate in, grouped gate/up, SwiGLU, grouped down, wts-weighted sum.
    pub fn run(e: *const Experts, s: Stream, l: Layer, x: u64, x_stride: usize, x_dtype: DType, pick: u64, wts: u64, rows: usize, out: u64, sc: Scratch) !void {
        try e.runWith(s, l, x, x_stride, x_dtype, pick, wts, rows, out, sc, .{}, 0);
    }

    /// The TF_GLM_OVN_X3_* kernel of one grouped launch, or null for run()'s own (the variants are mcg only).
    fn ovnKernel(e: *const Experts, l: Layer, loads: Ovn.Loads, xrow: bool, fuse: bool, k: usize, sk: u32, ns: usize, b0: u64, b1: u64) ?Function {
        const four = l.uniform_k2 == 8;
        const steps = k / 16 / sk / 4; // k steps a warp
        switch (loads) {
            .nc1, .nc2 => {
                const pd: usize = if (loads == .nc1) 1 else 2;
                if (four and ns == 0 and b0 != 0 and b1 != 0 and steps % pd == 0) return e.ovn_nc[pd - 1];
            },
            .word => if (four) return e.ovn_walk[1],
            .off => {},
        }
        return if (xrow or fuse) e.ovn_walk[0] else null;
    }

    /// run() with the TF_GLM_OVN_X3_* variants `o` (all off: run()'s launches); a fused down writes `ey` [P][D].
    pub fn runWith(e: *const Experts, s: Stream, l: Layer, x: u64, x_stride: usize, x_dtype: DType, pick: u64, wts: u64, rows: usize, out: u64, sc: Scratch, o: Ovn, ey: u64) !void {
        const d = l.dims;
        const w = l.width;
        const slots = sc.slots;
        std.debug.assert(rows >= 1 and rows <= sc.rows);
        std.debug.assert(slots >= 1 and slots <= 32);
        std.debug.assert(l.experts >= 1 and l.experts <= 4096);
        const g_sk = try plan(d, w, true);
        const d_sk = try plan(w, d, false);
        const p = rows * slots;
        const maxu = @min(p, l.experts);
        const mt = (rows + 15) / 16;
        const i32d = struct {
            fn f(v: usize) i32 {
                return @intCast(v);
            }
        }.f;
        if (o.fuse_down and (!fusable(l) or ey == 0 or wts != 0)) return error.TileNotBuilt;
        const ov: Ovn = if (l.codebook == .mcg) o else .{}; // the variants' kernels are mcg only
        const xrow = ov.xrow and l.shared_input;
        const fuse = ov.fuse_down;

        var a: launch_.Args = .{};
        a.add(pick);
        a.add(sc.ids);
        a.add(sc.count);
        a.add(sc.members);
        a.add(i32d(rows));
        a.add(i32d(slots));
        a.add(i32d(l.experts));
        a.add(i32d(rows));
        try launch_.launch(e.group, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = @intCast(rows * slots * 4) }, s, &a);

        var r: launch_.Args = .{};
        r.add(x);
        r.add(i32d(x_stride));
        r.add(@as(i32, @backingInt(x_dtype)));
        r.add(pick);
        r.add(l.suh_g);
        r.add(l.suh_u);
        r.add(sc.xg);
        r.add(if (xrow) sc.xg else sc.xu);
        r.add(i32d(d));
        r.add(i32d(slots));
        r.add(i32d(l.experts));
        if (xrow) {
            try launch_.launch(e.ovn_rot_rows, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(d / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &r);
        } else try launch_.launch(e.rot_in, .{ .grid = .{ .x = @intCast(p), .y = @intCast(d / 128), .z = 2 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &r);

        var g: launch_.Args = .{};
        g.add(sc.xg);
        g.add(if (xrow) sc.xg else sc.xu);
        g.add(l.gate);
        g.add(l.up);
        g.add(l.gate_k2);
        g.add(l.up_k2);
        g.add(sc.ids);
        g.add(sc.count);
        g.add(sc.members);
        g.add(sc.z);
        g.add(i32d(d));
        g.add(i32d(w));
        g.add(i32d(p));
        g.add(i32d(g_sk));
        g.add(i32d(rows));
        g.add(i32d(slots));
        g.add(i32d(maxu));
        g.add(@as(i32, 2));
        g.add(i32d(l.gu_stride));
        g.add(@as(i32, 0));
        g.add(@as(i32, -1));
        g.add(i32d(decOrder()));
        const gk = e.ovnKernel(l, ov.loads, xrow, false, d, g_sk, l.gu_stride, l.gate_base, l.up_base);
        if (gk != null) addTail(&g, l.gate_base, l.up_base, expertWords(d, w), 0, 0, xrow, false);
        const ord_g: u32 = decOrder();
        const gx: u32 = @intCast(if (ord_g == 1) w / 128 else maxu);
        const gy: u32 = @intCast(if (ord_g == 1) maxu else w / 128);
        try launch_.launch(if (gk) |f| f else if (pipe()) e.grouped_pipe[@backingInt(l.codebook)] else e.grouped[@backingInt(l.codebook)], .{ .grid = .{ .x = gx, .y = gy, .z = @intCast(2 * g_sk * mt) }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &g);

        var ep: launch_.Args = .{};
        ep.add(sc.z);
        ep.add(pick);
        ep.add(l.svh_g);
        ep.add(l.svh_u);
        ep.add(l.suh_d);
        ep.add(sc.xd);
        ep.add(i32d(p));
        ep.add(i32d(w));
        ep.add(i32d(g_sk));
        ep.add(i32d(l.experts));
        ep.add(l.limit);
        ep.add(l.act_mode);
        try launch_.launch(e.gateup, .{ .grid = .{ .x = @intCast(p), .y = @intCast(w / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &ep);

        var dn: launch_.Args = .{};
        dn.add(sc.xd);
        dn.add(sc.xd);
        dn.add(l.down);
        dn.add(l.down);
        dn.add(l.down_k2);
        dn.add(l.down_k2);
        dn.add(sc.ids);
        dn.add(sc.count);
        dn.add(sc.members);
        dn.add(sc.z);
        dn.add(i32d(w));
        dn.add(i32d(d));
        dn.add(i32d(p));
        dn.add(i32d(d_sk));
        dn.add(i32d(rows));
        dn.add(i32d(slots));
        dn.add(i32d(maxu));
        dn.add(@as(i32, 1));
        dn.add(@as(i32, 0));
        dn.add(@as(i32, 0));
        dn.add(@as(i32, -1));
        dn.add(i32d(decOrder()));
        const dk = e.ovnKernel(l, ov.loads, false, fuse, w, d_sk, 0, l.down_base, l.down_base);
        if (dk != null) addTail(&dn, l.down_base, l.down_base, expertWords(w, d), l.svh_d, ey, false, fuse);
        const ord_d: u32 = decOrder();
        const dx: u32 = @intCast(if (ord_d == 1) d / 128 else maxu);
        const dy: u32 = @intCast(if (ord_d == 1) maxu else d / 128);
        try launch_.launch(if (dk) |f| f else if (pipe()) e.grouped_pipe[@backingInt(l.codebook)] else e.grouped[@backingInt(l.codebook)], .{ .grid = .{ .x = dx, .y = dy, .z = @intCast(d_sk * mt) }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &dn);
        if (fuse) return; // the down kernel wrote ey's slot rows: no Z, no down_combine

        var cb: launch_.Args = .{};
        cb.add(sc.z);
        cb.add(pick);
        cb.add(l.svh_d);
        cb.add(sc.y);
        cb.add(wts);
        cb.add(out);
        cb.add(i32d(p));
        cb.add(i32d(d));
        cb.add(i32d(d_sk));
        cb.add(i32d(l.experts));
        cb.add(i32d(slots));
        try launch_.launch(e.down_combine, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(d / 128), .z = 1 }, .block = .{ .x = @intCast(32 * slots), .y = 1, .z = 1 } }, s, &cb);
    }
};

/// OvnTail (exl3_ovn.cuh) after a grouped launch's GroupedArgs, in its field order.
fn addTail(a: *launch_.Args, tb0: u64, tb1: u64, ew: i64, svh: u64, y: u64, xrow: bool, fuse: bool) void {
    a.add(tb0);
    a.add(tb1);
    a.add(ew);
    a.add(svh);
    a.add(y);
    a.add(@as(i32, @intFromBool(xrow)));
    a.add(@as(i32, @intFromBool(fuse)));
}

test "the plan matches experts.py's for known shapes" {
    try std.testing.expectEqual(@as(u32, 4), try plan(4096, 2048, true)); // GLM_GATEUP
    try std.testing.expectEqual(@as(u32, 1), try plan(2048, 4096, false)); // GLM_DOWN
    try std.testing.expectEqual(@as(u32, 4), try plan(512, 256, true));
    try std.testing.expectEqual(@as(u32, 1), try plan(256, 512, false));
    try std.testing.expectError(error.TileNotBuilt, plan(300, 256, true));
}

test "the scratch layout is 256-aligned and covered by scratchBytes" {
    const l: Layer = .{
        .gate = 0,
        .up = 0,
        .down = 0,
        .gate_k2 = 0,
        .up_k2 = 0,
        .down_k2 = 0,
        .suh_g = 0,
        .suh_u = 0,
        .svh_g = 0,
        .svh_u = 0,
        .suh_d = 0,
        .svh_d = 0,
        .experts = 9,
        .dims = 512,
        .width = 256,
        .codebook = .mul1,
    };
    const bytes = try scratchBytes(l, 64, 8);
    var buf: [4096]u8 align(256) = undefined;
    const sc = try scratch(@intFromPtr(&buf), l, 64, 8);
    try std.testing.expectEqual(@as(u64, @intFromPtr(&buf)), sc.xg);
    try std.testing.expectEqual(@as(u64, 0), sc.xg % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.xu % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.xd % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.z % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.y % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.ids % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.count % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.members % 256);
    try std.testing.expectEqual(@as(usize, @intFromPtr(&buf) + bytes), sc.members + @as(usize, 9) * 64 * 4);
}

test "TF_GLM_OVN_X3_FUSEDOWN: nine slots without y fit in the eight-slot scratch, and y is never handed out" {
    for ([_][2]usize{ .{ 4096, 768 }, .{ 4096, 640 }, .{ 512, 256 } }) |shape| {
        var l: Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = 288, .dims = shape[0], .width = shape[1], .codebook = .mcg, .uniform_k2 = 8 };
        try std.testing.expect(fusable(l));
        for (1..65) |rows| try std.testing.expect(try fusedBytes(l, rows, 9) <= try scratchBytes(l, rows, 8));
        const sc = try fusedScratch(0x10000, l, 4, 9);
        try std.testing.expectEqual(@as(u64, 0), sc.y);
        try std.testing.expectEqual(@as(u64, 0x10000 + try fusedBytes(l, 4, 9)), sc.members + align256(36 * 4 * 4));
        l.codebook = .mul1;
        try std.testing.expect(!fusable(l));
    }
}
