//! glm-kda-seg (GPU): TF_GLM_OVN_SEG_KDA's segmented chains (tf_glm_kda_prep_seg, _step_seg, one tf_glm_kda_out;
//! tf_glm_kda_chain_seg) against the per-segment chains a shared window runs today, byte for byte: every output row,
//! every segment's new state and saved rows, at layer 1 of two (the table's layer strides), TP=3's 22 heads.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const H = 22;
const dk = 128;
const n = H * dk;
const p_stride = 3 * n + 2 * dk + H + 2; // the projection rows' width (padded past the stack, as TF_GLM_KDA_PAD)
const b_off = 3 * n + 2 * dk;
const layers = 2;
const li = 1;
const window_rows = 8;
const seg_rows = [_]usize{ 3, 1, 8, 5 };
const segs = seg_rows.len;
const rows = 17;
const conv_e = 3 * 3 * n; // a layer's conv window: 3 rows of q | k | v
const rec_e = H * dk * dk;
const kv_e = window_rows * H * dk;
const b_e = window_rows * H;

const Seg = extern struct { row0: i64, rows: i64, cs: u64, state_in: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };

/// One run's outputs: the window's rows, and each segment's states and saved rows (all layers, li's written).
const Outs = struct {
    out: cuda.DeviceBuffer,
    state: [segs]cuda.DeviceBuffer,
    k: [segs]cuda.DeviceBuffer,
    v: [segs]cuda.DeviceBuffer,
    g: [segs]cuda.DeviceBuffer,
    b: [segs]cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver) !Outs {
        var o: Outs = undefined;
        o.out = try cuda.DeviceBuffer.alloc(d, rows * n * 2);
        o.q = try cuda.DeviceBuffer.alloc(d, rows * n * 4);
        o.y = try cuda.DeviceBuffer.alloc(d, rows * n * 2);
        for (0..segs) |s| {
            o.state[s] = try cuda.DeviceBuffer.alloc(d, layers * rec_e * 4);
            o.k[s] = try cuda.DeviceBuffer.alloc(d, layers * kv_e * 4);
            o.v[s] = try cuda.DeviceBuffer.alloc(d, layers * kv_e * 2);
            o.g[s] = try cuda.DeviceBuffer.alloc(d, layers * kv_e * 4);
            o.b[s] = try cuda.DeviceBuffer.alloc(d, layers * b_e * 4);
            for ([_]cuda.DeviceBuffer{ o.state[s], o.k[s], o.v[s], o.g[s], o.b[s] }) |x| try x.fill8(0, null);
        }
        try o.out.fill8(0, null);
        return o;
    }

    fn deinit(o: *Outs) void {
        o.out.free();
        o.q.free();
        o.y.free();
        for (0..segs) |s| {
            for ([_]*cuda.DeviceBuffer{ &o.state[s], &o.k[s], &o.v[s], &o.g[s], &o.b[s] }) |x| x.free();
        }
    }
};

fn random(gpa: std.mem.Allocator, d: *const cuda.Driver, count: usize, comptime bf: bool, scale: f32, salt: u64) !cuda.DeviceBuffer {
    var x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (salt *% 0xD1B5_4A32_D192_ED03);
    const bytes = try gpa.alloc(u8, count * @as(usize, if (bf) 2 else 4));
    defer gpa.free(bytes);
    for (0..count) |i| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const v: f32 = scale * (@as(f32, @floatFromInt(x >> 40)) / 8388608.0 - 1.0);
        const b: u32 = @bitCast(v);
        if (bf) std.mem.writeInt(u16, bytes[2 * i ..][0..2], @intCast(b >> 16), .little) else std.mem.writeInt(u32, bytes[4 * i ..][0..4], b, .little);
    }
    return cuda.DeviceBuffer.fromHost(d, bytes);
}

fn same(gpu: check.Gpu, what: []const u8, a: cuda.DeviceBuffer, b: cuda.DeviceBuffer) !void {
    const x = try check.download(gpu, a);
    defer gpu.gpa.free(x);
    const y = try check.download(gpu, b);
    defer gpu.gpa.free(y);
    try check.sameBytes(what, y, x);
}

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var m = try cuda.Module.load(d, cuda.kernels.glm_kda);
    defer m.unload();
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var p = try random(gpa, d, rows * p_stride, true, 2.0, 1);
    defer p.free();
    var a = try random(gpa, d, rows * n, true, 2.0, 2);
    defer a.free();
    var g = try random(gpa, d, rows * n, true, 3.0, 3);
    defer g.free();
    var cw = try random(gpa, d, 3 * n * 4, false, 0.3, 4);
    defer cw.free();
    var a_log = try random(gpa, d, H, false, 1.5, 5);
    defer a_log.free();
    var dt = try random(gpa, d, n, false, 0.5, 6);
    defer dt.free();
    var norm_w = try random(gpa, d, dk, true, 1.0, 7);
    defer norm_w.free();
    var conv: [segs]cuda.DeviceBuffer = undefined;
    var state: [segs]cuda.DeviceBuffer = undefined;
    for (0..segs) |k| {
        conv[k] = try random(gpa, d, layers * conv_e, true, 1.0, 10 + k);
        state[k] = try random(gpa, d, layers * rec_e, false, 0.2, 20 + k);
    }
    defer for (0..segs) |k| {
        conv[k].free();
        state[k].free();
    };
    var table = try cuda.DeviceBuffer.alloc(d, segs * @sizeOf(Seg));
    defer table.free();
    for ([_]bool{ true, false }) |wide| {
        var want = try Outs.init(d);
        defer want.deinit();
        var got = try Outs.init(d);
        defer got.deinit();
        var t: [segs]Seg = undefined;
        var row0: usize = 0;
        for (seg_rows, 0..) |r, k| { // today's loop: each segment's chain on its own pointers, layer li
            const cs = conv[k].ptr + li * conv_e * 2;
            const sin = state[k].ptr + li * rec_e * 4;
            const sout = want.state[k].ptr + li * rec_e * 4;
            const sk = want.k[k].ptr + li * kv_e * 4;
            const sv = want.v[k].ptr + li * kv_e * 2;
            const sg = want.g[k].ptr + li * kv_e * 4;
            const sb = want.b[k].ptr + li * b_e * 4;
            const pr = p.ptr + row0 * p_stride * 2;
            try chainOne(m, s, wide, r, pr, a.ptr + row0 * n * 2, g.ptr + row0 * n * 2, cs, cw.ptr, sin, a_log.ptr, dt.ptr, norm_w.ptr, want.out.ptr + row0 * n * 2, sout, sk, sv, sg, sb, want.q.ptr, want.y.ptr);
            t[k] = .{ .row0 = @intCast(row0), .rows = @intCast(r), .cs = conv[k].ptr, .state_in = state[k].ptr, .state_out = got.state[k].ptr, .k = got.k[k].ptr, .v = got.v[k].ptr, .g = got.g[k].ptr, .b = got.b[k].ptr };
            row0 += r;
        }
        try table.uploadAsync(0, std.mem.sliceAsBytes(&t), s.handle);
        try chainSegs(m, s, wide, table.ptr, p.ptr, a.ptr, g.ptr, cw.ptr, a_log.ptr, dt.ptr, norm_w.ptr, got.out.ptr, got.q.ptr, got.y.ptr);
        try s.synchronize();
        try same(gpu, "kda-seg out", want.out, got.out);
        for (0..segs) |k| {
            try same(gpu, "kda-seg state", want.state[k], got.state[k]);
            try same(gpu, "kda-seg saved k", want.k[k], got.k[k]);
            try same(gpu, "kda-seg saved v", want.v[k], got.v[k]);
            try same(gpu, "kda-seg saved decay", want.g[k], got.g[k]);
            try same(gpu, "kda-seg saved beta", want.b[k], got.b[k]);
        }
    }
    check.pass("glm-kda-seg: {d} segments of {d} rows, wide and fused chains, equal the per-segment chains byte for byte", .{ segs, rows });
    try commits(gpu, m, s);
}

const KdaKeep = extern struct { keep: i64, pending: i64, state_in: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };
const ConvSeg = extern struct { conv: u64, proj: u64, keep: i64 };

/// TF_GLM_OVN_SEG_COMMIT's tf_glm_kda_replay_layers_seg and tf_glm_conv_shift_seg against the per-stream
/// tf_glm_kda_replay_layers and tf_glm_conv_shift (Forward.commit's), streams keeping fewer and all of their rows.
fn commits(gpu: check.Gpu, m: cuda.Module, s: cuda.Stream) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var gm = try cuda.Module.load(d, cuda.kernels.glm_glue);
    defer gm.unload();
    const keep = [_]i64{ 2, 5, 1 };
    const pending = [_]i64{ 5, 5, 8 };
    const k = keep.len;
    const c = 3 * n; // the conv window's channels
    const proj_layer = window_rows * p_stride;
    var ins: [k][5]cuda.DeviceBuffer = undefined; // state, saved k, v, decay, beta
    var outs: [2][k]cuda.DeviceBuffer = undefined; // the per-stream and segmented runs' new states
    var convs: [2][k]cuda.DeviceBuffer = undefined;
    var projs: [k]cuda.DeviceBuffer = undefined;
    var keeps: [k]KdaKeep = undefined;
    var shifts: [k]ConvSeg = undefined;
    for (0..k) |i| {
        ins[i] = .{ try random(gpa, d, layers * rec_e, false, 0.2, 40 + i), try random(gpa, d, layers * kv_e, false, 0.1, 50 + i), try random(gpa, d, layers * kv_e, true, 1.0, 60 + i), try random(gpa, d, layers * kv_e, false, 1.0, 70 + i), try random(gpa, d, layers * b_e, false, 1.0, 80 + i) };
        projs[i] = try random(gpa, d, layers * proj_layer, true, 1.0, 90 + i);
        for (0..2) |r| {
            outs[r][i] = try cuda.DeviceBuffer.alloc(d, layers * rec_e * 4);
            try outs[r][i].fill8(0, null);
            convs[r][i] = try random(gpa, d, layers * 3 * c, true, 1.0, 100 + i); // the same rows in both runs
        }
        keeps[i] = .{ .keep = keep[i], .pending = pending[i], .state_in = ins[i][0].ptr, .state_out = outs[1][i].ptr, .k = ins[i][1].ptr, .v = ins[i][2].ptr, .g = ins[i][3].ptr, .b = ins[i][4].ptr };
        shifts[i] = .{ .conv = convs[1][i].ptr, .proj = projs[i].ptr, .keep = keep[i] };
    }
    defer for (0..k) |i| {
        for (&ins[i]) |*x| x.free();
        projs[i].free();
        for (0..2) |r| {
            outs[r][i].free();
            convs[r][i].free();
        }
    };
    const h: c_int = H;
    for (0..k) |i| { // Forward.commit, a stream at a time
        if (keep[i] < pending[i]) {
            var a: cuda.Args = .{};
            a.add(h);
            a.add(ins[i][0].ptr);
            a.add(@as(usize, rec_e));
            for (ins[i][1..]) |x| a.add(x.ptr);
            a.add(@as(usize, kv_e));
            a.add(@as(usize, b_e));
            a.add(@as(c_int, @intCast(keep[i])));
            a.add(outs[0][i].ptr);
            try go(try m.function("tf_glm_kda_replay_layers"), .{ .x = layers * H }, 1024, s, &a);
        }
        var b: cuda.Args = .{};
        b.add(convs[0][i].ptr);
        b.add(projs[i].ptr);
        for ([_]i64{ keep[i], 3 * c, proj_layer, p_stride, c }) |v| b.add(v);
        try go(try gm.function("tf_glm_conv_shift"), .{ .x = layers, .y = (c + 255) / 256 }, 256, s, &b);
    }
    var table = try cuda.DeviceBuffer.alloc(d, k * (@sizeOf(KdaKeep) + @sizeOf(ConvSeg)));
    defer table.free();
    try table.uploadAsync(0, std.mem.sliceAsBytes(&keeps), s.handle);
    try table.uploadAsync(k * @sizeOf(KdaKeep), std.mem.sliceAsBytes(&shifts), s.handle);
    var a: cuda.Args = .{};
    a.add(table.ptr);
    a.add(h);
    for ([_]usize{ rec_e, kv_e, b_e }) |v| a.add(v);
    try go(try m.function("tf_glm_kda_replay_layers_seg"), .{ .x = layers * H, .y = k }, 1024, s, &a);
    var b: cuda.Args = .{};
    b.add(table.ptr + k * @sizeOf(KdaKeep));
    for ([_]i64{ 3 * c, proj_layer, p_stride, c }) |v| b.add(v);
    try go(try gm.function("tf_glm_conv_shift_seg"), .{ .x = layers, .y = (c + 255) / 256, .z = k }, 256, s, &b);
    try s.synchronize();
    for (0..k) |i| {
        try same(gpu, "seg-commit state", outs[0][i], outs[1][i]);
        try same(gpu, "seg-commit conv window", convs[0][i], convs[1][i]);
    }
    check.pass("glm-kda-seg: {d} streams' segmented commits equal the per-stream replay and conv shift", .{k});
}

const lower: f32 = -5.0;
const eps: f32 = 1e-6;

fn go(f: cuda.Function, grid: cuda.launch.Dim3, threads: u32, s: cuda.Stream, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, args);
}

/// One segment's chain as cuda_kda.chain launches it (wide: prep, step, out; else the fused kernel).
fn chainOne(m: cuda.Module, s: cuda.Stream, wide: bool, r: usize, pr: u64, a: u64, g: u64, cs: u64, cw: u64, sin: u64, a_log: u64, dt: u64, norm_w: u64, out: u64, sout: u64, sk: u64, sv: u64, sg: u64, sb: u64, q: u64, y: u64) !void {
    const h: c_int = H;
    if (!wide) {
        var x: cuda.Args = .{};
        x.add(h);
        x.add(pr);
        x.add(@as(c_int, p_stride));
        x.add(@as(c_int, b_off));
        x.add(a);
        x.add(@as(c_int, n));
        x.add(g);
        x.add(@as(c_int, n));
        for ([_]u64{ cs, cw, sin, a_log, dt, norm_w }) |v| x.add(v);
        x.add(eps);
        x.add(lower);
        x.add(@as(c_int, @intCast(r)));
        for ([_]u64{ out, sout, sk, sv, sg, sb }) |v| x.add(v);
        return go(try m.function("tf_glm_kda_chain"), .{ .x = H }, 1024, s, &x);
    }
    var x: cuda.Args = .{};
    x.add(h);
    x.add(pr);
    x.add(@as(c_int, p_stride));
    x.add(@as(c_int, b_off));
    x.add(a);
    x.add(@as(c_int, n));
    for ([_]u64{ cs, cw, a_log, dt }) |v| x.add(v);
    x.add(lower);
    for ([_]u64{ q, sk, sv, sg, sb }) |v| x.add(v);
    try go(try m.function("tf_glm_kda_prep"), .{ .x = @intCast(r), .y = H }, 512, s, &x);
    var z: cuda.Args = .{};
    z.add(h);
    for ([_]u64{ sin, q, sk, sv, sg, sb }) |v| z.add(v);
    z.add(@as(c_int, @intCast(r)));
    z.add(y);
    z.add(sout);
    try go(try m.function("tf_glm_kda_step"), .{ .x = H, .y = 4 }, 256, s, &z);
    var o: cuda.Args = .{};
    o.add(h);
    o.add(y);
    o.add(g);
    o.add(@as(c_int, n));
    o.add(norm_w);
    o.add(eps);
    o.add(out);
    try go(try m.function("tf_glm_kda_out"), .{ .x = @intCast(r), .y = H }, dk, s, &o);
}

/// Every segment at once as cuda_kda_seg.SegKda.chains launches them.
fn chainSegs(m: cuda.Module, s: cuda.Stream, wide: bool, table: u64, p: u64, a: u64, g: u64, cw: u64, a_log: u64, dt: u64, norm_w: u64, out: u64, q: u64, y: u64) !void {
    const h: c_int = H;
    if (!wide) {
        var x: cuda.Args = .{};
        x.add(table);
        for ([_]i64{ li, conv_e, rec_e, kv_e, b_e }) |v| x.add(v);
        x.add(h);
        x.add(p);
        x.add(@as(c_int, p_stride));
        x.add(@as(c_int, b_off));
        x.add(a);
        x.add(@as(c_int, n));
        x.add(g);
        x.add(@as(c_int, n));
        for ([_]u64{ cw, a_log, dt, norm_w }) |v| x.add(v);
        x.add(eps);
        x.add(lower);
        x.add(out);
        return go(try m.function("tf_glm_kda_chain_seg"), .{ .x = H, .y = segs }, 1024, s, &x);
    }
    var x: cuda.Args = .{};
    x.add(table);
    for ([_]i64{ li, conv_e, kv_e, b_e }) |v| x.add(v);
    x.add(h);
    x.add(p);
    x.add(@as(c_int, p_stride));
    x.add(@as(c_int, b_off));
    x.add(a);
    x.add(@as(c_int, n));
    for ([_]u64{ cw, a_log, dt }) |v| x.add(v);
    x.add(lower);
    x.add(q);
    try go(try m.function("tf_glm_kda_prep_seg"), .{ .x = window_rows, .y = H, .z = segs }, 512, s, &x);
    var z: cuda.Args = .{};
    z.add(table);
    for ([_]i64{ li, rec_e, kv_e, b_e }) |v| z.add(v);
    z.add(h);
    z.add(q);
    z.add(y);
    try go(try m.function("tf_glm_kda_step_seg"), .{ .x = H, .y = 4, .z = segs }, 256, s, &z);
    var o: cuda.Args = .{};
    o.add(h);
    o.add(y);
    o.add(g);
    o.add(@as(c_int, n));
    o.add(norm_w);
    o.add(eps);
    o.add(out);
    try go(try m.function("tf_glm_kda_out"), .{ .x = rows, .y = H }, dk, s, &o);
}

comptime {
    var total: usize = 0;
    for (seg_rows) |r| total += r;
    std.debug.assert(total == rows);
}
