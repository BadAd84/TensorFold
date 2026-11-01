//! glm-kda-pieces (GPU): TF_GLM_OVN_KDA_PIECES's one-launch chains against the per-piece chains, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");

const H = 22;
const dk = 128;
const dv = 128;
const n = H * dk;
const layers = 4;
const li = 1; // the layer under test: the table's conv_e / rec_e strides must reach it from layer 0
const conv_e = 3 * 3 * n; // a layer's conv window, bf16 elements: 3 rows of q | k | v
const rec_e = H * dk * dv; // a layer's recurrent state, fp32 elements
const p_stride = 3 * n + 2 * dk + H; // kda_block.rowWidth(H) with no pad (pad_to 0): the projection row's width

/// cuda_kda_seg.Seg, the layout glm_kda.cu's *_seg kernels read (also cuda_kda_pieces builds it).
const Seg = extern struct { row0: i64, rows: i64, cs: u64, state_in: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };

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

/// A run's outputs: the chunk's read-out rows (a sentinel past them) and each piece's recurrent state.
const Outs = struct {
    out: cuda.DeviceBuffer,
    state: [glm.kda_pieces.max_pieces]cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    v: cuda.DeviceBuffer,
    g: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver, gpa: std.mem.Allocator, total: usize, segs: usize) !Outs {
        var o: Outs = undefined;
        o.out = try cuda.DeviceBuffer.alloc(d, (total + 8) * n * 2);
        o.k = try cuda.DeviceBuffer.alloc(d, total * n * 4);
        o.v = try cuda.DeviceBuffer.alloc(d, total * n * 2);
        o.g = try cuda.DeviceBuffer.alloc(d, total * n * 4);
        o.b = try cuda.DeviceBuffer.alloc(d, total * H * 4);
        o.q = try cuda.DeviceBuffer.alloc(d, total * n * 4);
        o.y = try cuda.DeviceBuffer.alloc(d, total * n * 2);
        try o.out.fill8(7, null); // a sentinel: the chains write only [0, total) rows
        for (0..segs) |kk| o.state[kk] = try random(gpa, d, 2 * layers * rec_e, false, 0.2, 400 + kk); // same bytes both runs
        return o;
    }

    fn deinit(o: *Outs, segs: usize) void {
        for ([_]*cuda.DeviceBuffer{ &o.out, &o.k, &o.v, &o.g, &o.b, &o.q, &o.y }) |x| x.free();
        for (0..segs) |kk| o.state[kk].free();
    }

    fn scratch(o: *const Outs, p: u64, ka: u64, kg: u64) glm.kda_block.Scratch {
        return .{ .p = p, .ka = ka, .kg = kg, .saves = .{ .out = o.out.ptr, .state_out = 0, .k = o.k.ptr, .v = o.v.ptr, .g = o.g.ptr, .b = o.b.ptr }, .wide = .{ .q = o.q.ptr, .y = o.y.ptr }, .mm_part = 0 };
    }
};

fn layer(cw: u64, a_log: u64, dt: u64, norm_w: u64) glm.kda_block.Layer {
    return .{ .proj = .{ .bf16 = 0 }, .fb = 0, .gb = 0, .o = .{ .bf16 = 0 }, .conv_w = cw, .a_log = a_log, .dt_bias = dt, .norm_w = norm_w, .eps = 1e-6, .lower = -5.0, .heads = H, .dims = n };
}

fn runCase(gpu: check.Gpu, m: cuda.Module, s: cuda.Stream, seg_rows: []const usize) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const segs = seg_rows.len;
    var total: usize = 0;
    var max_rows: usize = 0;
    for (seg_rows) |r| {
        total += r;
        max_rows = @max(max_rows, r);
    }
    var ks = try glm.kda.Kernels.load(m);
    // shared read-only inputs (both runs read these; the chain never writes conv)
    var p = try random(gpa, d, total * p_stride, true, 2.0, 1);
    var ka = try random(gpa, d, total * n, true, 2.0, 2);
    var kg = try random(gpa, d, total * n, true, 3.0, 3);
    var cw = try random(gpa, d, 3 * n * 4, false, 0.3, 4);
    var a_log = try random(gpa, d, H, false, 1.5, 5);
    var dt = try random(gpa, d, n, false, 0.5, 6);
    var norm_w = try random(gpa, d, dk, true, 1.0, 7);
    var conv: [glm.kda_pieces.max_pieces]cuda.DeviceBuffer = undefined;
    for (0..segs) |kk| conv[kk] = try random(gpa, d, layers * conv_e, true, 1.0, 10 + kk);
    defer {
        for ([_]*cuda.DeviceBuffer{ &p, &ka, &kg, &cw, &a_log, &dt, &norm_w }) |x| x.free();
        for (0..segs) |kk| conv[kk].free();
    }
    const l = layer(cw.ptr, a_log.ptr, dt.ptr, norm_w.ptr);

    var want = try Outs.init(d, gpa, total, segs); // the per-piece arm (cuda_shared.mixer's prompt loop)
    defer want.deinit(segs);
    const sw = want.scratch(p.ptr, ka.ptr, kg.ptr);
    var row0: usize = 0;
    for (seg_rows, 0..) |r, kk| {
        var v = sw;
        v.p = p.ptr + row0 * p_stride * 2;
        v.ka = ka.ptr + row0 * n * 2;
        v.kg = kg.ptr + row0 * n * 2;
        v.saves.out = want.out.ptr + row0 * n * 2;
        v.saves.state_out = want.state[kk].ptr + @as(u64, (layers + li) * rec_e) * 4; // recAt(1, li)
        const conv_state = conv[kk].ptr + @as(u64, li * conv_e) * 2; // layer li's conv window
        const state_in = want.state[kk].ptr + @as(u64, li * rec_e) * 4; // recAt(0, li): fresh, cur 0
        try glm.kda_block.chainRows(&ks, s, l, v, r, conv_state, state_in, false, null); // the mixer's args: wide false, no chunked
        row0 += r;
    }

    var got = try Outs.init(d, gpa, total, segs); // the one-launch arm (PiecesKda.chains over the table)
    defer got.deinit(segs);
    for (0..segs) |kk| try d.check(d.api.cuMemcpyDtoDAsync_v2(got.state[kk].ptr, want.state[kk].ptr, 2 * layers * rec_e * 4, s.handle), "cuMemcpyDtoDAsync"); // same state_in bytes
    var pk = try glm.kda_pieces.PiecesKda.init(d, m);
    defer pk.deinit();
    var t: [glm.kda_pieces.max_pieces]Seg = undefined;
    row0 = 0;
    for (seg_rows, 0..) |r, kk| {
        const krow = row0 * n;
        t[kk] = .{ .row0 = @intCast(row0), .rows = @intCast(r), .cs = conv[kk].ptr, .state_in = got.state[kk].ptr, .state_out = got.state[kk].ptr + @as(u64, layers * rec_e) * 4, .k = got.k.ptr + @as(u64, krow) * 4, .v = got.v.ptr + @as(u64, krow) * 2, .g = got.g.ptr + @as(u64, krow) * 4, .b = got.b.ptr + @as(u64, row0) * H * 4 };
        row0 += r;
    }
    try pk.table.uploadAsync(0, std.mem.sliceAsBytes(t[0..segs]), s.handle);
    pk.segs = segs;
    pk.wide = seg_rows[0] >= glm.kda.wide_rows;
    pk.max_rows = max_rows;
    pk.conv_e = conv_e;
    pk.rec_e = rec_e;
    try pk.chains(&ks, s, l, got.scratch(p.ptr, ka.ptr, kg.ptr), li, total);
    try s.synchronize();

    try same(gpu, "kda-pieces out", want.out, got.out);
    for (0..segs) |kk| try same(gpu, "kda-pieces state", want.state[kk], got.state[kk]);
    check.pass("glm-kda-pieces: {d} pieces of {any} rows ({s}) equal the per-piece chains byte for byte", .{ segs, seg_rows, if (pk.wide) "wide" else "serial" });
}

pub fn run(gpu: check.Gpu) !void {
    var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
    defer m.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    try runCase(gpu, m, s, &.{ 39, 39, 39, 39 }); // production's four 39-row fork pieces: the serial chain_seg
    try runCase(gpu, m, s, &.{ 1, 5, 31, 33 }); // uneven small pieces, including a one-row piece
    try runCase(gpu, m, s, &.{ 64, 300, 128 }); // all at or past the wide floor: prep_seg / step_seg / out
    try runCase(gpu, m, s, &.{ 2390, 1706 }); // two wide pieces filling the prompt buffer
}
