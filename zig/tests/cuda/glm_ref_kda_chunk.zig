//! TF_GLM_OVN_KDA_CHUNKED's chunked (WY) KDA chain: within float64 and serial tolerance, deterministic, resumes on 32.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const kref = @import("glm_ref_kda.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;

const kda = glm.kda;
const kc = glm.kda_chunk;
const heads = kref.heads;
const n = kref.n;
const dk = kda.dk;
const rows = 70; // two whole sub-chunks and a short last one
const cut = 32; // a resume on the sub-chunk grid
const state_bytes = heads * kda.dv * dk * 4;
const out_bytes = rows * n * 2;

fn chainOf(d: [9]u64, p_stride: usize) kda.Chain {
    return .{ .p = d[0], .p_stride = p_stride, .b_off = 3 * n, .a = d[1], .a_stride = n, .g = d[2], .g_stride = n, .conv_state = d[3], .conv_w = d[4], .state_in = d[5], .a_log = d[6], .dt_bias = d[7], .norm_w = d[8], .eps = kref.eps, .lower = kref.lower, .heads = heads };
}

/// glm_ref_kda's tolerance: a bf16 ulp of the value and 2^-8 of its row's largest, times `scale`.
fn tolerance(r: *Rig, want: []const f64, scale: f64) ![]f64 {
    const tol = try r.a.alloc(f64, want.len);
    for (0..rows) |row| {
        var most: f64 = 0;
        for (want[row * n ..][0..n]) |e| most = @max(most, @abs(e));
        for (tol[row * n ..][0..n], want[row * n ..][0..n]) |*t, e| t.* = scale * (@abs(e) * std.math.pow(f64, 2, -8) + most * std.math.pow(f64, 2, -8)) + 1e-30;
    }
    return tol;
}

/// The chunked state against the serial one: each head's worst difference over 2^-10 of its largest serial entry.
fn statesClose(r: *Rig, name: []const u8, got: u64, want: u64) !void {
    const g = try r.back(got, state_bytes);
    const w = try r.back(want, state_bytes);
    const col = try r.a.alloc(f64, heads);
    const per = kda.dv * dk;
    for (col, 0..) |*c, h| {
        var most: f64 = 0;
        for (0..per) |i| most = @max(most, @abs(@as(f64, std.mem.bytesToValue(f32, w[4 * (h * per + i) ..][0..4]))));
        var worst: f64 = 0;
        for (0..per) |i| {
            const a: f64 = std.mem.bytesToValue(f32, g[4 * (h * per + i) ..][0..4]);
            const b: f64 = std.mem.bytesToValue(f32, w[4 * (h * per + i) ..][0..4]);
            const q = @abs(a - b) / (most * std.math.pow(f64, 2, -10) + 1e-30);
            worst = if (std.math.isNan(q)) std.math.inf(f64) else @max(worst, q);
        }
        c.* = worst;
    }
    try ref.worstCol(r, name, col);
}

/// `glm-ref`'s chunked KDA (TF_GLM_OVN_KDA_CHUNKED), 70 rows of 32 heads on its own seed: later checks draw alike.
pub fn chains(r: *Rig) !void {
    const saved = r.rng;
    defer r.rng = saved;
    var prng = std.Random.DefaultPrng.init(3232);
    r.rng = prng.random();
    var mod = try cuda.Module.load(r.gpu.d, cuda.kernels.glm_kda);
    defer mod.unload();
    const ks = try kda.Kernels.load(mod);
    var kernel = try kc.Kernel.load(r.gpu.d);
    defer kernel.unload();
    const ps = kref.p_stride;
    const in: kref.Inputs = .{
        .p = try r.bfs(rows * ps, 1.0), // q | k | v, then beta's logits at 3 n
        .a = try r.bfs(rows * n, 2.0),
        .g = try r.bfs(rows * n, 2.0),
        .cs = try r.bfs(3 * 3 * n, 1.0),
        .cw = try kref.floats(r, 3 * n * 4, -0.5, 0.5),
        .state = try kref.floats(r, heads * kda.dv * dk, -0.05, 0.05),
        .a_log = try kref.floats(r, heads, -1, 1),
        .dt_bias = try kref.floats(r, n, -0.5, 0.5),
        .norm = try r.bfs(dk, 1.0),
    };
    const d = [9]u64{
        try r.dev(std.mem.sliceAsBytes(in.p)),     try r.dev(std.mem.sliceAsBytes(in.a)),       try r.dev(std.mem.sliceAsBytes(in.g)),
        try r.dev(std.mem.sliceAsBytes(in.cs)),    try r.dev(std.mem.sliceAsBytes(in.cw)),      try r.dev(std.mem.sliceAsBytes(in.state)),
        try r.dev(std.mem.sliceAsBytes(in.a_log)), try r.dev(std.mem.sliceAsBytes(in.dt_bias)), try r.dev(std.mem.sliceAsBytes(in.norm)),
    };
    const ch = chainOf(d, ps);
    const at0: kc.Prompt = .{ .kernel = &kernel, .pos = 0 };

    // the serial three-kernel chain, a prompt chunk's path with the knob off
    const s_out = try r.zeros(out_bytes);
    const s_state = try r.zeros(state_bytes);
    const saves: kda.Saves = .{ .out = s_out, .state_out = s_state, .k = try r.zeros(rows * n * 4), .v = try r.zeros(rows * n * 2), .g = try r.zeros(rows * n * 4), .b = try r.zeros(rows * heads * 4) };
    try kda.chain(&ks, r.s, ch, rows, saves, .{ .q = try r.zeros(rows * n * 4), .y = try r.zeros(rows * n * 2) });

    // the chunked chain twice from position 0: the same bytes
    var outs: [2]u64 = undefined;
    var states: [2]u64 = undefined;
    for (&outs, &states) |*o, *s| {
        o.* = try r.zeros(out_bytes);
        s.* = try r.zeros(state_bytes);
        try kc.chain(at0, r.s, ch, rows, o.*, s.*);
    }
    try ref.sameRows(r, "KDA chunked chain: a second run's read-outs equal the first's", outs[0], outs[1], out_bytes);
    try ref.sameRows(r, "KDA chunked chain: a second run's state equals the first's", states[0], states[1], state_bytes);

    // resumed at row 32 (on the grid) from the kept state and conv window: the whole chunk's bytes
    const window = try r.a.alloc(u16, 3 * 3 * n);
    for (0..3) |j| @memcpy(window[j * 3 * n ..][0 .. 3 * n], in.p[(cut - 3 + j) * ps ..][0 .. 3 * n]);
    const head_state = try r.zeros(state_bytes);
    const split_out = try r.zeros(out_bytes);
    const split_state = try r.zeros(state_bytes);
    try kc.chain(at0, r.s, ch, cut, split_out, head_state);
    var tail = ch;
    tail.p += cut * ps * 2;
    tail.a += cut * n * 2;
    tail.g += cut * n * 2;
    tail.conv_state = try r.dev(std.mem.sliceAsBytes(window));
    tail.state_in = head_state;
    try kc.chain(.{ .kernel = &kernel, .pos = cut }, r.s, tail, rows - cut, split_out + cut * n * 2, split_state);
    try ref.sameRows(r, "KDA chunked chain resumed at row 32: read-outs equal one chunk's", outs[0], split_out, out_bytes);
    try ref.sameRows(r, "KDA chunked chain resumed at row 32: the state equals one chunk's", states[0], split_state, state_bytes);

    // against float64 and the serial chain
    const want = try kref.reference(r, in, rows);
    try ref.closeCols(r, "KDA chunked chain, 70 rows x 32 heads, against float64 (column = head x 128 + t)", outs[0], false, want, try tolerance(r, want, 1), n);
    const serial_raw = try r.back(s_out, out_bytes);
    const serial = try r.a.alloc(f64, rows * n);
    for (serial, 0..) |*e, i| e.* = bf(std.mem.bytesToValue(u16, serial_raw[2 * i ..][0..2]));
    try ref.closeCols(r, "KDA chunked chain against the serial chain's read-outs (both within float64's tolerance)", outs[0], false, serial, try tolerance(r, want, 2), n);
    try statesClose(r, "KDA chunked chain's state against the serial chain's (column = head)", states[0], s_state);

    // off the grid (row 0 at position 7) on rows of an uneven stride (the unpadded projection, 8-byte loads)
    const odd = ps + 1;
    const p_odd = try r.a.alloc(u16, rows * odd);
    for (0..rows) |row| @memcpy(p_odd[row * odd ..][0..ps], in.p[row * ps ..][0..ps]);
    var shifted = chainOf(d, odd);
    shifted.p = try r.dev(std.mem.sliceAsBytes(p_odd));
    const o7 = try r.zeros(out_bytes);
    try kc.chain(.{ .kernel = &kernel, .pos = 7 }, r.s, shifted, rows, o7, try r.zeros(state_bytes));
    try ref.closeCols(r, "KDA chunked chain from position 7 on an uneven row stride, against float64", o7, false, want, try tolerance(r, want, 1), n);
    check.pass("KDA chunked chain {d} heads: deterministic, resumed at 32 equals whole, within float64 and serial tolerance", .{heads});
}
