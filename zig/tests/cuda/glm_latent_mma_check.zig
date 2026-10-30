//! glm-latent-mma-check (GPU): TF_GLM_OVN_LATENT_MMA's tensor-core absorb and expand against float64, deterministic,
//! each row's bytes whatever rows share its launch (resumed == fresh), and the E4M3 forms on exact kv_b values.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const latent = glm.latent;
const glue = glm.glue;

const lw = 512;
const d = 256; // qk and v head widths
const heads = 22; // TP=3's widest rank
const rows = 70; // two 64-row blocks, the second 6 rows
const nan: u16 = 0x7fc0;

fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// Device kv_b (bf16, or E4M3 codes with their scales) and its exact values, [heads][256][512] either way.
const Weights = struct { w: u64, s: u64 = 0, exact: []const f64 };

/// out [rows][heads][n] in float64 and its bound: a bf16 step of |y| (2^-7) and 2^-12 of sum |x w| (as glm-ref's).
fn reference(r: *Rig, x: []const u16, w: []const f64, absorb: bool) !struct { want: []f64, tol: []f64 } {
    const k: usize = if (absorb) d else lw;
    const n: usize = if (absorb) lw else d;
    const want = try r.a.alloc(f64, rows * heads * n);
    const tol = try r.a.alloc(f64, rows * heads * n);
    const abs = try r.a.alloc(f64, n);
    for (0..rows) |row| for (0..heads) |h| {
        const xs = x[(row * heads + h) * k ..][0..k];
        const sum = want[(row * heads + h) * n ..][0..n];
        @memset(sum, 0);
        @memset(abs, 0);
        if (absorb) {
            for (xs, 0..) |xv, kk| for (w[(h * d + kk) * lw ..][0..lw], sum, abs) |wv, *s, *a| {
                const t = bf(xv) * wv;
                s.* += t;
                a.* += @abs(t);
            };
        } else {
            for (sum, abs, 0..) |*s, *a, nn| for (xs, w[(h * d + nn) * lw ..][0..lw]) |xv, wv| {
                const t = bf(xv) * wv;
                s.* += t;
                a.* += @abs(t);
            };
        }
        for (sum, abs, tol[(row * heads + h) * n ..][0..n]) |s, a, *t| t.* = @abs(s) * std.math.pow(f64, 2, -7) + a * std.math.pow(f64, 2, -12) + 1e-30;
    };
    return .{ .want = want, .tol = tol };
}

/// One launch of `count` rows from `x` into a fresh output (0xff where nothing is written).
fn launch(r: *Rig, set: glue.Set, absorb: bool, x: u64, w: Weights, count: usize) !u64 {
    const out = try r.zeros(count * heads * @as(usize, if (absorb) lw else d) * 2);
    if (absorb) try latent.absorbQ(set, r.s, x, w.w, w.s, out, count, heads, d, lw) else try latent.expandV(set, r.s, x, w.w, w.s, out, count, heads, d, lw);
    return out;
}

/// Float64 within tolerance; a second run the same bytes; rows 37 on and row 69 alone as in the full launch; a NaN
/// in row 5 leaves every other row's bytes alone. Returns the inputs and the full launch's output.
fn kernel(r: *Rig, set: glue.Set, absorb: bool, w: Weights, what: []const u8) !struct { x: u64, out: u64 } {
    const k: usize = if (absorb) d else lw;
    const n: usize = if (absorb) lw else d;
    const bytes = rows * heads * n * 2;
    const row_bytes = heads * n * 2;
    const xh = try r.bfs(rows * heads * k, 1.0);
    const x = try r.dev(std.mem.sliceAsBytes(xh));
    const full = try launch(r, set, absorb, x, w, rows);
    const ref = try reference(r, xh, w.exact, absorb);
    var name: [128]u8 = undefined;
    try r.close(try std.fmt.bufPrint(&name, "{s} against float64", .{what}), full, false, ref.want, ref.tol);
    const want = try r.back(full, bytes);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: a second run", .{what}), try r.back(try launch(r, set, absorb, x, w, rows), bytes), want);
    for ([_]usize{ 37, 69 }) |lo| {
        const part = try launch(r, set, absorb, x + @as(u64, lo * heads * k * 2), w, rows - lo);
        try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: rows {d} on alone", .{ what, lo }), try r.back(part, (rows - lo) * row_bytes), want[lo * row_bytes ..]);
    }
    const poisoned = try r.a.dupe(u16, xh);
    poisoned[(5 * heads) * k + 3] = nan;
    const got = try r.back(try launch(r, set, absorb, try r.dev(std.mem.sliceAsBytes(poisoned)), w, rows), bytes);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: rows before a NaN row", .{what}), got[0 .. 5 * row_bytes], want[0 .. 5 * row_bytes]);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: rows after a NaN row", .{what}), got[6 * row_bytes ..], want[6 * row_bytes ..]);
    for (0..n) |nn| {
        const v = std.mem.bytesToValue(u16, got[5 * row_bytes + 2 * nn ..][0..2]);
        try check.expect(std.math.isNan(bf(v)), "{s}: the NaN row's head 0 output {d} is NaN", .{ what, nn });
    }
    return .{ .x = x, .out = full };
}

/// bf16 kv_b in [-0.05, 0.05).
fn bf16Weights(r: *Rig) !Weights {
    const wh = try r.bfs(heads * d * lw, 0.05);
    const exact = try r.a.alloc(f64, wh.len);
    for (exact, wh) |*e, v| e.* = bf(v);
    return .{ .w = try r.dev(std.mem.sliceAsBytes(wh)), .exact = exact };
}

/// E4M3 kv_b at power-of-two scales a k row and 128 columns (code x scale exact in bf16), and those values as bf16.
fn f8Weights(r: *Rig) !struct { f8: Weights, wide: Weights } {
    const n = heads * d * lw;
    const codes = try r.a.alloc(u8, n);
    const sc = try r.a.alloc(f32, n / 128);
    const wide = try r.a.alloc(u16, n);
    const exact = try r.a.alloc(f64, n);
    for (sc) |*s| s.* = std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(r.rng.uintLessThan(u32, 13))));
    for (codes, wide, exact, 0..) |*c, *w, *e, i| {
        var b = r.rng.int(u8);
        if (b & 0x7f == 0x7f) b ^= 1; // no NaN codes
        c.* = b;
        const v = core.exlr8.codec.e4m3Value(b) * sc[i / 128];
        w.* = @intCast(@as(u32, @bitCast(v)) >> 16);
        e.* = v;
    }
    const f8: Weights = .{ .w = try r.dev(codes), .s = try r.dev(std.mem.sliceAsBytes(sc)), .exact = exact };
    return .{ .f8 = f8, .wide = .{ .w = try r.dev(std.mem.sliceAsBytes(wide)), .exact = exact } };
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x3a7);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    const set: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .mma = true } };
    const b16 = try bf16Weights(&r);
    _ = try kernel(&r, set, true, b16, "absorb_mma");
    _ = try kernel(&r, set, false, b16, "expand_mma");
    const f8 = try f8Weights(&r);
    _ = try kernel(&r, set, false, f8.f8, "expand_mma8");
    // absorb_mma8's weights are bf16(code x scale), here exact: _absorb_mma's chain on the same values, byte for byte
    const a = try kernel(&r, set, true, f8.wide, "absorb_mma on E4M3 values");
    const bytes = rows * heads * lw * 2;
    try check.sameBytes("absorb_mma8 against absorb_mma on the exact values", try r.back(try launch(&r, set, true, a.x, f8.f8, rows), bytes), try r.back(a.out, bytes));
    check.pass("glm-latent-mma-check: tensor-core absorb/expand within float64 tolerance, deterministic, rows independent of their launch; worst {d:.3}", .{r.worst});
}
