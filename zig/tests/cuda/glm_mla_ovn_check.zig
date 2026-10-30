//! glm-mla-ovn-check (GPU): TF_GLM_OVN_MLA_FUSEMERGE, _ABSORB_VEC and _KVB8 kernels against today's, byte for byte.
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
const heads = 22; // TP=3's widest rank: padded heads in both head blocks
const scale: f32 = 0.0625;
const keys = 6000;
const width = 2051; // sparse.width: five 512-key chunks, the last with 3 keys
const nan: u16 = 0x7fc0;

/// today's launches, the bounded tile loops, and every 32-head launch fused with its merge
const Sets = struct { off: glue.Set, skip: glue.Set, fused: glue.Set };

fn position(r: *Rig, v: usize) !u64 {
    const p: i32 = @intCast(v);
    return r.dev(std.mem.asBytes(&p));
}

fn partials(r: *Rig, nch: usize, rows: usize) !latent.Partials {
    return .{ .po = try r.zeros(nch * rows * heads * lw * 4), .pm = try r.zeros(nch * rows * heads * 4), .pl = try r.zeros(nch * rows * heads * 4) };
}

/// The latent cache as bf16 (+inf in the last key, which only sparse lists reach) and as E4M3 rows with scales.
const Cache = struct { bf: u64, fp: u64, scales: u64 };

fn caches(r: *Rig, set: glue.Set) !Cache {
    const x = try r.bfs(keys * lw, 1.0);
    const fp = try r.zeros(try latent.cacheBytes(keys, lw, true));
    const scales = fp + keys * lw;
    try latent.latentWriteFormat(set, r.s, try r.dev(std.mem.sliceAsBytes(x)), lw, fp, scales, .{}, try position(r, 0), keys, lw);
    x[(keys - 1) * lw + 3] = 0x7f80;
    return .{ .bf = try r.dev(std.mem.sliceAsBytes(x)), .fp = fp, .scales = scales };
}

/// The skip and fused outputs equal today's bytes (a row left alone stays 0xff in all three).
fn same(r: *Rig, what: []const u8, outs: [3]u64, bytes: usize) !void {
    var name: [160]u8 = undefined;
    const want = try r.back(outs[0], bytes);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: bounded tile loops", .{what}), try r.back(outs[1], bytes), want);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: fused merge", .{what}), try r.back(outs[2], bytes), want);
}

/// Queries with a NaN in row 3 (when there is one): its outputs go NaN alike on every path.
fn queries(r: *Rig, rows: usize) !u64 {
    const q = try r.bfs(rows * heads * lw, 1.0);
    if (rows > 3) q[3 * heads * lw + 5] = nan;
    return r.dev(std.mem.sliceAsBytes(q));
}

fn dense(r: *Rig, sets: Sets, c: Cache, fp8: bool, pos: usize, rows: usize, hb: usize) !void {
    const dq = try queries(r, rows);
    const at = try position(r, pos);
    const nch = latent.chunksFor(pos + rows);
    var outs: [3]u64 = undefined;
    for (&outs, [_]glue.Set{ sets.off, sets.skip, sets.fused }) |*o, set| {
        o.* = try r.zeros(rows * heads * lw * 2);
        try latent.attentionFormat(set, r.s, dq, if (fp8) c.fp else c.bf, if (fp8) c.scales else 0, .{}, at, try partials(r, nch, rows), o.*, try r.zeros(4), rows, heads, lw, nch, scale, hb);
    }
    var name: [96]u8 = undefined;
    try same(r, try std.fmt.bufPrint(&name, "dense {s} {d} rows at {d}, {d}-head blocks", .{ if (fp8) "e4m3" else "bf16", rows, pos, hb }), outs, rows * heads * lw * 2);
}

fn sparse(r: *Rig, sets: Sets, c: Cache, fp8: bool, rows: usize) !void {
    const counts = [_]i32{ 0, 1, 31, 32, 33, 511, 512, 513, 1024, 2048, 2051, 3 }; // chunk and tile edges, count 0
    const tok = try r.a.alloc(i32, rows * width);
    @memset(tok, -1);
    const cnt = try r.a.alloc(i32, rows);
    for (cnt, 0..) |*n, row| {
        n.* = if (rows > 8 and row < counts.len) counts[row] else @intCast(1 + r.rng.uintLessThan(usize, width));
        for (tok[row * width ..][0..@intCast(n.*)]) |*t| t.* = @intCast(r.rng.uintLessThan(usize, keys));
    }
    const dq = try queries(r, rows);
    const dtok = try r.dev(std.mem.sliceAsBytes(tok));
    const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
    const nch = latent.chunksFor(width);
    var outs: [3]u64 = undefined;
    for (&outs, [_]glue.Set{ sets.off, sets.skip, sets.fused }) |*o, set| {
        o.* = try r.zeros(rows * heads * lw * 2);
        try latent.sparseAttentionFormat(set, r.s, dq, if (fp8) c.fp else c.bf, if (fp8) c.scales else 0, .{}, dtok, dcnt, width, try partials(r, nch, rows), o.*, rows, heads, lw, scale);
    }
    var name: [96]u8 = undefined;
    try same(r, try std.fmt.bufPrint(&name, "sparse {s} {d} rows", .{ if (fp8) "e4m3" else "bf16", rows }), outs, rows * heads * lw * 2);
}

/// tf_glm_absorb_q / tf_glm_expand_v against their 16-byte-load forms, NaN and -inf among the inputs.
fn vectors(r: *Rig, off: glue.Set, vec: glue.Set, rows: usize) !void {
    const q = try r.bfs(rows * heads * d, 1.0);
    q[0] = nan;
    const o = try r.bfs(rows * heads * lw, 1.0);
    o[lw + 1] = 0xff80;
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const dol = try r.dev(std.mem.sliceAsBytes(o));
    const wk = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
    const wv = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
    const a = [2]u64{ try r.zeros(rows * heads * lw * 2), try r.zeros(rows * heads * lw * 2) };
    const e = [2]u64{ try r.zeros(rows * heads * d * 2), try r.zeros(rows * heads * d * 2) };
    for (a, e, [_]glue.Set{ off, vec }) |qa, vn, set| {
        try latent.absorbQ(set, r.s, dq, wk, 0, qa, rows, heads, d, lw);
        try latent.expandV(set, r.s, dol, wv, 0, vn, rows, heads, d, lw);
    }
    var name: [96]u8 = undefined;
    try check.sameBytes(try std.fmt.bufPrint(&name, "absorb_q_vec {d} rows", .{rows}), try r.back(a[1], rows * heads * lw * 2), try r.back(a[0], rows * heads * lw * 2));
    try check.sameBytes(try std.fmt.bufPrint(&name, "expand_v_vec {d} rows", .{rows}), try r.back(e[1], rows * heads * d * 2), try r.back(e[0], rows * heads * d * 2));
}

/// E4M3 kv_b at power-of-two scales (code x scale exact in bf16) against the bf16 kernels on those exact values.
fn kvb8(r: *Rig, off: glue.Set, rows: usize) !void {
    const n = heads * d * lw;
    const codes = try r.a.alloc(u8, n);
    const sc = try r.a.alloc(f32, n / 128);
    const wide = try r.a.alloc(u16, n);
    for (sc) |*s| s.* = std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(r.rng.uintLessThan(u32, 13))));
    for (codes, wide, 0..) |*c, *w, i| {
        var b = r.rng.int(u8);
        if (b & 0x7f == 0x7f) b ^= 1; // no NaN codes
        c.* = b;
        w.* = @intCast(@as(u32, @bitCast(core.exlr8.codec.e4m3Value(b) * sc[i / 128])) >> 16);
    }
    const dq = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * d, 1.0)));
    const dol = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * lw, 1.0)));
    const dc = try r.dev(codes);
    const ds = try r.dev(std.mem.sliceAsBytes(sc));
    const dw = try r.dev(std.mem.sliceAsBytes(wide));
    const a = [2]u64{ try r.zeros(rows * heads * lw * 2), try r.zeros(rows * heads * lw * 2) };
    const e = [2]u64{ try r.zeros(rows * heads * d * 2), try r.zeros(rows * heads * d * 2) };
    try latent.absorbQ(off, r.s, dq, dc, ds, a[0], rows, heads, d, lw);
    try latent.absorbQ(off, r.s, dq, dw, 0, a[1], rows, heads, d, lw);
    try latent.expandV(off, r.s, dol, dc, ds, e[0], rows, heads, d, lw);
    try latent.expandV(off, r.s, dol, dw, 0, e[1], rows, heads, d, lw);
    var name: [96]u8 = undefined;
    try check.sameBytes(try std.fmt.bufPrint(&name, "absorb_q8 {d} rows", .{rows}), try r.back(a[0], rows * heads * lw * 2), try r.back(a[1], rows * heads * lw * 2));
    try check.sameBytes(try std.fmt.bufPrint(&name, "expand_v8 {d} rows", .{rows}), try r.back(e[0], rows * heads * d * 2), try r.back(e[1], rows * heads * d * 2));
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x0b5);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    const sets: Sets = .{ .off = .{ .glue = &ks }, .skip = .{ .glue = &ks, .mla = .{ .skip = true } }, .fused = .{ .glue = &ks, .mla = .{ .fuse_rows = 1, .skip = true } } };
    const c = try caches(&r, sets.off);
    for ([_]bool{ false, true }) |fp8| {
        for ([_][3]usize{ .{ 1430, 70, 32 }, .{ 1430, 70, 16 }, .{ 1000, 1, 16 }, .{ 100, 2, 16 }, .{ 2040, 5, 32 }, .{ 2560, 64, 32 } }) |x| try dense(&r, sets, c, fp8, x[0], x[1], x[2]);
        for ([_]usize{ 70, 2, 5 }) |rows| try sparse(&r, sets, c, fp8, rows);
    }
    const vec: glue.Set = .{ .glue = &ks, .mla = .{ .vec = true } };
    for ([_]usize{ 2, 70 }) |rows| {
        try vectors(&r, sets.off, vec, rows);
        try kvb8(&r, sets.off, rows);
    }
    check.pass("glm-mla-ovn-check: bounded and fused chunk passes, 16-byte absorb/expand and exact E4M3 kv_b equal today's bytes", .{});
}
