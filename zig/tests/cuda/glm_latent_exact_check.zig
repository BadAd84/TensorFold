//! glm-latent-exact-check (GPU): exact CUDA-core absorb/expand equal base byte for byte; LATENT_MMA and facets differ.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const latent = glm.latent;
const glue = glm.glue;

const lw = 512;
const d = 256; // qk and v head widths
const rows = 4096; // one prompt chunk

/// rowBlocks' launch (cuda_latent.zig), for the ORDER and TREE facet kernels loaded by name.
fn raw(f: cuda.Function, s: cuda.Stream, x: u64, w: u64, out: u64, n_rows: usize, heads: usize) !void {
    const rb: usize = if (n_rows >= 64) 64 else n_rows; // latent.prompt_rows / prompt_rb
    var a: cuda.Args = .{};
    inline for (.{ x, w, out }) |v| a.add(@as(u64, v));
    inline for (.{ @as(i64, @intCast(n_rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    const z: u32 = @intCast((n_rows + rb - 1) / rb);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = 16, .z = z }, .block = .{ .x = 128 } }, s, &a);
}

/// got must not equal want (a red control: the kernel changes the bytes, so the check is not vacuous).
fn differs(what: []const u8, got: []const u8, want: []const u8) !void {
    try check.expect(!std.mem.eql(u8, got, want), "{s}: equals the base bytes, so the red control is vacuous", .{what});
}

const Set3 = struct { base: glue.Set, exact: glue.Set, mma: glue.Set };

/// absorb or expand at `heads`: exact equals base byte for byte, a resumed tail matches, mma and facets each differ.
fn direction(r: *Rig, s3: Set3, heads: usize, absorb: bool, ord: cuda.Function, tre: cuda.Function, what: []const u8) !void {
    const k: usize = if (absorb) d else lw;
    const n: usize = if (absorb) lw else d;
    const row_bytes = heads * n * 2;
    const bytes = rows * row_bytes;
    const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * k, 1.0)));
    const w = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05)));
    var name: [160]u8 = undefined;
    const base = try run1(r, s3.base, absorb, x, w, rows, heads);
    const exact = try run1(r, s3.exact, absorb, x, w, rows, heads);
    const want = try r.back(base, bytes);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s} {d} heads: exact equals base", .{ what, heads }), try r.back(exact, bytes), want);
    // a resumed tail: rows `lo` on, launched alone, equal the full pass's rows lo on
    for ([_]usize{ 37, 4000 }) |lo| {
        const part = try run1(r, s3.exact, absorb, x + @as(u64, lo * heads * k * 2), w, rows - lo, heads);
        try check.sameBytes(try std.fmt.bufPrint(&name, "{s} {d} heads: exact tail {d} on", .{ what, heads, lo }), try r.back(part, (rows - lo) * row_bytes), want[lo * row_bytes ..]);
    }
    // red controls: the tensor-core form and the two facets each change the bytes
    const mma = try run1(r, s3.mma, absorb, x, w, rows, heads);
    try differs(try std.fmt.bufPrint(&name, "{s} {d} heads: LATENT_MMA", .{ what, heads }), try r.back(mma, bytes), want);
    const order = try r.zeros(bytes);
    const tree = try r.zeros(bytes);
    try raw(ord, r.s, x, w, order, rows, heads);
    try raw(tre, r.s, x, w, tree, rows, heads);
    try differs(try std.fmt.bufPrint(&name, "{s} {d} heads: ORDER off", .{ what, heads }), try r.back(order, bytes), want);
    try differs(try std.fmt.bufPrint(&name, "{s} {d} heads: TREE off", .{ what, heads }), try r.back(tree, bytes), want);
}

fn run1(r: *Rig, set: glue.Set, absorb: bool, x: u64, w: u64, n_rows: usize, heads: usize) !u64 {
    const out = try r.zeros(n_rows * heads * @as(usize, if (absorb) lw else d) * 2);
    if (absorb) try latent.absorbQ(set, r.s, x, w, 0, out, n_rows, heads, d, lw) else try latent.expandV(set, r.s, x, w, 0, out, n_rows, heads, d, lw);
    return out;
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x1a7e);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    const s3: Set3 = .{
        .base = .{ .glue = &ks, .prompt = true },
        .exact = .{ .glue = &ks, .prompt = true, .mla = .{ .mma_exact = true } },
        .mma = .{ .glue = &ks, .prompt = true, .mla = .{ .mma = true } },
    };
    const ao = try mod.function("tf_glm_absorb_exact_order");
    const at = try mod.function("tf_glm_absorb_exact_tree");
    const eo = try mod.function("tf_glm_expand_exact_order");
    const et = try mod.function("tf_glm_expand_exact_tree");
    for ([_]usize{ 22, 21 }) |heads| {
        try direction(&r, s3, heads, true, ao, at, "absorb");
        try direction(&r, s3, heads, false, eo, et, "expand");
    }
    check.pass("glm-latent-exact-check: CUDA-core absorb/expand equal base byte for byte, rows independent of their launch; LATENT_MMA and the ORDER/TREE facets each differ", .{});
}
