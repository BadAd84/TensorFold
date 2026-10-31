//! glm-onepass-exact-check (GPU): tf_glm_onepass_exact == chunk programs + merge byte for byte; one-pass red control.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const latent = glm.latent;
const glue = glm.glue;
const op = glm.onepass;

const lw = 512;
const scale: f32 = 0.0625;
const keys = 6000;
const width = 2051; // sparse.width: five 512-key chunks, the last with 3 keys
const heads_max = 24;

const Cache = struct { fp: u64, scales: u64 };

fn position(r: *Rig, v: usize) !u64 {
    const p: i32 = @intCast(v);
    return r.dev(std.mem.asBytes(&p));
}

/// The latent cache the FP8 writer packs: E4M3 codes then one fp32 scale a row, as glm_attn_ovn::chunks reads a slab.
fn caches(r: *Rig, set: glue.Set) !Cache {
    const x = try r.bfs(keys * lw, 1.0);
    const fp = try r.zeros(try latent.cacheBytes(keys, lw, true));
    const scales = fp + keys * lw;
    try latent.latentWriteFormat(set, r.s, try r.dev(std.mem.sliceAsBytes(x)), lw, fp, scales, .{}, try position(r, 0), keys, lw);
    return .{ .fp = fp, .scales = scales };
}

/// One case: the exact kernel equals the default's bytes, and the existing one-pass does not (the red control).
fn one(r: *Rig, ks: *const glue.Kernels, onepass_f: cuda.Function, pipe_exact: cuda.Function, c: Cache, rows: usize, heads: usize) !usize {
    const counts = [_]i32{ 0, 1, 31, 32, 33, 511, 512, 513, 1024, 2048, 2051, 3 }; // chunk and tile edges, count 0
    const tok = try r.a.alloc(i32, rows * width);
    @memset(tok, -1);
    const cnt = try r.a.alloc(i32, rows);
    for (cnt, 0..) |*n, row| {
        n.* = if (rows > counts.len and row < counts.len) counts[row] else @intCast(1 + r.rng.uintLessThan(usize, width));
        for (tok[row * width ..][0..@intCast(n.*)]) |*t| t.* = @intCast(r.rng.uintLessThan(usize, keys));
    }
    const dq = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * lw, 0.5)));
    const dtok = try r.dev(std.mem.sliceAsBytes(tok));
    const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
    const bytes = rows * heads * lw * 2;
    const nch = latent.chunksFor(width);
    const off: glue.Set = .{ .glue = ks };
    const out_def = try r.zeros(bytes);
    const part: latent.Partials = .{ .po = try r.zeros(nch * rows * heads * lw * 4), .pm = try r.zeros(nch * rows * heads * 4), .pl = try r.zeros(nch * rows * heads * 4) };
    try latent.sparseAttentionFormat(off, r.s, dq, c.fp, c.scales, .{}, dtok, dcnt, width, part, out_def, rows, heads, lw, scale);
    const out_exact = try r.zeros(bytes);
    try op.runExact(ks.onepass_exact, r.s, dq, c.fp, c.scales, .{}, dtok, dcnt, width, out_exact, rows, heads, scale);
    const out_op = try r.zeros(bytes);
    try op.run(onepass_f, r.s, dq, c.fp, c.scales, .{}, dtok, dcnt, width, out_op, rows, heads, scale);
    var name: [96]u8 = undefined;
    const want = try r.back(out_def, bytes);
    var fails: usize = 0;
    check.sameBytes(try std.fmt.bufPrint(&name, "exact one-pass == chunk programs + merge, {d} rows {d} heads", .{ rows, heads }), try r.back(out_exact, bytes), want) catch {
        fails += 1;
    };
    const out_pipe = try r.zeros(bytes);
    try op.runPipe(pipe_exact, r.s, dq, c.fp, c.scales, .{}, dtok, dcnt, width, out_pipe, rows, heads, scale);
    check.sameBytes(try std.fmt.bufPrint(&name, "pipe exact == chunk programs + merge, {d} rows {d} heads", .{ rows, heads }), try r.back(out_pipe, bytes), want) catch {
        fails += 1;
    };
    const ctrl = try r.back(out_op, bytes);
    check.expect(!std.mem.eql(u8, ctrl, want), "glm-onepass-exact-check: {d} rows {d} heads: the existing one-pass matched the default (the red control must differ)", .{ rows, heads }) catch {
        fails += 1;
    };
    return fails;
}

pub fn run(gpu: check.Gpu) !void {
    const major: u32 = @intCast(try gpu.ctx.attribute(.compute_capability_major));
    try check.expect(major >= 9, "glm-onepass-exact-check: the fused chunk kernel needs sm_90 or later (this GPU: sm_{d}x)", .{major});
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x0e1);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    var opk = try op.Kernels.load(gpu.d, cuda.kernels.glm_onepass, major);
    defer opk.unload();
    var ppk = try op.PipeKernels.load(gpu.d, cuda.kernels.glm_onepass_pipe, major);
    defer ppk.unload();
    const c = try caches(&r, .{ .glue = &ks });
    var fails: usize = 0;
    // rows >= 5 at 17-24 heads: the default uses 32-head blocks (takesExact's shapes); one and four rows stay on chunks
    for ([_][2]usize{ .{ 5, 22 }, .{ 5, 21 }, .{ 16, 24 }, .{ 24, 17 }, .{ 70, 22 } }) |x| fails += try one(&r, &ks, opk.f, ppk.exact, c, x[0], x[1]);
    try check.expect(fails == 0, "glm-onepass-exact-check: {d} checks failed", .{fails});
    check.pass("glm-onepass-exact-check: tf_glm_onepass_exact equals the chunk programs plus merge byte for byte; the existing one-pass differs", .{});
}
