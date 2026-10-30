//! glm-seg-pass (GPU): TF_GLM_OVN_SEG_PASS's chunk partials and merged rows against today's launches, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const latent = glm.latent;
const glue = glm.glue;

const lw = 512;
const width = glm.sparse.width; // a token list's 2,051 keys: five chunks, the last with 3
const page_keys = 512; // positions a page holds (one dense chunk)
const page_rows = 600; // a page part's stride in latent rows: its rows (E4M3: codes, then scales), then foreign bytes
const scale: f32 = 0.0625;
const nan: u16 = 0x7fc0;
const caps = [_]usize{ 6000, 1 << 18 }; // a short cache and one past 200k rows
const heads_list = [_]usize{ 22, 21, 32 }; // TP=3's ranks (one 24-head block), TP=2's (two 16-head blocks)
const counts = [_]i32{ 0, 1, 31, 32, 33, 511, 512, 513, 1024, 2048, 2050, 2051, 3 }; // chunk and tile edges, count 0

/// One cache in the four layouts a decode reads: BF16 or E4M3 rows (+inf in the last key), on a slab or on pages.
const Cache = struct { cap: usize, bf: u64, fp: u64, scales: u64, bf_pages: u64, fp_pages: u64, pt: u64, ids: []const i32 };
const Layout = struct { name: []const u8, lc: u64, ls: u64, t: latent.Table };

fn position(r: *Rig, v: usize) !u64 {
    const p: i32 = @intCast(v);
    return r.dev(std.mem.asBytes(&p));
}

/// Random rows, the E4M3 writer's codes and scales on a slab, and both formats copied onto shuffled pages.
fn build(r: *Rig, set: glue.Set, cap: usize) !Cache {
    const x = try r.bfs(cap * lw, 1.0);
    const fp = try r.zeros(try latent.cacheBytes(cap, lw, true));
    const scales = fp + cap * lw;
    try latent.latentWriteFormat(set, r.s, try r.dev(std.mem.sliceAsBytes(x)), lw, fp, scales, .{}, try position(r, 0), cap, lw);
    x[(cap - 1) * lw + 3] = 0x7f80;
    const codes = try r.back(fp, cap * lw);
    const sc = try r.back(scales, cap * 4);
    const ids = try r.a.alloc(i32, cap / page_keys + 4);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    r.rng.shuffle(i32, ids);
    const bf_part = page_rows * lw * 2;
    const fp_part = page_rows * lw;
    const bfa = try r.a.alloc(u8, ids.len * bf_part);
    const fpa = try r.a.alloc(u8, ids.len * fp_part);
    @memset(bfa, 0x5a); // rows no position owns: a misplaced read changes the bytes
    @memset(fpa, 0x5a);
    const xb = std.mem.sliceAsBytes(x);
    for (0..cap) |p| {
        const at: usize = @intCast(ids[p / page_keys]);
        const off = p % page_keys;
        @memcpy(bfa[at * bf_part + off * lw * 2 ..][0 .. lw * 2], xb[p * lw * 2 ..][0 .. lw * 2]);
        @memcpy(fpa[at * fp_part + off * lw ..][0..lw], codes[p * lw ..][0..lw]);
        @memcpy(fpa[at * fp_part + page_keys * lw + off * 4 ..][0..4], sc[p * 4 ..][0..4]);
    }
    const pages_fp = try r.dev(fpa);
    return .{ .cap = cap, .bf = try r.dev(xb), .fp = fp, .scales = scales, .bf_pages = try r.dev(bfa), .fp_pages = pages_fp, .pt = try r.dev(std.mem.sliceAsBytes(ids)), .ids = ids };
}

fn layouts(c: Cache) [4]Layout {
    return .{
        .{ .name = "bf16 slab", .lc = c.bf, .ls = 0, .t = .{} },
        .{ .name = "e4m3 slab", .lc = c.fp, .ls = c.scales, .t = .{} },
        .{ .name = "bf16 pages", .lc = c.bf_pages, .ls = 0, .t = .{ .pt = c.pt, .ps = page_rows * lw * 2, .page_rows = page_rows } },
        .{ .name = "e4m3 pages", .lc = c.fp_pages, .ls = c.fp_pages + page_keys * lw, .t = .{ .pt = c.pt, .ps = page_rows * lw, .page_rows = page_rows } },
    };
}

/// A case's device buffers, freed when the case ends (the big cache's partials run to hundreds of MB).
const Bufs = struct {
    r: *Rig,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn blank(b: *Bufs, bytes: usize) !u64 { // 0xff: a slot no launch writes fails the comparison
        const x = try cuda.DeviceBuffer.alloc(b.r.gpu.d, @max(bytes, 16));
        try b.list.append(b.r.gpu.gpa, x);
        try x.fill8(0xff, b.r.s.handle);
        return x.ptr;
    }

    fn deinit(b: *Bufs) void {
        for (b.list.items) |*x| x.free();
        b.list.deinit(b.r.gpu.gpa);
    }
};

/// One run's partials and output.
const Run = struct { p: latent.Partials, out: u64 };

fn run1(b: *Bufs, nch: usize, rows: usize, heads: usize) !Run {
    const n = nch * rows * heads;
    return .{ .p = .{ .po = try b.blank(n * lw * 4), .pm = try b.blank(n * 4), .pl = try b.blank(n * 4) }, .out = try b.blank(rows * heads * lw * 2) };
}

/// `got` against `want` byte for byte; on a difference, how many values differ and by how much (fp32 or bf16).
fn same(r: *Rig, what: []const u8, part: []const u8, got: u64, want: u64, bytes: usize, bf16: bool) !bool {
    const gpa = r.gpu.gpa;
    try r.s.synchronize();
    const g = try gpa.alloc(u8, bytes);
    defer gpa.free(g);
    const w = try gpa.alloc(u8, bytes);
    defer gpa.free(w);
    try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = got, .len = bytes, .borrowed = true }).download(0, g);
    try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = want, .len = bytes, .borrowed = true }).download(0, w);
    if (std.mem.eql(u8, g, w)) return true;
    const size: usize = if (bf16) 2 else 4;
    var diff: usize = 0;
    var worst: f64 = 0;
    var top: f64 = 0;
    var i: usize = 0;
    while (i + size <= bytes) : (i += size) {
        const a = value(g[i..][0..size], bf16);
        const e = value(w[i..][0..size], bf16);
        if (std.math.isFinite(e)) top = @max(top, @abs(e));
        if (std.mem.eql(u8, g[i..][0..size], w[i..][0..size])) continue;
        diff += 1;
        if (std.math.isFinite(a) and std.math.isFinite(e)) worst = @max(worst, @abs(a - e));
    }
    std.debug.print("FAIL glm-seg-pass {s} {s}: {d} of {d} values differ, largest |diff| {e:.3} (largest |value| {e:.3})\n", .{ what, part, diff, bytes / size, worst, top });
    return false;
}

fn value(x: []const u8, bf16: bool) f64 {
    if (bf16) return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, x[0..2], .little)) << 16));
    return @as(f32, @bitCast(std.mem.readInt(u32, x[0..4], .little)));
}

/// Today's run, the seg pass and a second seg pass (no race): partials and merged rows, byte for byte.
fn compare(r: *Rig, what: []const u8, runs: [3]Run, nch: usize, rows: usize, heads: usize) !usize {
    const n = nch * rows * heads;
    var fails: usize = 0;
    for (runs[1..], [_][]const u8{ "", " (second run)" }) |x, again| {
        var name: [160]u8 = undefined;
        const w = try std.fmt.bufPrint(&name, "{s}{s}", .{ what, again });
        fails += @intFromBool(!try same(r, w, "po", x.p.po, runs[0].p.po, n * lw * 4, false));
        fails += @intFromBool(!try same(r, w, "pm", x.p.pm, runs[0].p.pm, n * 4, false));
        fails += @intFromBool(!try same(r, w, "pl", x.p.pl, runs[0].p.pl, n * 4, false));
        fails += @intFromBool(!try same(r, w, "merged", x.out, runs[0].out, rows * heads * lw * 2, true));
    }
    return fails;
}

/// Queries with a NaN in row 3 (when there is one): its outputs go NaN alike on both paths.
fn queries(r: *Rig, rows: usize, heads: usize) !u64 {
    const q = try r.bfs(rows * heads * lw, 1.0);
    if (rows > 3) q[3 * heads * lw + 5] = nan;
    return r.dev(std.mem.sliceAsBytes(q));
}

fn dense(r: *Rig, sets: [2]glue.Set, lay: Layout, heads: usize, pos: usize, rows: usize) !usize {
    var b: Bufs = .{ .r = r };
    defer b.deinit();
    const q = try queries(r, rows, heads);
    const at = try position(r, pos);
    const nch = latent.chunksFor(pos + rows);
    const hb = latent.headBlock(rows, heads); // dsa_block's: 16-head blocks below five rows, as the decode launches
    var runs: [3]Run = undefined;
    for (&runs, [_]glue.Set{ sets[0], sets[1], sets[1] }) |*x, set| {
        x.* = try run1(&b, nch, rows, heads);
        try latent.attentionFormat(set, r.s, q, lay.lc, lay.ls, lay.t, at, x.p, x.out, try b.blank(4), rows, heads, lw, nch, scale, hb);
    }
    var name: [128]u8 = undefined;
    return compare(r, try std.fmt.bufPrint(&name, "dense {s} {d} heads, {d} rows at {d}", .{ lay.name, heads, rows, pos }), runs, nch, rows, heads);
}

fn sparse(r: *Rig, sets: [2]glue.Set, lay: Layout, c: Cache, heads: usize, rows: usize, shift: usize) !usize {
    var b: Bufs = .{ .r = r };
    defer b.deinit();
    const tok = try r.a.alloc(i32, rows * width);
    @memset(tok, -1);
    const cnt = try r.a.alloc(i32, rows);
    const paged = lay.t.pt != 0;
    for (cnt, 0..) |*n, row| {
        n.* = counts[(row + shift) % counts.len];
        for (tok[row * width ..][0..@intCast(n.*)]) |*t| {
            const p = r.rng.uintLessThan(usize, c.cap);
            t.* = if (paged) c.ids[p / page_keys] * page_rows + @as(i32, @intCast(p % page_keys)) else @intCast(p);
        }
    }
    const q = try queries(r, rows, heads);
    const dtok = try r.dev(std.mem.sliceAsBytes(tok));
    const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
    const nch = latent.chunksFor(width);
    var runs: [3]Run = undefined;
    for (&runs, [_]glue.Set{ sets[0], sets[1], sets[1] }) |*x, set| {
        x.* = try run1(&b, nch, rows, heads);
        try latent.sparseAttentionFormat(set, r.s, q, lay.lc, lay.ls, lay.t, dtok, dcnt, width, x.p, x.out, rows, heads, lw, scale);
    }
    var name: [128]u8 = undefined;
    return compare(r, try std.fmt.bufPrint(&name, "sparse {s} {d} heads, {d} rows of {d} cache rows", .{ lay.name, heads, rows, c.cap }), runs, nch, rows, heads);
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5e9);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.loadOvn(mod, true);
    const sets = [2]glue.Set{ .{ .glue = &ks }, .{ .glue = &ks, .mla = .{ .seg = true } } };
    var fails: usize = 0;
    var cases: usize = 0;
    for (caps) |cap| {
        const c = try build(&r, sets[0], cap);
        const spots = [_][2]usize{ .{ 0, 1 }, .{ 30, 3 }, .{ 509, 4 }, .{ 1000, 8 }, .{ 2040, 8 }, .{ 4093, 5 }, .{ 200_000, 2 }, .{ cap - 8, 8 } };
        for (layouts(c)) |lay| {
            for (heads_list) |heads| {
                for (spots) |x| {
                    if (x[0] + x[1] > cap) continue;
                    fails += try dense(&r, sets, lay, heads, x[0], x[1]);
                    cases += 1;
                }
                for ([_]usize{ 1, 2, 3, 5, 8 }, 0..) |rows, i| {
                    fails += try sparse(&r, sets, lay, c, heads, rows, i * 3);
                    cases += 1;
                }
            }
        }
    }
    try check.expect(fails == 0, "glm-seg-pass: {d} comparisons differ (FAIL lines above)", .{fails});
    check.pass("glm-seg-pass: {d} dense and sparse windows (1-8 rows, BF16 and E4M3, slab and pages, caches of 6,000 and 262,144 rows) equal today's partials and merged rows, twice", .{cases});
}
