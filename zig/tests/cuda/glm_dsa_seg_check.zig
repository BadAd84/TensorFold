//! glm-dsa-seg (GPU): TF_GLM_OVN_SEG_DSA's SegDsa.attend against dsa_block.attend a segment at a time, byte for byte:
//! attention rows, tokens, counts and every cache; slab and pages, dense, straddling and sparse rows, narrow and wide
//! selects, 16- and 32-head blocks, BF16 and E4M3 caches, at layer 1 of two (the table's layer stride).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const dsa = glm.dsa_block;
const latent = glm.latent;
const sparse = glm.sparse;
const seg = glm.dsa_seg;
const glue = glm.glue;

const heads = 22; // TP=3's widest rank
const lw = 512;
const qk = 256;
const ih = 32; // index heads
const id = 128; // index width
const kw = id + ih;
const capacity = 16384; // the paged segments': up to 4,096 pools, so the deepest selects wide
const slab_cap = 8192; // the slab segments' (2,048 pools: their selects stay narrow)
const layers = 2;
const li = 1;
const shared_rows = 32; // Engine.shared_rows
const window_rows = glm.forward.window_rows;
const nch = latent.chunksFor(@min(capacity, 2560) + window_rows); // a shared Buffers' nch
const pages = capacity / 512;

const Spec = struct { rows: usize, pos: usize, paged: bool };
const specs = [_]Spec{
    .{ .rows = 3, .pos = 700, .paged = true }, // dense on pages, 16-head blocks
    .{ .rows = 1, .pos = 5000, .paged = false }, // sparse on a slab, narrow select
    .{ .rows = 8, .pos = 2046, .paged = false }, // dense rows then sparse ones (dense limit 2051), 32-head blocks
    .{ .rows = 5, .pos = 12000, .paged = true }, // sparse on pages, wide select (4,096 pools)
};
const n = specs.len;
const rows = 17;

/// One run's caches: each segment's at both layers, and the buffers compared by part and layer (0: not compared).
const Run = struct { caches: [n][layers]dsa.Caches, bufs: [n][4][2]u64 };

/// E4M3 codes without NaN, or power-of-two scales.
fn codes(r: *Rig, count: usize) ![]u8 {
    const out = try r.a.alloc(u8, count);
    for (out) |*c| {
        var b = r.rng.int(u8);
        if (b & 0x7f == 0x7f) b ^= 1;
        c.* = b;
    }
    return out;
}

fn scales(r: *Rig, count: usize) ![]u8 {
    const out = try r.a.alloc(f32, count);
    for (out) |*s| s.* = std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(r.rng.uintLessThan(u32, 8))));
    return std.mem.sliceAsBytes(out);
}

fn bfBytes(r: *Rig, count: usize) ![]u8 {
    return std.mem.sliceAsBytes(try r.bfs(count, 1.0));
}

/// A latent slab of `slab_cap` rows: BF16, or E4M3 rows followed by their scales (State's layout).
fn slab(r: *Rig, fp8: bool) ![]u8 {
    if (!fp8) return bfBytes(r, slab_cap * lw);
    return std.mem.concat(r.a, u8, &.{ try codes(r, slab_cap * lw), try scales(r, slab_cap) });
}

fn capOf(sp: Spec) usize {
    return if (sp.paged) capacity else slab_cap;
}

/// One layer's part of a page: its 512 latent rows (BF16, or codes then scales), then their 128 pools' keys.
fn partBytes(fp8: bool) usize {
    return (if (fp8) @as(usize, 512 * lw + 512 * 4) else @as(usize, 512 * lw * 2)) + 128 * id * 2;
}

/// Every page's parts for both layers, in State.place's order.
fn arena(r: *Rig, fp8: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (0..pages * layers) |_| {
        try out.appendSlice(r.a, if (fp8) try codes(r, 512 * lw) else try bfBytes(r, 512 * lw));
        if (fp8) try out.appendSlice(r.a, try scales(r, 512));
        try out.appendSlice(r.a, try bfBytes(r, 128 * id));
    }
    return out.items;
}

/// Both runs' caches from the same bytes; pages through a permuted table.
fn runs(r: *Rig, fp8: bool, pt: u64) ![2]Run {
    var out: [2]Run = undefined;
    for (specs, 0..) |sp, i| {
        const ps = layers * partBytes(fp8);
        const none: []const u8 = &.{};
        const kv: []const u8 = if (sp.paged) try arena(r, fp8) else none;
        const lat = [layers][]const u8{ if (sp.paged) none else try slab(r, fp8), if (sp.paged) none else try slab(r, fp8) };
        var ik: [layers][]const u8 = undefined;
        var ig: [layers][]const u8 = undefined;
        var pk: [layers][]const u8 = undefined;
        const cap = capOf(sp);
        for (0..layers) |j| {
            ik[j] = try bfBytes(r, cap * id);
            ig[j] = try bfBytes(r, cap * id);
            pk[j] = if (sp.paged) none else try bfBytes(r, (cap / 4 + 2) * id);
        }
        for (&out) |*one| {
            const base = if (sp.paged) try r.dev(kv) else 0;
            for (0..layers) |j| {
                var c: dsa.Caches = .{ .lc = 0, .ik = try r.dev(ik[j]), .ig = try r.dev(ig[j]), .pk = 0, .pk_rows = cap / 4 + 2 };
                if (sp.paged) {
                    c.lc = base + j * partBytes(fp8);
                    c.ls = if (fp8) c.lc + 512 * lw else 0;
                    c.pk = c.lc + partBytes(fp8) - 128 * id * 2;
                    c.pt = pt;
                    c.ps = ps;
                } else {
                    c.lc = try r.dev(lat[j]);
                    c.ls = if (fp8) c.lc + cap * lw else 0;
                    c.pk = try r.dev(pk[j]);
                }
                one.caches[i][j] = c;
                one.bufs[i][0][j] = c.lc;
                one.bufs[i][1][j] = c.ik;
                one.bufs[i][2][j] = c.ig;
                one.bufs[i][3][j] = c.pk;
            }
            if (sp.paged) one.bufs[i][3] = .{ 0, 0 }; // the pooled keys are in the arena, compared as a whole
            if (sp.paged) one.bufs[i][0] = .{ base, 0 };
        }
    }
    return out;
}

fn bytesOf(sp: Spec, fp8: bool, part: usize) usize {
    const cap = capOf(sp);
    return switch (part) {
        0 => if (sp.paged) pages * layers * partBytes(fp8) else cap * (if (fp8) @as(usize, lw + 4) else @as(usize, lw * 2)),
        1, 2 => cap * id * 2,
        else => (cap / 4 + 2) * id * 2,
    };
}

const Inputs = struct { lat: u64, ikr: u64, igr: u64, qa: u64, qi: u64 };

/// A window's scratch over the shared inputs, its outputs (0xff) its own.
fn scratch(r: *Rig, in: Inputs) !dsa.Scratch {
    var s = std.mem.zeroes(dsa.Scratch);
    s.lat = in.lat;
    s.ikr = in.ikr;
    s.igr = in.igr;
    s.qa = in.qa;
    s.qi = in.qi;
    s.ol = try r.zeros(rows * heads * lw * 2);
    const slots = @max(nch, latent.chunksFor(sparse.width)) * window_rows * heads; // a shared Buffers' partial rows
    s.partials = .{ .po = try r.zeros(slots * lw * 4), .pm = try r.zeros(slots * 4), .pl = try r.zeros(slots * 4) };
    s.dummy = try r.zeros(4);
    s.select = .{ .scores = try r.zeros(window_rows * (capacity / 4) * 4), .ls = try r.zeros(16), .li = try r.zeros(16), .pools = try r.zeros(shared_rows * sparse.topk_pools * 8), .at = try r.zeros(4) };
    s.tokens = try r.zeros(shared_rows * sparse.width * 4);
    s.counts = try r.zeros(shared_rows * 4);
    s.at = try r.zeros(4);
    return s;
}

fn window(r: *Rig, k: dsa.Kernels, gm: cuda.Module, ops: cuda.Module, l: dsa.Layer, fp8: bool, pt: u64, in: Inputs, pos: [n]u64) !void {
    const both = try runs(r, fp8, pt);
    const sw = try scratch(r, in);
    const sg = try scratch(r, in);
    var row0: [n]usize = undefined;
    var at: usize = 0;
    for (specs, 0..) |sp, i| { // shared.mixer's loop: each segment's attend, its token rows at its own rows
        row0[i] = at;
        var v = sw;
        v.tokens += at * sparse.width * 4;
        v.counts += at * 4;
        const nc = @min(latent.chunksFor(sp.pos + sp.rows), nch);
        try dsa.attend(k, r.s, l, both[0].caches[i][li], v, at, sp.rows, pos[i], sp.pos, nc, window_rows);
        at += sp.rows;
    }
    var x = try seg.SegDsa.initShape(r.gpu.gpa, r.gpu.d, gm, ops, l, layers, shared_rows, nch, capacity);
    defer x.deinit();
    var parts: [n]seg.Part = undefined;
    for (specs, 0..) |sp, i| parts[i] = .{ .row0 = row0[i], .rows = sp.rows, .pos = sp.pos, .pos_dev = pos[i], .caches = &both[1].caches[i] };
    try x.load(&parts, r.s);
    try check.expect(x.takes(n, k.set, l, li) and x.select, "glm-dsa-seg: the window does not take the segmented path", .{});
    try x.attend(k, r.s, l, li, sg);
    const what = if (fp8) "e4m3" else "bf16";
    var name: [96]u8 = undefined;
    const ol = try r.back(sw.ol, rows * heads * lw * 2);
    for (std.mem.bytesAsSlice(u16, ol)) |v| try check.expect(v != 0xffff, "glm-dsa-seg {s}: an attention row left unwritten", .{what});
    try check.sameBytes(try std.fmt.bufPrint(&name, "dsa-seg {s} attention rows", .{what}), try r.back(sg.ol, ol.len), ol);
    try check.sameBytes(try std.fmt.bufPrint(&name, "dsa-seg {s} tokens", .{what}), try r.back(sg.tokens, rows * sparse.width * 4), try r.back(sw.tokens, rows * sparse.width * 4));
    try check.sameBytes(try std.fmt.bufPrint(&name, "dsa-seg {s} counts", .{what}), try r.back(sg.counts, rows * 4), try r.back(sw.counts, rows * 4));
    const parts_name = [_][]const u8{ "latent", "index keys", "gates", "pooled keys" };
    for (specs, 0..) |sp, i| for (0..4) |p| for (0..layers) |j| {
        const want = both[0].bufs[i][p][j];
        if (want == 0) continue;
        const len = bytesOf(sp, fp8, p);
        try check.sameBytes(try std.fmt.bufPrint(&name, "dsa-seg {s} segment {d} {s} layer {d}", .{ what, i, parts_name[p], j }), try r.back(both[1].bufs[i][p][j], len), try r.back(want, len));
    };
}

pub fn run(gpu: check.Gpu) !void {
    var arena_alloc: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena_alloc.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5e9d5a);
    var r: Rig = .{ .gpu = gpu, .a = arena_alloc.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    const ks = try glue.Kernels.loadOvn(gm, false);
    const k: dsa.Kernels = .{ .set = .{ .glue = &ks }, .ops = ops };
    var l: dsa.Layer = undefined;
    l.heads = heads;
    l.qk = qk;
    l.lw = lw;
    l.dense_limit = (glm.config.Config{}).keyWidth();
    l.index.heads = ih;
    l.index.d = id;
    l.index.ln_w = try r.dev(try bfBytes(&r, id));
    l.index.ln_b = try r.dev(try bfBytes(&r, id));
    l.index.ape = try r.dev(try bfBytes(&r, 4 * id));
    const gates = try r.a.alloc(f32, rows * id);
    for (gates) |*g| g.* = r.rng.float(f32) * 4 - 2;
    const in: Inputs = .{ .lat = try r.dev(try bfBytes(&r, rows * lw)), .ikr = try r.dev(try bfBytes(&r, rows * kw)), .igr = try r.dev(std.mem.sliceAsBytes(gates)), .qa = try r.dev(try bfBytes(&r, rows * heads * lw)), .qi = try r.dev(try bfBytes(&r, rows * ih * id)) };
    var table: [pages]i32 = undefined;
    for (&table, 0..) |*t, i| t.* = @intCast((i * 7 + 3) % pages);
    const pt = try r.dev(std.mem.sliceAsBytes(&table));
    var pos: [n]u64 = undefined;
    for (specs, &pos) |sp, *p| {
        const v: i32 = @intCast(sp.pos);
        p.* = try r.dev(std.mem.asBytes(&v));
    }
    for ([_]bool{ false, true }) |fp8| try window(&r, k, gm, ops, l, fp8, pt, in, pos);
    check.pass("glm-dsa-seg: {d} segments of {d} rows (slab and pages, dense, straddling and sparse, narrow and wide selects), bf16 and e4m3, equal the per-segment attend byte for byte", .{ n, rows });
}

comptime {
    var t = 0;
    for (specs) |sp| t += sp.rows;
    std.debug.assert(t == rows and rows <= shared_rows);
}
