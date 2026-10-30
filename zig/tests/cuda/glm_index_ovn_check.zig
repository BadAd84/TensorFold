//! glm-index-ovn-check (GPU): TF_GLM_OVN_DECODE_INDEX, _INDEX_SCHED and _POOLS32 against today's kernels, byte equal.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const base = @import("glm_index_check.zig");
const sp = glm.sparse;

const d = base.d;
const Case = base.Case;
const Bufs = base.Bufs;

/// Decode windows the split takes: fewer than 8 rows over 8,192 pools or more.
const decode = [_]Case{
    .{ .cap = 8192, .pos = 4 * 8192 - 600, .rows = 5 },
    .{ .cap = 16384, .pos = 4 * 16384 - 40, .rows = 1 },
    .{ .cap = 65536, .pos = 4 * 65536 - 9, .rows = 3, .paged = true },
    .{ .cap = 147456, .pos = 4 * 147456 - 9, .rows = 7, .bits = true }, // a cap off the power of two
    .{ .cap = 262144, .pos = 4 * 149000, .rows = 4 }, // 596k tokens, the agents' depth
    .{ .cap = 262144, .pos = 4 * 262144 - 20, .rows = 2, .paged = true }, // 1M tokens
};
/// Prompt blocks for the scheduled scorer: a tail block under 8 rows' multiple, pages, fewer heads.
const prompt = [_]Case{
    .{ .cap = 16384, .pos = 4 * 16384 - 1130, .rows = 1100 },
    .{ .cap = 65536, .pos = 4 * 65536 - 520, .rows = 512, .paged = true },
    .{ .cap = 147456, .pos = 4 * 147456 - 520, .rows = 512, .heads = 20 },
};

/// A case's device inputs: index queries, head weights, pooled keys (on shuffled pages if c.paged) and the position.
pub const Inputs = struct { q: u64, w: u64, k: u64, t: glm.latent.Table, pos: u64 };

pub fn inputs(gpu: check.Gpu, bufs: *Bufs, rand: std.Random, c: Case, np: usize) !Inputs {
    const gpa = gpu.gpa;
    const qi = try gpa.alloc(u16, c.rows * c.heads * d);
    defer gpa.free(qi);
    for (qi) |*x| x.* = base.bf16(rand.floatNorm(f32));
    const wts = try gpa.alloc(u16, c.rows * c.heads);
    defer gpa.free(wts);
    for (wts) |*x| x.* = base.bf16(rand.floatNorm(f32));
    const pk = try gpa.alloc(u16, c.cap * d);
    defer gpa.free(pk);
    for (pk) |*x| x.* = base.bf16(rand.floatNorm(f32) * 2);
    var in: Inputs = .{ .q = try base.fromHost(gpu, bufs, std.mem.sliceAsBytes(qi)), .w = try base.fromHost(gpu, bufs, std.mem.sliceAsBytes(wts)), .k = 0, .t = .{}, .pos = 0 };
    if (c.paged) { // as glm-index-check lays them out: the table long enough for the token lists' rows too
        const page_rows = base.page_bytes / (512 * 2);
        const entries = @max((c.cap + 127) / 128, (4 * np + page_rows - 1) / page_rows + 8);
        const ids = try gpa.alloc(i32, entries);
        defer gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        rand.shuffle(i32, ids);
        const arena = try gpa.alloc(u8, entries * base.page_bytes);
        defer gpa.free(arena);
        @memset(arena, 0);
        for (0..c.cap) |p| @memcpy(arena[@as(usize, @intCast(ids[p / 128])) * base.page_bytes + (p % 128) * d * 2 ..][0 .. d * 2], std.mem.sliceAsBytes(pk[p * d ..][0..d]));
        in.k = try base.fromHost(gpu, bufs, arena);
        in.t = .{ .pt = try base.fromHost(gpu, bufs, std.mem.sliceAsBytes(ids)), .ps = base.page_bytes, .page_rows = page_rows };
    } else in.k = try base.fromHost(gpu, bufs, std.mem.sliceAsBytes(pk));
    const p0: i32 = @intCast(c.pos);
    in.pos = try base.fromHost(gpu, bufs, std.mem.asBytes(&p0));
    return in;
}

fn scratch(gpu: check.Gpu, bufs: *Bufs, rows: usize, np: usize) !sp.Scratch {
    const ls = @max(rows * sp.select_list * 4, glm.index.splitBytes(rows, np));
    return .{ .scores = try base.zeros(gpu, bufs, @min(rows, sp.select_rows) * np * 4), .ls = try base.zeros(gpu, bufs, ls), .li = try base.zeros(gpu, bufs, 4), .pools = try base.zeros(gpu, bufs, rows * sp.topk_pools * 8), .at = try base.zeros(gpu, bufs, 4) };
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    const ks = try glm.glue.Kernels.load(glue_mod);
    var idx = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .all); // the knobs unset: today's kernels
    defer idx.unload();
    var on = idx; // the knobs' kernels, set as their loads set them
    on.split_rows = glm.index.split_floor_rows;
    on.split_fn = try idx.module.function("tf_glm_index_split");
    on.sched_fn = try idx.module.function("tf_glm_index_scores_sched");
    var rng = std.Random.DefaultPrng.init(0x0b5);
    const rand = rng.random();
    var fails: usize = 0;
    for (decode) |c| fails += try decodeCase(gpu, s, rand, ops, &ks, &on, c);
    for (prompt) |c| fails += try promptCase(gpu, s, rand, &idx, &on, c);
    fails += try pools32(gpu, s, rand, ops, false);
    fails += try pools32(gpu, s, rand, ops, true);
    try check.expect(fails == 0, "glm-index-ovn-check: {d} buffers differ", .{fails});
    check.pass("glm-index-ovn-check: split select, scheduled scores and int32 pools byte-equal at {d} cases", .{decode.len + prompt.len + 2});
}

/// selectTokens today against the split (tokens and counts), then tied and random scores through both selects.
fn decodeCase(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, ops: cuda.Module, ks: *const glm.glue.Kernels, on: *const glm.index.Kernels, c: Case) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const np = sp.poolBucket(c.pos, c.rows, c.cap);
    try check.expect(on.splits(c.rows, np), "glm-index-ovn-check: {d} rows over {d} pools must take the split", .{ c.rows, np });
    const in = try inputs(gpu, &bufs, rand, c, np);
    const sa = try scratch(gpu, &bufs, c.rows, np);
    const sb = try scratch(gpu, &bufs, c.rows, np);
    const tok = c.rows * sp.width * 4;
    const ta = try base.zeros(gpu, &bufs, tok);
    const tb = try base.zeros(gpu, &bufs, tok);
    const ca = try base.zeros(gpu, &bufs, c.rows * 4);
    const cb = try base.zeros(gpu, &bufs, c.rows * 4);
    const old: glm.glue.Set = .{ .glue = ks };
    const new: glm.glue.Set = .{ .glue = ks, .index = on };
    try sp.selectTokens(old, ops, s, in.q, in.w, c.heads, in.k, in.t, in.pos, c.pos, c.rows, c.heads, d, c.cap, sa, ta, ca);
    try sp.selectTokens(new, ops, s, in.q, in.w, c.heads, in.k, in.t, in.pos, c.pos, c.rows, c.heads, d, c.cap, sb, tb, cb);
    try s.synchronize();
    var fails = try base.same(gpu, "split tokens", ta, tb, tok, np, c);
    fails += try base.same(gpu, "split counts", ca, cb, c.rows * 4, np, c);
    const host = try gpu.gpa.alloc(f32, c.rows * np);
    defer gpu.gpa.free(host);
    const special = [_]f32{ std.math.nan(f32), -std.math.nan(f32), 0.0, -0.0, std.math.inf(f32), -std.math.inf(f32) };
    const assemble = try ops.function("tf_glm_dsa_tokens");
    for (0..2) |round| {
        for (0..c.rows) |r| {
            const vis = (c.pos + r + 1) / 4;
            for (host[r * np ..][0..np], 0..) |*x, i| {
                x.* = if (i >= vis) -std.math.inf(f32) else if (round == 0) @as(f32, @floatFromInt(rand.uintLessThan(u32, 16))) * 0.25 else rand.floatNorm(f32);
                if (c.bits and i < vis and rand.uintLessThan(u32, 8) == 0) x.* = special[rand.uintLessThan(usize, special.len)];
            }
        }
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = sa.scores, .len = c.rows * np * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(host));
        var sel: cuda.Args = .{};
        inline for (.{ sa.scores, sa.pools, @as(i64, @intCast(np)), in.pos, @as(c_int, sp.topk_pools) }) |v| sel.add(v);
        try cuda.launch.launch(ks.select_rows, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 128 } }, s, &sel);
        var tk: cuda.Args = .{};
        inline for (.{ sa.pools, in.pos, @as(c_int, sp.topk_pools), ta, ca, in.t.pt, @as(i64, @intCast(in.t.page_rows)) }) |v| tk.add(v);
        try cuda.launch.launch(assemble, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 256 } }, s, &tk);
        try on.selectSplit(s, sa.scores, in.pos, c.rows, np, sb.ls, tb, cb, in.t.pt, in.t.page_rows);
        try s.synchronize();
        fails += try base.same(gpu, if (round == 0) "tied split tokens" else "random split tokens", ta, tb, tok, np, c);
        fails += try base.same(gpu, "split counts", ca, cb, c.rows * 4, np, c);
    }
    return fails;
}

/// Every 512-row block's scores from tf_glm_index_scores and tf_glm_index_scores_sched.
fn promptCase(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, idx: *const glm.index.Kernels, on: *const glm.index.Kernels, c: Case) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const np = sp.poolBucket(c.pos, c.rows, c.cap);
    const in = try inputs(gpu, &bufs, rand, c, np);
    const scale: f32 = @floatCast(std.math.pow(f64, d, -0.5));
    const first: usize = @min(c.rows, sp.select_rows);
    const out_a = try base.zeros(gpu, &bufs, first * np * 4);
    const out_b = try base.zeros(gpu, &bufs, first * np * 4);
    var fails: usize = 0;
    var a: usize = 0;
    while (a < c.rows) : (a += sp.select_rows) {
        const n: usize = @min(sp.select_rows, c.rows - a);
        if (n < glm.index.rows_from) continue; // the scheduled scorer takes prompt-sized launches only
        const at_v: i32 = @intCast(c.pos + a);
        const at = try base.fromHost(gpu, &bufs, std.mem.asBytes(&at_v));
        const nb = (sp.scoreColumns(c.pos + a, n, np) + 63) / 64;
        const q = in.q + a * c.heads * d * 2;
        const w = in.w + a * c.heads * 2;
        try idx.scores(s, q, w, c.heads, in.k, out_a, at, n, np, nb, scale, sp.headScale(c.heads), c.heads, in.t.pt, in.t.ps);
        try on.scores(s, q, w, c.heads, in.k, out_b, at, n, np, nb, scale, sp.headScale(c.heads), c.heads, in.t.pt, in.t.ps);
        try s.synchronize();
        fails += try base.same(gpu, "scheduled scores", out_a, out_b, n * np * 4, np, c);
    }
    return fails;
}

/// tf_glm_dsa_tokens on int64 pools against tf_glm_pools_i32 then tf_glm_dsa_tokens_i32 (ids up to 1M tokens' pools).
fn pools32(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, ops: cuda.Module, paged: bool) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const c: Case = .{ .cap = 262144, .pos = 4 * 262144 - 400, .rows = 300, .paged = paged };
    const k = sp.topk_pools;
    const ids = try gpu.gpa.alloc(i64, c.rows * k);
    defer gpu.gpa.free(ids);
    for (ids) |*v| v.* = rand.uintLessThan(u32, @intCast(c.cap));
    const page_rows = base.page_bytes / (512 * 2);
    var t: glm.latent.Table = .{};
    if (paged) {
        const table = try gpu.gpa.alloc(i32, (4 * c.cap + 511) / 512 + 8);
        defer gpu.gpa.free(table);
        for (table, 0..) |*v, i| v.* = @intCast(i);
        rand.shuffle(i32, table);
        t = .{ .pt = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(table)), .ps = base.page_bytes, .page_rows = page_rows };
    }
    const p0: i32 = @intCast(c.pos);
    const pos = try base.fromHost(gpu, &bufs, std.mem.asBytes(&p0));
    const wide = try base.fromHost(gpu, &bufs, std.mem.sliceAsBytes(ids));
    const narrow = try base.zeros(gpu, &bufs, c.rows * k * 4);
    const tok = c.rows * sp.width * 4;
    const ta = try base.zeros(gpu, &bufs, tok);
    const tb = try base.zeros(gpu, &bufs, tok);
    const ca = try base.zeros(gpu, &bufs, c.rows * 4);
    const cb = try base.zeros(gpu, &bufs, c.rows * 4);
    var pa: cuda.Args = .{};
    inline for (.{ wide, narrow, @as(i64, @intCast(c.rows * k)) }) |v| pa.add(v);
    try cuda.launch.launch(try ops.function("tf_glm_pools_i32"), .{ .grid = .{ .x = @intCast((c.rows * k + 255) / 256) }, .block = .{ .x = 256 } }, s, &pa);
    inline for (.{ .{ "tf_glm_dsa_tokens", wide, ta, ca }, .{ "tf_glm_dsa_tokens_i32", narrow, tb, cb } }) |x| {
        var args: cuda.Args = .{};
        inline for (.{ x[1], pos, @as(c_int, k), x[2], x[3], t.pt, @as(i64, @intCast(t.page_rows)) }) |v| args.add(v);
        try cuda.launch.launch(try ops.function(x[0]), .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 256 } }, s, &args);
    }
    try s.synchronize();
    return try base.same(gpu, "int32 pools' tokens", ta, tb, tok, sp.width, c) + try base.same(gpu, "int32 pools' counts", ca, cb, c.rows * 4, 1, c);
}
