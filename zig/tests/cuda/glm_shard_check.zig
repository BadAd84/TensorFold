//! glm-shard-check (GPU): TF_GLM_OVN_DECODE_SHARD's ranks emulated on one GPU (each its share and pairs, the gather's
//! buffer, the merge) against today's selectTokens and tf_glm_select_rows + tf_glm_dsa_tokens, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const base = @import("glm_index_check.zig");
const ovn = @import("glm_index_ovn_check.zig");
const sp = glm.sparse;
const sh = glm.index_shard;

const d = base.d;
const Case = base.Case;
const Bufs = base.Bufs;

/// Decode windows from the dense rows to 1M tokens; every case runs over two, three and four ranks.
const cases = [_]Case{
    .{ .cap = 1024, .pos = 300, .rows = 3 }, // under 512 visible pools: dense rows, three ranks leave the last share empty
    .{ .cap = 4096, .pos = 2100, .rows = 2 }, // shares under 512 pools: each rank sends every pool it sees
    .{ .cap = 16384, .pos = 4 * 16384 - 40, .rows = 7, .paged = true },
    .{ .cap = 65536, .pos = 133000, .rows = 4, .heads = 20 }, // 133k tokens
    .{ .cap = 65536, .pos = 4 * 65536 - 9, .rows = 3, .paged = true }, // 262k tokens
    .{ .cap = 262144, .pos = 4 * 149000, .rows = 2 }, // 596k tokens, the agents' depth, a 2-row verify
    .{ .cap = 262144, .pos = 4 * 262144 - 20, .rows = 8, .paged = true }, // 1M tokens
};
const worlds = [_]usize{ 2, 3, 4 };

fn scratch(gpu: check.Gpu, bufs: *Bufs, rows: usize, np: usize) !sp.Scratch {
    return .{ .scores = try base.zeros(gpu, bufs, rows * np * 4), .ls = try base.zeros(gpu, bufs, rows * sp.select_list * 4), .li = try base.zeros(gpu, bufs, rows * sp.select_list * 4), .pools = try base.zeros(gpu, bufs, rows * sp.topk_pools * 8), .at = try base.zeros(gpu, bufs, 4) };
}

/// n bytes of 0x7F (a large score, a pool past every cache): a byte the sharded path should have written and did not
/// shows up in the comparison.
fn sentinel(gpu: check.Gpu, s: cuda.Stream, ptr: u64, n: usize) !void {
    try s.synchronize();
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = ptr, .len = n, .borrowed = true }).fill8(0x7F, null);
    try gpu.d.synchronize(); // the memset runs on the default stream, not on s
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    const ks = try glm.glue.Kernels.loadOvn(glue_mod, true); // with the TF_GLM_OVN forms: the shard's pick and merge
    var idx = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .all); // decode windows on tf_glm_index_scores
    defer idx.unload();
    const old: glm.glue.Set = .{ .glue = &ks };
    const all: glm.glue.Set = .{ .glue = &ks, .index = &idx };
    var rng = std.Random.DefaultPrng.init(0x5a4d);
    const rand = rng.random();
    var fails: usize = 0;
    for (cases) |c| fails += try case(gpu, s, rand, ops, old, all, c);
    try check.expect(fails == 0, "glm-shard-check: {d} buffers differ", .{fails});
    check.pass("glm-shard-check: sharded decode tokens byte-equal to selectTokens at {d} cases over 2, 3 and 4 ranks, both scorers, tied and special scores", .{cases.len});
}

/// One case: today's selectTokens, then every rank's share from qi and the keys (both scorers) merged, then the tied
/// and special scores; each against today's bytes.
fn case(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, ops: cuda.Module, old: glm.glue.Set, all: glm.glue.Set, c: Case) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const np = sp.poolBucket(c.pos, c.rows, c.cap);
    const cols = sp.scoreColumns(c.pos, c.rows, np);
    const in = try ovn.inputs(gpu, &bufs, rand, c, np);
    const tok = c.rows * sp.width * 4;
    const sa = try scratch(gpu, &bufs, c.rows, np);
    const ta = try base.zeros(gpu, &bufs, tok);
    const ca = try base.zeros(gpu, &bufs, c.rows * 4);
    const tb = try base.zeros(gpu, &bufs, tok);
    const cb = try base.zeros(gpu, &bufs, c.rows * 4);
    const bytes = sh.pairBytes(c.rows);
    const recv = try base.zeros(gpu, &bufs, worlds[worlds.len - 1] * bytes);
    var ranks: [worlds[worlds.len - 1]]sp.Scratch = undefined;
    for (&ranks) |*x| x.* = try scratch(gpu, &bufs, c.rows, np);
    try sp.selectTokens(old, ops, s, in.q, in.w, c.heads, in.k, in.t, in.pos, c.pos, c.rows, c.heads, d, c.cap, sa, ta, ca);
    try s.synchronize();
    var fails: usize = 0;
    for (worlds) |world| {
        std.debug.print("  [np {d} rows {d} heads {d} ranks {d}{s}]\n", .{ np, c.rows, c.heads, world, if (c.paged) " paged" else "" });
        for ([_]glm.glue.Set{ old, all }, [_][]const u8{ "shard tokens (tf_glm_scores)", "shard tokens (tf_glm_index_scores)" }) |set, what| {
            for (ranks[0..world]) |x| try sentinel(gpu, s, x.scores, c.rows * np * 4); // each rank's own scores only
            try sentinel(gpu, s, recv, world * bytes);
            try sentinel(gpu, s, tb, tok);
            try sentinel(gpu, s, cb, c.rows * 4);
            for (ranks[0..world], 0..) |x, r| try sh.share(set, ops, s, r, world, in.q, in.w, c.heads, in.k, in.t, in.pos, c.pos, c.rows, c.heads, d, c.cap, x, recv + r * bytes, null);
            try sh.merge(set, s, recv, c.rows, world, in.pos, tb, cb, in.t);
            try s.synchronize();
            fails += try base.same(gpu, what, ta, tb, tok, np, c);
            fails += try base.same(gpu, "shard counts", ca, cb, c.rows * 4, np, c);
        }
    }
    return fails + try ties(gpu, s, rand, ops, old, c, np, cols, in, sa, ta, ca, tb, cb, recv);
}

/// Tied (16 values) and random scores with NaN, -NaN, +-0 and +-inf, -inf past each row's visible pools: today's
/// tf_glm_select_rows then tf_glm_dsa_tokens against every rank's pick of the same scores and the merge.
fn ties(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, ops: cuda.Module, set: glm.glue.Set, c: Case, np: usize, cols: usize, in: ovn.Inputs, sa: sp.Scratch, ta: u64, ca: u64, tb: u64, cb: u64, recv: u64) !usize {
    const host = try gpu.gpa.alloc(f32, c.rows * np);
    defer gpu.gpa.free(host);
    const special = [_]f32{ std.math.nan(f32), -std.math.nan(f32), 0.0, -0.0, std.math.inf(f32), -std.math.inf(f32) };
    const assemble = try ops.function("tf_glm_dsa_tokens");
    const tok = c.rows * sp.width * 4;
    const bytes = sh.pairBytes(c.rows);
    var fails: usize = 0;
    for (0..2) |round| {
        for (0..c.rows) |r| {
            const vis = (c.pos + r + 1) / 4;
            for (host[r * np ..][0..np], 0..) |*x, i| {
                x.* = if (i >= vis) -std.math.inf(f32) else if (round == 0) @as(f32, @floatFromInt(rand.uintLessThan(u32, 16))) * 0.25 else rand.floatNorm(f32);
                if (round == 1 and i < vis and rand.uintLessThan(u32, 8) == 0) x.* = special[rand.uintLessThan(usize, special.len)];
            }
        }
        try s.synchronize();
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = sa.scores, .len = c.rows * np * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(host));
        var sel: cuda.Args = .{};
        inline for (.{ sa.scores, sa.pools, @as(i64, @intCast(np)), in.pos, @as(c_int, sp.topk_pools) }) |v| sel.add(v);
        try cuda.launch.launch(set.glue.select_rows, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 128 } }, s, &sel);
        var tk: cuda.Args = .{};
        inline for (.{ sa.pools, in.pos, @as(c_int, sp.topk_pools), ta, ca, in.t.pt, @as(i64, @intCast(in.t.page_rows)) }) |v| tk.add(v);
        try cuda.launch.launch(assemble, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 256 } }, s, &tk);
        for (worlds) |world| {
            try sentinel(gpu, s, recv, world * bytes);
            try sentinel(gpu, s, tb, tok);
            try sentinel(gpu, s, cb, c.rows * 4);
            for (0..world) |r| try sh.pick(set, s, sa.scores, np, sh.part(cols, r, world), in.pos, c.rows, recv + r * bytes);
            try sh.merge(set, s, recv, c.rows, world, in.pos, tb, cb, in.t);
            try s.synchronize();
            fails += try base.same(gpu, if (round == 0) "tied shard tokens" else "special shard tokens", ta, tb, tok, np, c);
            fails += try base.same(gpu, "shard counts", ca, cb, c.rows * 4, np, c);
        }
    }
    return fails;
}
