//! glm-index-regroup-check (GPU): INDEX_REGROUP's and INDEX_SCHED's scores and pools against today's, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const base = @import("glm_index_check.zig");
const ovn = @import("glm_index_ovn_check.zig");
const sp = glm.sparse;

const d = base.d;
const Case = base.Case;
const Bufs = base.Bufs;

/// Prompt chunks at 32k, 200k, 600k positions: 2,048 / 4,096 rows, pieces from rows_from (64), pages, 20 heads.
const cases = [_]Case{
    .{ .cap = 4096, .pos = 2050, .rows = 512 },
    .{ .cap = 16384, .pos = 32768, .rows = 2048 },
    .{ .cap = 16384, .pos = 32768 - 333, .rows = 333, .heads = 20 },
    .{ .cap = 65536, .pos = 200000, .rows = 4096, .paged = true },
    .{ .cap = 65536, .pos = 200003, .rows = 64 },
    .{ .cap = 262144, .pos = 600000, .rows = 2048, .paged = true },
    .{ .cap = 262144, .pos = 600001, .rows = 100, .heads = 20 },
    .{ .cap = 262144, .pos = 600000 - 1130, .rows = 1100 }, // a 76-row tail block
};

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var idx = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .both); // the knobs unset: today's kernels
    defer idx.unload();
    try check.expect(idx.regroup_fn == null and idx.sched_fn == null, "glm-index-regroup-check: run with TF_GLM_OVN_INDEX_* unset", .{});
    var sched = idx;
    sched.sched_fn = try idx.module.function("tf_glm_index_scores_sched");
    var regroup = sched; // both knobs set, as production runs ots0109 (it subsumes 0103's schedule)
    regroup.regroup_fn = try idx.module.function("tf_glm_index_scores_regroup");
    try check.expect(regroup.scorer(glm.index.rows_from) == .regroup, "glm-index-regroup-check: the regroup scorer must be taken", .{});
    var rng = std.Random.DefaultPrng.init(0x109);
    const rand = rng.random();
    var fails: usize = 0;
    for (cases) |c| fails += try one(gpu, s, rand, &idx, &sched, &regroup, c);
    try check.expect(fails == 0, "glm-index-regroup-check: {d} buffers differ", .{fails});
    check.pass("glm-index-regroup-check: regrouped and scheduled scores and their pools byte-equal at {d} cases", .{cases.len});
}

/// Every 512-row block's scores from the three scorers, then each one's pools through glm_index_pools.
fn one(gpu: check.Gpu, s: cuda.Stream, rand: std.Random, idx: *const glm.index.Kernels, sched: *const glm.index.Kernels, regroup: *const glm.index.Kernels, c: Case) !usize {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const np = sp.poolBucket(c.pos, c.rows, c.cap);
    const in = try ovn.inputs(gpu, &bufs, rand, c, np);
    const scale: f32 = @floatCast(std.math.pow(f64, d, -0.5));
    const first: usize = @min(c.rows, sp.select_rows);
    const arms = [_]*const glm.index.Kernels{ idx, sched, regroup };
    var out: [3]u64 = undefined;
    var pools: [3]u64 = undefined;
    for (0..3) |i| {
        out[i] = try base.zeros(gpu, &bufs, first * np * 4);
        pools[i] = try base.zeros(gpu, &bufs, first * sp.topk_pools * 8);
    }
    var fails: usize = 0;
    var a: usize = 0;
    while (a < c.rows) : (a += sp.select_rows) {
        const n: usize = @min(sp.select_rows, c.rows - a);
        if (n < glm.index.rows_from) continue; // the regrouped scorer takes prompt-sized launches only
        const at_v: i32 = @intCast(c.pos + a);
        const at = try base.fromHost(gpu, &bufs, std.mem.asBytes(&at_v));
        const nb = (sp.scoreColumns(c.pos + a, n, np) + 63) / 64;
        const q = in.q + a * c.heads * d * 2;
        const w = in.w + a * c.heads * 2;
        for (arms, 0..) |k, i| {
            // a NaN fill: a column the scorer skips would read as a difference, not as zeros both arms left alone
            try s.d.check(s.d.api.cuMemsetD32Async(out[i], 0x7fc00001, first * np, s.handle), "cuMemsetD32Async");
            try k.scores(s, q, w, c.heads, in.k, out[i], at, n, np, nb, scale, sp.headScale(c.heads), c.heads, in.t.pt, in.t.ps);
            try k.pools(s, out[i], pools[i], at, n, np, 0);
        }
        try s.synchronize();
        for (1..3) |i| {
            const what = if (i == 1) "scheduled" else "regrouped";
            std.debug.print("  [np {d} rows {d} at {d}] {s}\n", .{ np, n, c.pos + a, what });
            fails += try base.same(gpu, if (i == 1) "scheduled scores" else "regrouped scores", out[0], out[i], n * np * 4, np, c);
            fails += try base.same(gpu, if (i == 1) "scheduled pools" else "regrouped pools", pools[0], pools[i], n * sp.topk_pools * 8, sp.topk_pools, c);
        }
    }
    return fails;
}
