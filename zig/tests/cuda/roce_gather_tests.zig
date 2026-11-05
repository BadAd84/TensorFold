//! Two to four ranks' RoCE all-gather (Group.openRoce) against NCCL's byte for byte, graph replays, call times.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const cap = 512 << 10;
/// Shards a rank: tiny, a few decode rows, the cap, and past it or not a multiple of 16 (those stay on NCCL).
const sizes = [_]usize{ 16, 48, 4096, 8192, 32768, 65536, 262144, cap, cap + 16, 100 };
const seeds = [_]u64{ 1, 2, 3 };
const graph_gathers = 4;
const graph_replays = 64;
const reps = 2000;

/// Rank `rank`'s shard for a case: bytes that differ by rank, size and seed.
fn shard(out: []u8, seed: u64, rank: usize) void {
    var rng = std.Random.DefaultPrng.init(seed *% 1000003 +% rank *% 7919 +% out.len);
    rng.random().bytes(out);
}

/// Every rank's shard of `n` bytes for `seed`, in rank order: what a gather must leave on every rank.
fn expected(want: []u8, n: usize, seed: u64, world: usize) void {
    for (0..world) |r| shard(want[r * n ..][0..n], seed, r);
}

/// The TF_GLM_OVN_ROCE_* knobs (rails, stripe threshold, loads first, proxy cores) run this unchanged: same bytes.
pub fn run(gpu: Gpu, rank: usize, world: usize) !void {
    const a = gpu.gpa;
    try check.expect(std.c.getenv("NCCL_COMM_ID") != null, "roce-gather: NCCL_COMM_ID names rank 0's address", .{});
    try check.expect(world >= 2 and world <= cuda.roce.max_world_roce and rank < world, "roce-gather: rank {d} of world {d}", .{ rank, world });
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.tp_group.Group.open(gpu.d, rank, world);
    defer g.close();
    try g.openRoce(a, gpu.io, gpu.d, s, cap);
    try check.expect(g.fast != null, "roce-gather: the RoCE path did not come up on every rank (see the warning above)", .{});

    const most = sizes[std.mem.indexOfMax(usize, &sizes)];
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, most);
    defer mine.free();
    var by_group = try cuda.DeviceBuffer.alloc(gpu.d, world * most);
    defer by_group.free();
    var by_nccl = try cuda.DeviceBuffer.alloc(gpu.d, world * most);
    defer by_nccl.free();
    const got = try a.alloc(u8, world * most);
    defer a.free(got);
    const want = try a.alloc(u8, world * most);
    defer a.free(want);

    var by_roce: usize = 0;
    for (sizes) |n| for (seeds) |seed| {
        expected(want[0 .. world * n], n, seed, world);
        try mine.upload(0, want[rank * n ..][0..n]);
        try g.allGather(s, mine.ptr, by_group.ptr, n);
        try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, by_nccl.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
        try s.synchronize();
        try g.check();
        try by_group.download(0, got[0 .. world * n]);
        try check.sameBytes("group all-gather against every rank's shard", got[0 .. world * n], want[0 .. world * n]);
        try by_nccl.download(0, got[0 .. world * n]);
        try check.sameBytes("NCCL all-gather against every rank's shard", got[0 .. world * n], want[0 .. world * n]);
        if (g.fast.?.fits(n)) by_roce += 1;
    };
    check.pass("roce-gather rank {d} of {d}: shards of {any} bytes (seeds 1-3) equal NCCL's all-gather; {d} of {d} cases by RoCE", .{ rank, world, sizes, by_roce, sizes.len * seeds.len });

    try replays(gpu, &g, s, rank, world);
    try timing(gpu, &g, s, rank, world);
}

/// A graph of `graph_gathers` RoCE gathers replayed with new shards each time: each replay's rows are its own.
fn replays(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize, world: usize) !void {
    const a = gpu.gpa;
    const n = 8192;
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, graph_gathers * n);
    defer mine.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, graph_gathers * world * n);
    defer all.free();
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..graph_gathers) |i| try g.allGather(s, mine.ptr + i * n, all.ptr + i * world * n, n);
    var captured = try cuda.graph.endCapture(s);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    const host = try a.alloc(u8, graph_gathers * n);
    defer a.free(host);
    const got = try a.alloc(u8, graph_gathers * world * n);
    defer a.free(got);
    const want = try a.alloc(u8, graph_gathers * world * n);
    defer a.free(want);
    for (0..graph_replays) |r| {
        for (0..graph_gathers) |i| {
            expected(want[i * world * n ..][0 .. world * n], n, 100 + r * graph_gathers + i, world);
            @memcpy(host[i * n ..][0..n], want[i * world * n + rank * n ..][0..n]);
        }
        try mine.upload(0, host);
        try exec.launchOn(s);
        try s.synchronize();
        try g.check();
        try all.download(0, got);
        try check.sameBytes("a replayed graph's gathers", got, want);
    }
    check.pass("roce-gather rank {d}: a captured graph of {d} gathers of {d} bytes replayed {d} times, each replay's rows its own", .{ rank, graph_gathers, n, graph_replays });
}

/// Median microseconds a call of `reps` back-to-back gathers, RoCE against NCCL, at decode-sized shards.
fn timing(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize, world: usize) !void {
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    const prompt_shard = 4096 * 2048 * 4 / 3; // a 4,096-row prompt chunk's fp32 partials, one rank's share at TP=3
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, prompt_shard);
    defer mine.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, world * prompt_shard);
    defer all.free();
    for ([_]usize{ 4096, 8192, 32768, 65536, 1 << 20, 1 << 22, prompt_shard }) |n| {
        const r: usize = if (n >= 1 << 20) reps / 50 else reps; // prompt-sized shards: fewer calls, the same median
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            for (0..2) |round| { // the first round warms
                try t0.record(s);
                for (0..r) |_| {
                    if (which == 0) try g.allGather(s, mine.ptr, all.ptr, n) else try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, all.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
                }
                try t1.record(s);
                try t1.synchronize();
                if (round == 1) us[which] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) * 1000.0 / @as(f64, @floatFromInt(r));
            }
        }
        try g.check();
        const gbs = @as(f64, @floatFromInt(n * (world - 1))) / (us[0] * 1e3); // bytes received a rank over the call
        check.pass("roce-gather rank {d}: {d} bytes a rank, {d} calls: RoCE {d:.1} us a call ({d:.2} GB/s in), NCCL {d:.1} us", .{ rank, n, r, us[0], gbs, us[1] });
    }
}
