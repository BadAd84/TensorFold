//! TF_GLM_OVN_SNAP_GATHER: a keep and a restore by tf_glm_snap_copy batches against the memcpy a piece, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const mib = 1 << 20;
const live_bytes = 12 * mib;
const span_page = 512 << 10; // a fake KV page's stride in the live buffer's paged region
const paged_at = 6 * mib; // the paged region's start in the live buffer
const ids = [_]u32{ 5, 1, 3, 0, 2 }; // its pages out of order

/// Parts as a paged sequence's plan has them: unaligned and aligned arena runs, page parts of 256 KiB and 2 KiB spans.
fn planAt(comptime S: type, base: u64) S.layout.Plan {
    var p: S.layout.Plan = .{ .ids = &ids };
    const parts = [_]S.layout.Part{
        .{ .live = base + 3, .len = 3 * mib + 5 },
        .{ .live = base + paged_at, .len = 4 * (256 << 10) + 1000, .span = 256 << 10, .page = span_page },
        .{ .live = base + paged_at + (300 << 10), .len = 5 * 2048 - 16, .span = 2048, .page = span_page },
        .{ .live = base + 4 * mib, .len = mib },
    };
    for (parts, 0..) |x, i| {
        p.parts[i] = x;
        p.len += x.len;
    }
    p.count = parts.len;
    return p;
}

fn leaseBytes(gpu: check.Gpu, comptime S: type, pool: *const S.pages.Pool, lease: *const S.pages.Lease, out: []u8) !void {
    const arena = pool.arena.?;
    for (lease.ids.items, 0..) |id, i| {
        const n = @min(S.pages.page, out.len - i * S.pages.page);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = arena.ptr + @as(u64, id) * S.pages.page, .len = n, .borrowed = true }).download(0, out[i * S.pages.page ..][0..n]);
    }
}

/// `glm-snap-gather`: keep (live to lease) and restore (lease to live) on both paths; the bytes must not differ.
pub fn run(gpu: check.Gpu) !void {
    const glm = @import("glm_cuda");
    const S = glm.snapshots;
    const gpa = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    const ks = try glm.glue.Kernels.load(glue_mod);
    var live = try cuda.DeviceBuffer.alloc(gpu.d, live_bytes);
    defer live.free();
    var back_memcpy = try cuda.DeviceBuffer.alloc(gpu.d, live_bytes);
    defer back_memcpy.free();
    var back_gather = try cuda.DeviceBuffer.alloc(gpu.d, live_bytes);
    defer back_gather.free();
    const host = try gpa.alloc(u8, live_bytes);
    defer gpa.free(host);
    var rng = std.Random.DefaultPrng.init(0x736e6170);
    rng.random().bytes(host);
    try live.upload(0, host);
    @memset(host, 0);
    try back_memcpy.upload(0, host);
    try back_gather.upload(0, host);
    var pool = try S.pages.Pool.init(gpa, gpu.d, 8 * S.pages.page);
    const g = try S.gather.Gather.init(gpa, gpu.d, ks.snap_copy);
    pool.gather = g;
    defer pool.deinit();
    const keep = planAt(S, live.ptr);
    var by_memcpy = try pool.take(keep.len);
    defer pool.give(&by_memcpy);
    var by_gather = try pool.take(keep.len);
    defer pool.give(&by_gather);
    pool.gather = null;
    try pool.move(s, &by_memcpy, &keep, true);
    pool.gather = g;
    try pool.move(s, &by_gather, &keep, true);
    try s.synchronize();
    const want = try gpa.alloc(u8, keep.len);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, keep.len);
    defer gpa.free(got);
    try leaseBytes(gpu, S, &pool, &by_memcpy, want);
    try leaseBytes(gpu, S, &pool, &by_gather, got);
    try check.sameBytes("keep: lease by tf_glm_snap_copy against cuMemcpyDtoDAsync", got, want);
    pool.gather = null;
    try pool.move(s, &by_memcpy, &planAt(S, back_memcpy.ptr), false);
    pool.gather = g;
    try pool.move(s, &by_gather, &planAt(S, back_gather.ptr), false);
    try s.synchronize();
    const a = try gpa.alloc(u8, live_bytes);
    defer gpa.free(a);
    try back_memcpy.download(0, host);
    try back_gather.download(0, a);
    try check.sameBytes("restore: live bytes by tf_glm_snap_copy against cuMemcpyDtoDAsync", a, host);
    try live.download(0, a);
    try check.sameBytes("restore: the unaligned run is the kept one", host[3..][0 .. 3 * mib + 5], a[3..][0 .. 3 * mib + 5]);
    check.pass("snapshot gather: {d} kept bytes in {d} parts, keep and restore equal to the memcpy path", .{ keep.len, keep.count });
}
