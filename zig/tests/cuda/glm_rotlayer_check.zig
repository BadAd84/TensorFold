//! GLM-5.3-Flash under TF_GLM_OVN_EXPERT_ROTATE against the standard expert split, byte for byte: every layer's
//! residual rows after its FFN glue (a paired prompt step, a pieced chunk and a decode window) and the logits.
//! Mutant: glm_glue_rows.cuh ranks8 summing in rank order (`first` read as 0) fails it on every rotated layer.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// One engine pass: whether its owners rotate, the prompt's first token, the next window's tokens and logits.
const Pass = struct { on: bool, first: u32 = 0, outs: [4]u32 = @splat(0), logits: []u16 = &.{} };

/// `glm-rotlayer-check <ckpt> <rank> <world> <idfile>`: two engines on every rank, the knob read as unset, then set.
pub fn rotLayerCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    if (cuda.ovn.text("EXPERT_ROTATE") == null) {
        std.debug.print("FAIL rotlayer: TF_GLM_OVN_EXPERT_ROTATE is not set on rank {d}\n", .{rank});
        return error.TestFailed;
    }
    const row = 4 * 4096 * 2; // a residual row: 4 streams of 4,096 bf16
    const host = try a.alloc(u8, glm.engine.prompt_rows * row); // the most rows one glue call posts
    var caps = [2]glm.rotlayer.Capture{ .{ .d = gpu.d, .host = host }, .{ .d = gpu.d, .host = host } };
    var runs: [2]Pass = undefined;
    defer glm.rotlayer.skip = false;
    for (&runs, &caps, 0..) |*p, *cap, k| {
        glm.rotlayer.skip = k == 0; // A: the standard split whatever the knob says
        p.* = try pass(gpu, a, checkpoint, &lib, comm, rank, world, set, cap);
    }
    var failed = false;
    if (runs[0].on or !runs[1].on) {
        std.debug.print("FAIL rotlayer: rank {d}: owners rotate {} off the knob and {} on it (TP={d} splits the experts evenly?)\n", .{ rank, runs[0].on, runs[1].on, world });
        return error.TestFailed;
    }
    var layers: usize = 0;
    var rows: usize = 0;
    for (caps[0].sums, caps[1].sums, caps[0].rows, caps[1].rows, 0..) |sa, sb, ra, rb, j| {
        if (ra == 0 and rb == 0) continue;
        layers += 1;
        rows += ra;
        if (sa != sb or ra != rb) {
            std.debug.print("FAIL rotlayer: rank {d}: layer {d}'s glued rows differ ({d} and {d} rows)\n", .{ rank, j, ra, rb });
            failed = true;
        }
    }
    if (layers == 0) {
        std.debug.print("FAIL rotlayer: rank {d}: no layer's rows were captured\n", .{rank});
        return error.TestFailed;
    }
    if (rank == 0) {
        const A = runs[0];
        const B = runs[1];
        if (A.first != B.first or !std.mem.eql(u32, &A.outs, &B.outs) or !std.mem.eql(u16, A.logits, B.logits)) {
            std.debug.print("FAIL rotlayer: first token {d} rotated against {d}; first logit difference at {?d}\n", .{ B.first, A.first, std.mem.indexOfDiff(u16, A.logits, B.logits) });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("TF_GLM_OVN_EXPERT_ROTATE at TP={d}: rank {d}'s glued rows of {d} layers ({d} in all, over a paired step, a pieced chunk and a decode window){s} equal the standard split's byte for byte", .{ world, rank, layers, rows, if (rank == 0) " and the logits" else "" });
}

/// One engine on this rank, its forwards' glued rows into `cap`: rank 0 leads a 9,192-row prompt and one window.
fn pass(gpu: Gpu, a: std.mem.Allocator, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: c_int, world: c_int, set: anytype, cap: anytype) !Pass {
    const glm = @import("glm_cuda");
    const capacity = 16384;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = false, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, lib, comm, e.stream, capacity);
    defer link.deinit();
    var p: Pass = .{ .on = glm.rotlayer.active };
    glm.rotlayer.capture = cap;
    defer glm.rotlayer.capture = null;
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        return p;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    try x.bind(0);
    try x.begin(null);
    const C = glm.engine.prompt_rows;
    const ids = try a.alloc(u32, 2 * C + 1000); // a paired step of two chunks, then a pieced chunk of 1,000 rows
    for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
    var at: usize = 0;
    while (true) {
        const r = try x.chunk(ids, at, true, null, null, null);
        if (r.first) |f| {
            p.first = f;
            break;
        }
        at += r.rows;
    }
    const win = [_]u32{ p.first, 101, 202, 303 };
    try x.verify(&win, &p.outs, 0);
    const V = e.share.vocab.len();
    p.logits = try a.alloc(u16, win.len * V);
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(p.logits.ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
    return p;
}
