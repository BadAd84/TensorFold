//! GLM-5.3-Flash's KV pages on CUDA: a paged stream against its twin on a whole window, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lanes_tests = @import("glm_lanes_tests.zig");
const Gpu = check.Gpu;

const chunk = @import("glm_cuda").engine.prompt_rows;
/// Prompts at a page's edges (512 rows), a prompt chunk's and past the dense pass (the sparse one reads pools), and
/// either side of 32,768 positions (64 page parts), where slab and pages served different replies at TP=3.
const lengths = [_]usize{ 511, 512, 513, chunk - 1, chunk, chunk + 1, 5000, 32767, 32768, 32769, 33000 };
/// A window of 33,280 (65 page parts) and a verify window, as cuda_native.capacity makes it.
const capacity = 33344;
const paged_slot = 0; // the engine's own sequence, on pages from its init
const shuffled = 1; // takes and gives back pages between prompts, so slot 0's come out of the free list's order
const twin_slot = 3; // past the paged sequences: a whole window, as without TF_KV_PAGES

/// `glm-paged-check`: pages taken out of order and NaN-filled; prompt, windows across page edges, keeps and drafts.
/// With `kv8` both sides hold FP8 latent rows: codes and scales on pages against a window's codes and scales.
pub fn pagedCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8, kv8: bool) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try lanes_tests.openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    set.pages = true; // every rank's engine makes its own sequence on pages, as TF_KV_PAGES does
    set.kv8 = kv8; // --kv-cache fp8: the twin's window (cuda_engine.bind) takes the same format
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} made rank 0's pool and mapped its pages through every step", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const P = glm.pages;
    const l = try P.layout(gpu.gpa, e, 0);
    const count = 2 * P.tableLen(capacity) + 3; // two whole windows and a few pages over
    try x.pool(2, count, l.pageBytes());
    e.pages = try P.Pages.init(gpu.gpa, e, l, 2, count, true);
    const pages = e.pages.?;
    pages.poison = true; // a row read before its write turns the reply to NaN
    const V = e.share.vocab.len();
    const mine = try a.alloc(u16, 4 * V);
    const twin = try a.alloc(u16, 4 * V);
    var failed = false;
    var unordered: usize = 0;
    var compared: usize = 0;
    for (lengths, 0..) |len, k| {
        if (k % 2 == 0) {
            try x.bind(shuffled);
            _ = try pages.open(shuffled, (k + 1) * 512 - 100, 0);
        } else x.release(shuffled) catch {};
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, i| u.* = @intCast(100 + (i * 7919 + k * 31) % 5000);
        const held = try pages.open(paged_slot, len, 64);
        for (held[1..], held[0 .. held.len - 1]) |b, prev| unordered += @intFromBool(b != prev + 1);
        var first: [2]u32 = undefined;
        var drafts: [2][glm.engine.max_drafts]u32 = undefined;
        for ([_]u32{ paged_slot, twin_slot }, 0..) |slot, j| {
            try x.bind(slot);
            if (slot == paged_slot) try x.beginOn(null, held) else try x.begin(null);
            first[j] = try x.prefill(ids, null, null, null);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bp.logits, V * 2), "cuMemcpyDtoH");
            try x.draft(true, &.{first[j]}, 2, &drafts[j]);
        }
        if (first[0] != first[1] or !std.mem.eql(u16, mine[0..V], twin[0..V]) or !std.mem.eql(u32, drafts[0][0..2], drafts[1][0..2])) {
            std.debug.print("FAIL paged: a {d}-token prompt's first token {d}, logits or drafts differ from its twin's ({d})\n", .{ len, first[0], first[1] });
            failed = true;
        }
        for (0..6) |round| { // windows of 1 to 4 rows from the prompt's end, across its page's edge
            const rows = 1 + round % 4;
            var w: [4]u32 = undefined;
            for (w[0..rows], 0..) |*u, r| u.* = @intCast(100 + (round * 31 + k * 7 + r * 13) % 5000);
            var outs: [2][4]u32 = undefined;
            for ([_]u32{ paged_slot, twin_slot }, 0..) |slot, j| {
                try x.bind(slot);
                try x.verify(w[0..rows], outs[j][0..rows], 0);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bd.logits, rows * V * 2), "cuMemcpyDtoH");
                const keep = 1 + (round + k) % rows;
                try x.keep(keep);
                try x.draft(false, outs[j][0..keep], 2, &drafts[j]);
            }
            if (!std.mem.eql(u16, mine[0 .. rows * V], twin[0 .. rows * V]) or !std.mem.eql(u32, outs[0][0..rows], outs[1][0..rows]) or !std.mem.eql(u32, drafts[0][0..2], drafts[1][0..2])) {
                std.debug.print("FAIL paged: a {d}-token prompt's window {d} ({d} rows) differs from its twin's\n", .{ len, round, rows });
                failed = true;
            }
            compared += rows;
        }
        try x.release(paged_slot);
        try x.release(twin_slot);
    }
    // prompts stepped through Leader.chunk with marks off every page and pool edge, as the prompt cache cuts them
    const Nop = struct {
        fn call(_: *anyopaque, _: u32) void {}
    };
    var nop: u8 = 0;
    const cuts = [_]u32{ 1001, 2049, 3337, 4097, 6151 };
    var stepped: usize = 0;
    for ([_]usize{ 5000, 9001 }, 0..) |len, k| {
        const marks: glm.engine.Marks = .{ .at = cuts[0..], .ctx = &nop, .call = Nop.call };
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, i| u.* = @intCast(100 + (i * 7919 + k * 97) % 5000);
        const held = try pages.open(paged_slot, len, 64);
        var first: [2]u32 = .{ 0, 0 };
        var drafts: [2][glm.engine.max_drafts]u32 = undefined;
        for ([_]u32{ paged_slot, twin_slot }, 0..) |slot, j| {
            try x.bind(slot);
            if (slot == paged_slot) try x.beginOn(null, held) else try x.begin(null);
            var start: usize = 0;
            while (start < len) {
                const step = try x.chunk(ids, start, false, null, null, marks);
                start += step.rows;
                if (step.first) |f| first[j] = f;
                stepped += @intFromBool(j == 0);
            }
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bp.logits, V * 2), "cuMemcpyDtoH");
            try x.draft(true, &.{first[j]}, 2, &drafts[j]);
        }
        if (first[0] != first[1] or !std.mem.eql(u16, mine[0..V], twin[0..V]) or !std.mem.eql(u32, drafts[0][0..2], drafts[1][0..2])) {
            std.debug.print("FAIL paged: a {d}-token prompt cut at marks: first token {d}, logits or drafts differ from its twin's ({d})\n", .{ len, first[0], first[1] });
            failed = true;
        }
        for (0..3) |round| {
            const rows = 1 + round % 4;
            var w: [4]u32 = undefined;
            for (w[0..rows], 0..) |*u, r| u.* = @intCast(100 + (round * 37 + k * 11 + r * 13) % 5000);
            var outs: [2][4]u32 = undefined;
            for ([_]u32{ paged_slot, twin_slot }, 0..) |slot, j| {
                try x.bind(slot);
                try x.verify(w[0..rows], outs[j][0..rows], 0);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bd.logits, rows * V * 2), "cuMemcpyDtoH");
                try x.keep(rows);
                try x.draft(false, outs[j][0..rows], 2, &drafts[j]);
            }
            if (!std.mem.eql(u16, mine[0 .. rows * V], twin[0 .. rows * V]) or !std.mem.eql(u32, outs[0][0..rows], outs[1][0..rows])) {
                std.debug.print("FAIL paged: a {d}-token prompt cut at marks: window {d} differs from its twin's\n", .{ len, round });
                failed = true;
            }
            compared += rows;
        }
        try x.release(paged_slot);
        try x.release(twin_slot);
    }
    std.debug.print("paged: {d} prompt steps cut at marks off the page and pool edges\n", .{stepped});
    x.release(shuffled) catch {};
    const c = pages.pool.census();
    if (!pages.pool.idle() or c.free != c.total) {
        std.debug.print("FAIL paged: the pool is not idle after every stream ended (free {d} of {d}, promised {d})\n", .{ c.free, c.total, c.promised });
        failed = true;
    }
    if (unordered == 0) {
        std.debug.print("FAIL paged: every stream's pages came in order; the check did not reach a shuffled table\n", .{});
        failed = true;
    }
    if (failed) return error.TestFailed;
    check.pass("pages at TP={d} ({s} KV): on the engine's own sequence, {d} prompts at the pages' edges (511 to 33,000 tokens) and {d} window rows equal their twins' logits, tokens and drafts byte for byte, pages out of order ({d} jumps) and NaN before use; the pool idle after", .{ world, if (kv8) "FP8" else "BF16", lengths.len, compared, unordered });
}
