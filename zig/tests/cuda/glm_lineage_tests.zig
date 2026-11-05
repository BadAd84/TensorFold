//! GLM-5.3-Flash's kept states under TF_GLM_OVN_LINEAGE against the knob-off path, on pages and on a window.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const max_keeps = 4;

/// One request's marks' keeps, in order.
const Keep = struct {
    e: *anyopaque,
    ids: [max_keeps]u64 = @splat(0),
    n: usize = 0,
    fn call(ctx: *anyopaque, _: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ctx));
        const e: *@import("glm_cuda").engine.Engine = @ptrCast(@alignCast(k.e));
        if (k.n == max_keeps) return;
        k.ids[k.n] = e.keepState() catch 0;
        k.n += 1;
    }
};

/// What one arm leaves to compare: every request's first token, next window's tokens and logits, every kept image.
const Arm = struct {
    a: std.mem.Allocator,
    out: std.ArrayList(Record) = .empty,
    fn add(r: *Arm, bytes: []const u8, at: u32) !void {
        try r.out.append(r.a, .{ .bytes = try r.a.dupe(u8, bytes), .at = at });
    }
};

/// A record's bytes; `at` > 0: rank 0's image of a state kept at `at`.
const Record = struct { bytes: []const u8, at: u32 = 0 };

/// The scenario's driver on rank 0: a request is a slot, its ids, the state it resumes and the marks it keeps at.
const Run = struct {
    x: @import("glm_cuda").follow.Leader,
    arm: *Arm,
    vocab: usize,
    paged: bool, // TF_KV_PAGES: slots 0 and 1 hold their rows on pages, slot 3 a window

    fn request(r: Run, slot: u32, ids: []const u32, from: ?@import("glm_cuda").engine.Resume, at: []const u32, release: bool) !Keep {
        const glm = @import("glm_cuda");
        const e = r.x.e;
        try r.x.bind(slot);
        if (r.paged and slot < 2) {
            e.pages.?.want = if (from) |f| f.id else 0; // TF_GLM_OVN_LINEAGE: the pages that hold its rows
            const held = try e.pages.?.open(slot, ids.len, 64);
            try r.x.beginOn(null, held, 0);
        } else try r.x.begin(null);
        var keep: Keep = .{ .e = e };
        var buf: [max_keeps]u32 = undefined; // a mark at the resume row: the restored state, kept before any pass
        var n: usize = 0;
        if (from) |f| {
            buf[0] = @intCast(f.at);
            n = 1;
        }
        @memcpy(buf[n..][0..at.len], at);
        const marks: glm.engine.Marks = .{ .at = buf[0 .. n + at.len], .ctx = &keep, .call = Keep.call };
        var start: usize = if (from) |f| f.at else 0;
        const first = while (true) {
            const c = try r.x.chunk(ids, start, false, if (start == (if (from) |f| f.at else 0)) from else null, null, marks);
            if (c.first) |f| break f;
            start += c.rows;
        };
        const win = [_]u32{ first, 101, 202, 303 };
        var outs: [4]u32 = undefined;
        try r.x.verify(&win, &outs, 0);
        const logits = try r.arm.a.alloc(u16, win.len * r.vocab);
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(logits.ptr, e.bd.logits, logits.len * 2), "cuMemcpyDtoH");
        try r.arm.add(std.mem.asBytes(&first), 0);
        try r.arm.add(std.mem.asBytes(&outs), 0);
        try r.arm.add(std.mem.sliceAsBytes(logits), 0);
        for (keep.ids[0..keep.n], 0..) |id, i| {
            if (id == 0) {
                std.debug.print("FAIL lineage: slot {d} kept nothing at mark {d}\n", .{ slot, buf[i] });
                return error.TestFailed;
            }
            try r.arm.add(try r.image(id), @intCast(e.snaps.get(id).?.pos));
        }
        if (keep.n != n + at.len) {
            std.debug.print("FAIL lineage: slot {d} kept {d} states at {d} marks\n", .{ slot, keep.n, n + at.len });
            return error.TestFailed;
        }
        if (n == 1) { // the restored state's image is taken: the request's own keeps stay
            e.dropState(keep.ids[0]);
            std.mem.copyForwards(u64, keep.ids[0..at.len], keep.ids[1..keep.n]);
            keep.n -= 1;
        }
        if (release) try r.x.release(slot);
        return keep;
    }

    /// Rank 0's disk image of kept state `id` (its words and parts, wherever its lease holds them).
    fn image(r: Run, id: u64) ![]u8 {
        const glm = @import("glm_cuda");
        const e = r.x.e;
        const s = e.snaps.get(id) orelse return error.NoKeptState;
        const out = try r.arm.a.alloc(u8, glm.disk_image.imageBytes(try glm.disk_ranks.shape(e), @intCast(s.pos)));
        try glm.disk_ranks.copy(e, id, 0, out, true);
        return out;
    }

    /// State `id` evicted and read back as from disk: each rank's image out, the state dropped, a blank filled, sealed.
    fn reload(r: Run, id: u64) !u64 {
        const glm = @import("glm_cuda");
        const disk = glm.disk_image;
        const ranks = glm.disk_ranks;
        const e = r.x.e;
        const at: u32 = @intCast(e.snaps.get(id).?.pos);
        const sh = try ranks.shape(e);
        const pre = try r.arm.a.alloc(u64, @intCast(e.share.world));
        try ranks.prefixes(e, pre);
        const imgs = try r.arm.a.alloc([]u8, pre.len);
        for (imgs, pre, 0..) |*img, p, rank| {
            var own = sh;
            own.prefix = p;
            img.* = try r.arm.a.alloc(u8, disk.imageBytes(own, at));
            try ranks.take(e, @intCast(rank), id, 0, img.*);
        }
        const words = try disk.Words.decode(std.mem.bytesToValue([6]u64, imgs[0][0..disk.Words.bytes]), at);
        e.dropState(id);
        const back = try ranks.blank(e, at);
        try ranks.seal(e, back, words); // the reader's order (cuda_glm_disk.zig scatter): sealed from the words, then filled
        for (imgs, 0..) |img, rank| try ranks.put(e, @intCast(rank), back, 0, img);
        return back;
    }
};

/// Every sequence's and page's lineage forgotten (rank 0), as at load: the knob-off arm wrote rows it did not tag.
fn forget(e: anytype) void {
    for (e.seqs.items) |q| if (q) |s| {
        s.st.lin = .{};
    };
    if (e.pages) |p| @memset(p.tags, .{});
}

/// `glm-lineage-check` (TF_GLM_OVN_LINEAGE=1): one scenario of turns on pages, forks, a window's resumes and a state
/// evicted and read back, run on rank 0 without the knob and with it; every first token and next window's tokens and
/// logits must be equal byte for byte, every kept image (the restored state at each resume row among them) in every
/// byte the state holds (cuda_disk_image.heldDiff), and the knob's arm must leave rows in place and share pages.
pub fn lineageCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    set.pages = true; // the engine's own sequence and slot 1 on pages, as TF_KV_PAGES serves
    const capacity = 33344;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    if (!glm.lineage.on) {
        std.debug.print("FAIL lineage: TF_GLM_OVN_LINEAGE is not set on rank {d}\n", .{rank});
        return error.TestFailed;
    }
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link); // with the knob in both arms: each rank's copies are its own
        check.pass("rank {d} followed rank 0's turns, forks and reloads in both arms", .{rank});
        return;
    }
    e.link = &link; // rank 0's disk steps (cuda_disk_ranks) cross on the link, as cuda_native.zig sets it for the serve
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const P = glm.pages;
    const l = try P.layout(gpu.gpa, e, 0);
    const count = 2 * P.tableLen(capacity) + 3;
    try x.pool(2, count, l.pageBytes());
    e.pages = try P.Pages.init(gpu.gpa, e, l, 2, count, true);
    try x.snapshots(8 << 30);
    const conv = try a.alloc(u32, 16400);
    for (conv, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
    const fork1 = try a.dupe(u32, conv[0..15000]);
    const fork2 = try a.dupe(u32, conv[0..14000]);
    for (fork1[13003..], 13003..) |*u, j| u.* = @intCast(100 + (j * 6007 + 17) % 5000);
    for (fork2[13003..], 13003..) |*u, j| u.* = @intCast(100 + (j * 4049 + 29) % 5000);
    var arms: [2]Arm = .{ .{ .a = a }, .{ .a = a } };
    var stats: glm.lineage_copy.Stats = .{};
    for (&arms, 0..) |*arm, k| {
        glm.lineage.on = k == 1;
        forget(e);
        glm.lineage_copy.stats = .{};
        const r: Run = .{ .x = x, .arm = arm, .vocab = e.share.vocab.len(), .paged = true };
        const R = glm.engine.Resume;
        const r1 = try r.request(1, conv[0..12100], null, &.{12001}, true); // a turn kept off a pool's start
        const k1 = r1.ids[0];
        const r2 = try r.request(1, conv[0..14500], R{ .id = k1, .at = 12001 }, &.{ 13003, 14400 }, true); // the next
        const r3 = try r.request(1, conv[0..16000], R{ .id = r2.ids[1], .at = 14400 }, &.{15997}, true);
        const f1 = try r.request(0, fork1, R{ .id = r2.ids[0], .at = 13003 }, &.{14501}, true); // a fork, other slot
        _ = try r.request(1, fork2, R{ .id = r2.ids[0], .at = 13003 }, &.{}, true); // the same state forked again
        _ = try r.request(3, conv[0..16200], R{ .id = r3.ids[0], .at = 15997 }, &.{}, false); // a window, twice
        _ = try r.request(3, conv[0..16100], R{ .id = r3.ids[0], .at = 15997 }, &.{}, true);
        const back = try r.reload(r2.ids[1]); // evicted, read back, resumed twice on one slot
        _ = try r.request(1, conv[0..16000], R{ .id = back, .at = 14400 }, &.{}, true);
        _ = try r.request(1, conv[0..16300], R{ .id = back, .at = 14400 }, &.{}, true);
        for ([_]u64{ k1, r2.ids[0], r3.ids[0], f1.ids[0], back }) |id| e.dropState(id);
        if (k == 1) stats = glm.lineage_copy.stats;
    }
    glm.lineage.on = true;
    var failed = false;
    if (arms[0].out.items.len != arms[1].out.items.len) {
        std.debug.print("FAIL lineage: {d} records without the knob, {d} with it\n", .{ arms[0].out.items.len, arms[1].out.items.len });
        failed = true;
    }
    const sh = try glm.disk_ranks.shape(e);
    for (arms[0].out.items, arms[1].out.items, 0..) |p, q, i| {
        const raw = std.mem.indexOfDiff(u8, p.bytes, q.bytes) orelse continue;
        // an image's window scratch carries the last decode window on its slot, which the arms ran in other orders
        const held = if (p.at == 0 or p.at != q.at) raw else try glm.disk_image.heldDiff(sh, p.at, p.bytes, q.bytes, a);
        if (held) |at| {
            std.debug.print("FAIL lineage: record {d} differs at byte {d} of {d}\n", .{ i, at, p.bytes.len });
            failed = true;
        } else std.debug.print("lineage: record {d} differs only past what a state holds, from byte {d}\n", .{ i, raw });
    }
    if (stats.skipped == 0 or stats.shared == 0) {
        std.debug.print("FAIL lineage: the knob's restores left {d} bytes in place and its keeps shared {d}\n", .{ stats.skipped, stats.shared });
        failed = true;
    }
    if (failed) return error.TestFailed;
    check.pass("lineage at TP={d}: {d} records (turns on pages, forks, a window's resumes, a reloaded state) equal the knob-off path in every byte a state holds; the knob's keeps and restores copied {d} MiB, left {d} MiB in place and shared {d} MiB", .{ world, arms[0].out.items.len, stats.copied >> 20, stats.skipped >> 20, stats.shared >> 20 });
}
