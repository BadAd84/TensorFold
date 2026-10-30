//! TF_GLM_OVN_DISK_CARRY's replay on every rank: its own part of a stored state against its fresh image, on its disk.
const std = @import("std");
const lanes = @import("lanes");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const ranks = @import("cuda_disk_ranks.zig");
const local = @import("cuda_disk_local.zig");
const part = @import("cuda_disk_part.zig");
const Engine = @import("cuda_engine.zig").Engine;
const compare = lanes.carry_compare;
const file = lanes.rows_file;
const rimg = lanes.rows_image;
const Op = ranks.Op;
const Path = local.Path;
const File = part.File;

const kind = @backingInt(follow.Kind.disk);

/// Rank 0: every rank's carry_compare word (rank 0's 0) for its own part of `files` of fresh state `id` at `at`.
pub fn ask(e: *Engine, dir: []const u8, id: u64, at: u32, block: u32, files: []const File, out: []u64) ![]const u64 {
    const world = e.share.world;
    if (world > out.len) return error.TooManyRanks;
    const x = ranks.leader(e) orelse {
        out[0] = 0;
        return out[0..1];
    };
    // the step's ids: the directory's path, the block, then each file as DISK_LOCAL's restore names it
    const head = (dir.len + 3) / 4;
    const ids = try e.gpa.alloc(u32, head + 1 + files.len * local.file_words);
    defer e.gpa.free(ids);
    var p: Path = .{};
    @memcpy(ids[0..head], try p.of(dir));
    ids[head] = block;
    for (files, 0..) |f, i| local.putFile(f, ids[head + 1 + i * local.file_words ..][0..local.file_words]);
    if (ids.len > x.link.ids.len) return error.TooManyIds;
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.carry_replay), .a = id, .seed = at, .depth = @intCast(dir.len), .top_k = @intCast(files.len) }, ids);
    try gather(x.link, e, 0, out[0..world]); // rank 0 compared its own files before it asked (rows_replay)
    return out[0..world];
}

/// A rank other than 0: its part of the step's files against its own image of the step's state; its word gathered.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32) !void {
    var out: [64]u64 = undefined;
    if (e.share.world > out.len) return error.BadStep;
    const dir = try Path.read(h, ids);
    const head = (h.depth + 3) / 4;
    if (ids.len != head + 1 + @as(usize, h.top_k) * local.file_words) return error.BadStep;
    const files = try e.gpa.alloc(File, h.top_k);
    defer e.gpa.free(files);
    for (files, 0..) |*f, i| f.* = try local.getFile(ids[head + 1 + i * local.file_words ..][0..local.file_words]);
    const rank = e.share.rank;
    const c = mine(e, dir, h.a, @intCast(h.seed), ids[head], files) catch |err| blk: {
        std.log.warn("glm rank {d}: comparing its part of {d} replayed tokens failed ({s})", .{ rank, h.seed, @errorName(err) });
        break :blk compare.Compare.unreadable;
    };
    switch (c) {
        .same => {},
        .unreadable => std.log.info("glm rank {d}: its part of the {d}-token state in {s} cannot be read: no proof", .{ rank, h.seed, dir }),
        .differs => |off| std.log.warn("glm rank {d}: its part of the {d}-token state in {s} differs from image byte {d}", .{ rank, h.seed, dir, off }),
    }
    try gather(l, e, compare.word(c), out[0..e.share.world]);
}

/// This rank's part of `files` in `dir` against its own image of kept state `id` at `at`, as image.canon matches them.
fn mine(e: *Engine, dir: []const u8, id: u64, at: u32, block: u32, files: []const File) !compare.Compare {
    const parts = try part.layout(e, at, e.gpa);
    defer e.gpa.free(parts);
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    var w: [6]u64 = undefined;
    const sh = try ranks.shape(e);
    var g: Fresh = .{ .copy = try ranks.copier(e, id, image.imageBytes(sh, at), &segs, &w) };
    const rank: u32 = @intCast(e.share.rank);
    const stored = (try storedWords(e, dir, rank, at, block, parts, files)) orelse return .unreadable;
    var ms: std.ArrayList(rimg.Match) = .empty;
    defer ms.deinit(e.gpa);
    try image.canon(sh, at, stored, try ranks.wordsOf(e, id), &ms, e.gpa);
    compare.order(ms.items);
    for (files) |f| {
        var rs: std.ArrayList(rimg.Range) = .empty;
        defer rs.deinit(e.gpa);
        try part.ranges(parts, at, block, f, &rs, e.gpa);
        const n = rimg.total(rs.items);
        const stage = try file.stage(e.gpa, n);
        defer e.gpa.free(stage);
        const buf = try e.gpa.alloc(u8, @intCast(n));
        defer e.gpa.free(buf);
        var pb: [1200]u8 = undefined;
        const path = try part.pathIn(&pb, dir, rank, .{ .stem = f.name[0..16].*, .kind = f.kind }, false);
        const c = try compare.same(e.gpa, path, f.kind, f.name, block, if (f.kind == .state) at else 0, rs.items, ms.items, stage, buf, .{ .ctx = &g, .gather = Fresh.gather });
        if (c != .same) return c;
    }
    return .same;
}

/// The words this rank's own file of `files` leads with (part.ranges), or null when it cannot be read.
fn storedWords(e: *Engine, dir: []const u8, rank: u32, at: u32, block: u32, parts: []const rimg.Part, files: []const File) !?image.Words {
    for (files) |f| if (f.kind == .state) {
        var rs: std.ArrayList(rimg.Range) = .empty;
        defer rs.deinit(e.gpa);
        try part.ranges(parts, at, block, f, &rs, e.gpa);
        const stage = try file.stage(e.gpa, rimg.total(rs.items));
        defer e.gpa.free(stage);
        var pb: [1200]u8 = undefined;
        const path = try part.pathIn(&pb, dir, rank, .{ .stem = f.name[0..16].*, .kind = f.kind }, false);
        var x: [6]u64 = undefined;
        if (!try compare.storedBytes(path, f.kind, f.name, block, at, rs.items, 0, stage, std.mem.sliceAsBytes(&x))) return null;
        return image.Words.decode(x, at) catch null;
    };
    return null;
}

/// This rank's image of the fresh state, ranges of it copied out as a part's save copies them.
const Fresh = struct {
    copy: image.Copy,

    fn gather(ctx: *anyopaque, ranges: []const rimg.Range, out: []u8) anyerror!void {
        const g: *Fresh = @ptrCast(@alignCast(ctx));
        var o: usize = 0;
        for (ranges) |r| {
            const n: usize = @intCast(r.len);
            try g.copy.run(r.off, out[o..][0..n], true);
            o += n;
        }
    }
};

fn gather(l: *follow.Link, e: *const Engine, v: u64, out: []u64) !void {
    try @import("cuda_ranks.zig").gather(u64, l.d, l.lib, l.comm, l.stream, e.share.rank, v, out);
}
