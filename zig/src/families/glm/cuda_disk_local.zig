//! TF_GLM_OVN_DISK_LOCAL's disk steps: rank 0 names each file of a kept state and every other rank acts on its own
//! part (cuda_disk_part) on its own disk, so no state byte crosses the fabric; a restore ends in every rank's yes or no
const std = @import("std");
const follow = @import("cuda_follow.zig");
const Kind = @import("lanes").rows_file.Kind;
const ranks = @import("cuda_disk_ranks.zig");
const part = @import("cuda_disk_part.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Op = ranks.Op;
pub const Key = part.Key;
pub const File = part.File;
pub const pinOf = part.pinOf;

const kind = @backingInt(follow.Kind.disk);

/// This process's own part: a rank other than 0, from rank 0's open step on.
var mine: ?*part.Part = null;

/// A file in a step: its kind, group and index, then its name's eight words (TF_GLM_OVN_DISK_CARRY's replay too).
pub const file_words = 3 + 8;
/// A key in a step: its kind, then its stem's four words.
const key_words = 1 + 4;

pub fn putFile(f: File, out: []u32) void {
    out[0..3].* = .{ @backingInt(f.kind), f.window, f.k };
    @memcpy(std.mem.sliceAsBytes(out[3..file_words]), &f.name);
}

pub fn getFile(in: []const u32) !File {
    if (in[0] != @backingInt(Kind.block) and in[0] != @backingInt(Kind.state)) return error.BadStep;
    var f: File = .{ .name = undefined, .kind = @fromBackingInt(in[0]), .window = in[1], .k = in[2] };
    @memcpy(&f.name, std.mem.sliceAsBytes(in[3..file_words]));
    return f;
}

fn putKey(k: Key, out: []u32) void {
    out[0] = @backingInt(k.kind);
    @memcpy(std.mem.sliceAsBytes(out[1..key_words]), &k.stem);
}

fn getKey(in: []const u32) !Key {
    if (in[0] != @backingInt(Kind.block) and in[0] != @backingInt(Kind.state)) return error.BadStep;
    var k: Key = .{ .stem = undefined, .kind = @fromBackingInt(in[0]) };
    @memcpy(&k.stem, std.mem.sliceAsBytes(in[1..key_words]));
    return k;
}

fn gather(l: *follow.Link, e: *const Engine, v: u64, out: []u64) !void {
    try @import("cuda_ranks.zig").gather(u64, l.d, l.lib, l.comm, l.stream, e.share.rank, v, out);
}

/// A directory's path as a disk step's ids, `depth` bytes of them (DISK_LOCAL's open, TF_GLM_OVN_DISK_CARRY's steps).
pub const Path = struct {
    ids: [300]u32 = @splat(0),

    /// `dir`'s bytes in `p`'s words, as many as they take.
    pub fn of(p: *Path, dir: []const u8) ![]const u32 {
        if (dir.len > p.ids.len * 4) return error.NameTooLong;
        @memcpy(std.mem.sliceAsBytes(&p.ids)[0..dir.len], dir);
        return p.ids[0 .. (dir.len + 3) / 4];
    }

    /// The path a step carries: `h.depth` bytes of its ids.
    pub fn read(h: follow.Header, ids: []const u32) ![]const u8 {
        if (h.depth > ids.len * 4) return error.BadStep;
        return std.mem.sliceAsBytes(ids)[0..h.depth];
    }
};

/// Rank 0: every other rank opens its own part of the store in `dir` (the same path on its own disk).
pub fn open(e: *Engine, dir: []const u8, floor: u64, block: u32, threads: u32) !void {
    const x = ranks.leader(e) orelse return;
    var p: Path = .{};
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_open), .a = floor, .seed = block, .depth = @intCast(dir.len), .top_k = threads }, try p.of(dir));
}

/// Rank 0 staged file `f` of kept state `id` at `at`: every other rank stages its own part of it.
pub fn file(e: *Engine, id: u64, at: u32, f: File) !void {
    const x = ranks.leader(e) orelse return;
    var ids: [file_words]u32 = undefined;
    putFile(f, &ids);
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_file), .a = id, .seed = at }, &ids);
}

/// Rank 0 forgot the file keyed `k`: every other rank removes its part (after any write of it queued before).
pub fn unlink(e: *Engine, k: Key) !void {
    const x = ranks.leader(e) orelse return;
    var ids: [key_words]u32 = undefined;
    putKey(k, &ids);
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_unlink) }, &ids);
}

/// Rank 0 removed identity `id`'s directory for room: every other rank removes its own.
pub fn other(e: *Engine, id: u64) !void {
    const x = ranks.leader(e) orelse return;
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_other), .a = id }, &.{});
}

/// Rank 0 restores `files` (blocks, then the own file) into blank `id` at `at`: every other rank reads its parts now.
pub fn read(e: *Engine, id: u64, at: u32, files: []const File) !void {
    const x = ranks.leader(e) orelse return;
    if (files.len * file_words > x.link.ids.len) return error.TooManyIds;
    const ids = try e.gpa.alloc(u32, files.len * file_words);
    defer e.gpa.free(ids);
    for (files, 0..) |f, i| putFile(f, ids[i * file_words ..][0..file_words]);
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_read), .a = id, .seed = at }, ids);
}

/// Rank 0's outcome `ok` and every other rank's: true only when every rank's part arrived.
pub fn status(e: *Engine, ok: bool) !bool {
    const x = ranks.leader(e) orelse return ok;
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_status) }, &.{});
    var out: [64]u64 = undefined;
    if (e.share.world > out.len) return error.TooManyRanks;
    try gather(x.link, e, @intFromBool(ok), out[0..e.share.world]);
    for (out[0..e.share.world]) |v| if (v != 1) return false;
    return true;
}

/// TF_IMPRINT_VERIFY=1: whether every other rank's part of kept state `back`, just read, equals its part of `saved`.
pub fn verify(e: *Engine, saved: u64, back: u64, at: u32) !bool {
    const x = ranks.leader(e) orelse return true;
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_verify), .a = saved, .seed = back, .top_k = at }, &.{});
    var out: [64]u64 = undefined;
    if (e.share.world > out.len) return error.TooManyRanks;
    try gather(x.link, e, 1, out[0..e.share.world]);
    for (out[0..e.share.world]) |v| if (v != 1) return false;
    return true;
}

/// Startup: for each key, whether some other rank lacks a checked file of it (each rank asked in turn, in batches).
pub fn lacks(e: *Engine, keys: []const Key, out: []bool) !void {
    const x = ranks.leader(e) orelse return;
    const per = x.link.ids.len / key_words; // its answer, a byte a key, fits the link's buffer as well
    const ids = try e.gpa.alloc(u32, per * key_words);
    defer e.gpa.free(ids);
    const answers = try e.gpa.alloc(u8, per);
    defer e.gpa.free(answers);
    var done: usize = 0;
    while (done < keys.len) {
        const n = @min(per, keys.len - done);
        for (keys[done..][0..n], 0..) |k, i| putKey(k, ids[i * key_words ..][0..key_words]);
        for (1..e.share.world) |r| {
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_have), .top_k = @intCast(r) }, ids[0 .. n * key_words]);
            try x.link.from(@intCast(r), n);
            try x.link.buf.download(64, answers[0..n]);
            for (answers[0..n], out[done..][0..n]) |b, *o| o.* = o.* or b != 0;
        }
        done += n;
    }
}

/// Startup, after the questions: every other rank removes the files no question named.
pub fn sweep(e: *Engine) !void {
    const x = ranks.leader(e) orelse return;
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.local_sweep) }, &.{});
}

/// A rank other than 0: its part's queued writes finished before the process leaves.
pub fn close() void {
    if (mine) |p| p.close();
    mine = null;
}

/// A rank other than 0: rank 0's DISK_LOCAL step. A failure on this rank's own disk never stops it: a read records
/// it for the status, anything else is logged (a missing part only costs that state a cold prefill on every rank).
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32, op: Op) !void {
    const rank: u32 = @intCast(e.share.rank);
    switch (op) {
        .local_open => {
            const dir = try Path.read(h, ids);
            close();
            mine = part.Part.open(e.gpa, dir, rank, h.a, @intCast(h.seed), @max(h.top_k, 1)) catch |err| blk: {
                std.log.warn("glm rank {d}: its part of the disk store did not open ({s}); its restores fail", .{ rank, @errorName(err) });
                break :blk null;
            };
        },
        .local_file => {
            if (ids.len != file_words) return error.BadStep;
            const f = try getFile(ids);
            const p = mine orelse return;
            p.save(e, h.a, @intCast(h.seed), f) catch |err| std.log.warn("glm rank {d}: staging its part of a kept state failed ({s})", .{ rank, @errorName(err) });
        },
        .local_unlink => {
            if (ids.len != key_words) return error.BadStep;
            const k = try getKey(ids);
            if (mine) |p| p.unlink(k) catch {};
        },
        .local_other => if (mine) |p| p.other(h.a) catch {},
        .local_have => {
            if (ids.len % key_words != 0) return error.BadStep;
            const n = ids.len / key_words;
            if (n > l.ids.len * 4) return error.BadStep;
            const answers = try e.gpa.alloc(u8, n);
            defer e.gpa.free(answers);
            for (answers, 0..) |*b, i| {
                const k = try getKey(ids[i * key_words ..][0..key_words]);
                const p = mine orelse {
                    b.* = 1;
                    continue;
                };
                p.seen.put(p.gpa, k, {}) catch {};
                b.* = if (h.top_k == rank and p.has(k)) 0 else 1;
            }
            if (h.top_k == rank) try l.buf.uploadAsync(64, answers, l.stream.handle);
            try l.from(h.top_k, n);
        },
        .local_sweep => if (mine) |p| p.sweep(),
        .local_read => {
            if (ids.len == 0 or ids.len % file_words != 0) return error.BadStep;
            const files = try e.gpa.alloc(File, ids.len / file_words);
            defer e.gpa.free(files);
            for (files, 0..) |*f, i| f.* = try getFile(ids[i * file_words ..][0..file_words]);
            if (mine) |p| p.read(e, h.a, @intCast(h.seed), files);
        },
        .local_status => {
            const ok = if (mine) |p| p.ok else false;
            if (mine) |p| p.ok = false; // the next status speaks for the next read only
            var out: [64]u64 = undefined;
            if (e.share.world > out.len) return error.BadStep;
            try gather(l, e, @intFromBool(ok), out[0..e.share.world]);
        },
        .local_verify => {
            const same = part.same(e, h.a, h.seed, h.top_k) catch |err| blk: {
                std.log.warn("glm rank {d}: comparing its part of a state read back failed ({s})", .{ rank, @errorName(err) });
                break :blk false;
            };
            if (!same) std.log.err("glm rank {d}: its part of {d} tokens read back differs from the kept state", .{ rank, h.top_k });
            var out: [64]u64 = undefined;
            if (e.share.world > out.len) return error.BadStep;
            try gather(l, e, @intFromBool(same), out[0..e.share.world]);
        },
        else => return error.BadStep,
    }
}
