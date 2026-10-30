//! TF_GLM_OVN_DISK_LOCAL, rank 0: what its store does to a state's files every other rank does to its own part on its
//! own disk (glm.disk_local's steps); the read slots pinned, and a save of any size staged by the pump (U6).
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const disk = @import("cuda_glm_disk.zig");
const rows = api.prompt_cache.rows;
const dl = glm.disk_local;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Saved = api.prompt_cache.Saved;

/// Production's PUMP_BLOCKS (recipe 0128): pieces staged a round when TF_GLM_OVN_DISK_PUMP leaves it unset, so a
/// state larger than the staging permit stages a few blocks a round instead of being skipped (ParityUpstream U6).
pub const pump_default: u32 = 4;

var mirror: rows.local.Mirror = undefined;
/// The ranks a restore woke (`begin`), back to sleep after its status (`end`): no step may run between the two, or
/// rank 0's own copies would queue behind a broadcast the reading ranks cannot join yet.
var reading: ?disk.Rest = null;

/// After the store opened: every other rank opens its own part, then the ranks agree on which states every rank holds.
pub fn open(e: *Engine, r: *rows.Rows, floor: u64) !void {
    if (r.sync_writes) return error.DiskLocalSyncWrites; // TENSORFOLD_ROWS_SYNC writes on rank 0's thread alone
    mirror = .{ .ctx = e, .file = file, .unlink = unlink, .other = other, .begin = begin, .end = end, .lacks = lacks, .sweep = sweep, .verify = verify, .pin = dl.pinOf(e) };
    r.readers = try lanes.disk_readers.knob(lanes.disk_readers.local_default);
    if (r.pump_blocks == 0) { // TF_GLM_OVN_DISK_PUMP unset: the pump for saves wider than the permit alone
        r.pump_blocks = pump_default;
        r.pump_auto = true;
    }
    {
        const rest = disk.Rest.of(e);
        defer rest.done();
        try dl.open(e, r.dir, floor, rows.block_tokens, r.readers);
    }
    disk.local = true;
    r.mirror = &mirror;
    const gone = try rows.local.agree(r);
    std.log.info("glm: TF_GLM_OVN_DISK_LOCAL: {d} readers a rank, {d} pieces staged a round; {d} states some rank lacked a part of are forgotten", .{ r.readers, r.pump_blocks, gone });
}

fn engineOf(ctx: *anyopaque) *Engine {
    return @ptrCast(@alignCast(ctx));
}

fn idOf(saved: Saved) u64 {
    const k: *Kept = @ptrCast(@alignCast(saved));
    return k.id;
}

fn file(ctx: *anyopaque, saved: Saved, at: u32, files: []const rows.local.File) anyerror!void {
    const e = engineOf(ctx);
    const rest = disk.Rest.of(e); // the other ranks woken once a batch, put back to sleep after its last file
    defer rest.done();
    for (files) |f| try dl.file(e, idOf(saved), at, .{ .name = f.name, .kind = f.kind, .window = f.window, .k = f.k });
}

fn unlink(ctx: *anyopaque, k: rows.local.Key) void {
    const e = engineOf(ctx);
    const rest = disk.Rest.of(e);
    defer rest.done();
    dl.unlink(e, .{ .stem = k.stem, .kind = k.kind }) catch |err| std.log.warn("glm: the other ranks were not told to remove a file ({s})", .{@errorName(err)});
}

fn other(ctx: *anyopaque, id: u64) void {
    const e = engineOf(ctx);
    const rest = disk.Rest.of(e);
    defer rest.done();
    dl.other(e, id) catch |err| std.log.warn("glm: the other ranks were not told to remove an identity ({s})", .{@errorName(err)});
}

fn begin(ctx: *anyopaque, saved: Saved, at: u32, files: []const rows.local.File) anyerror!void {
    const e = engineOf(ctx);
    const fs = try e.gpa.alloc(dl.File, files.len);
    defer e.gpa.free(fs);
    for (files, fs) |f, *x| x.* = .{ .name = f.name, .kind = f.kind, .window = f.window, .k = f.k };
    const rest = disk.Rest.of(e);
    try dl.read(e, idOf(saved), at, fs);
    reading = rest;
}

fn end(ctx: *anyopaque, ok: bool) anyerror!bool {
    const e = engineOf(ctx);
    defer if (reading) |rest| {
        reading = null;
        rest.done();
    };
    return dl.status(e, ok);
}

fn lacks(ctx: *anyopaque, keys: []const rows.local.Key, out: []bool) anyerror!void {
    const e = engineOf(ctx);
    const ks = try e.gpa.alloc(dl.Key, keys.len);
    defer e.gpa.free(ks);
    for (keys, ks) |k, *x| x.* = .{ .stem = k.stem, .kind = k.kind };
    const rest = disk.Rest.of(e);
    defer rest.done();
    try dl.lacks(e, ks, out);
}

fn sweep(ctx: *anyopaque) anyerror!void {
    const e = engineOf(ctx);
    const rest = disk.Rest.of(e);
    defer rest.done();
    try dl.sweep(e);
}

fn verify(ctx: *anyopaque, saved: Saved, back: Saved, at: u32) anyerror!bool {
    const e = engineOf(ctx);
    const rest = disk.Rest.of(e);
    defer rest.done();
    return dl.verify(e, idOf(saved), idOf(back), at);
}
