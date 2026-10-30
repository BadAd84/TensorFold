//! TF_GLM_OVN_DISK_LOCAL, rank 0's side: every other rank keeps its own part of each state on its own disk under the
//! same names; rank 0's store decides (names, budget, eviction) and the others follow through a family's Mirror. A
//! restore is every rank's part or none: a rank that lacks its part makes every rank drop the state together.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const chain = @import("rows_chain.zig");
const file = @import("lanes").rows_file;
const readers = @import("lanes").disk_readers;
const write = @import("rows_write.zig");
const Rows = @import("rows_store.zig").Rows;
const Name = chain.Name;
const Stem = chain.Stem;

/// A file of a state as the other ranks stage or read it: its name, its kind and (a block) its group and index.
pub const File = struct { name: Name, kind: file.Kind, window: u32 = 0, k: u32 = 0 };

/// A file as the other ranks key it on disk: its name's stem and its kind.
pub const Key = struct { stem: Stem, kind: file.Kind };

/// What the other ranks do when rank 0's store acts (a family's: GLM sends each as a disk step on its link).
pub const Mirror = struct {
    ctx: *anyopaque,
    /// A file rank 0 staged of `saved` at `at`: every other rank stages its own part of it and writes it behind.
    file: *const fn (ctx: *anyopaque, saved: pc.Saved, at: u32, f: File) anyerror!void,
    /// A file rank 0's store forgot: every other rank removes its part (after any write of it queued before).
    unlink: *const fn (ctx: *anyopaque, k: Key) void,
    /// Another identity's directory rank 0 removed for room: every other rank removes its own.
    other: *const fn (ctx: *anyopaque, id: u64) void,
    /// Every other rank reads its parts of `files` (blocks, its own file last) into blank `saved`, on its own disk.
    begin: *const fn (ctx: *anyopaque, saved: pc.Saved, at: u32, files: []const File) anyerror!void,
    /// Rank 0's outcome `ok` beside every other rank's: true only when every rank's part arrived.
    end: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
    /// Startup: for each key, whether some other rank lacks its part (`out` set per key).
    lacks: *const fn (ctx: *anyopaque, keys: []const Key, out: []bool) anyerror!void,
    /// Startup, after `agree`: each other rank's files no key named go.
    sweep: *const fn (ctx: *anyopaque) anyerror!void,
    /// TF_IMPRINT_VERIFY=1: whether every other rank's part of `back` (just read) equals its part of `saved`.
    verify: ?*const fn (ctx: *anyopaque, saved: pc.Saved, back: pc.Saved, at: u32) anyerror!bool = null,
    /// Pins the read slots for one host-to-device hop (null: pageable).
    pin: ?readers.Pin = null,
};

/// Startup: a state some rank lacks a file of is forgotten on every rank, so no restore is ever partial; then the
/// other ranks' leftovers go. Returns the states forgotten.
pub fn agree(r: *Rows) !usize {
    const m = r.mirror orelse return 0;
    var keys: std.ArrayList(Key) = .empty;
    defer keys.deinit(r.gpa);
    for (r.states.items) |s| try keys.append(r.gpa, .{ .stem = chain.stem(s.name), .kind = .state });
    var it = r.blocks.keyIterator();
    while (it.next()) |k| try keys.append(r.gpa, .{ .stem = k.*, .kind = .block });
    const lack = try r.gpa.alloc(bool, keys.items.len);
    defer r.gpa.free(lack);
    @memset(lack, false);
    try m.lacks(m.ctx, keys.items, lack);
    for (keys.items, lack) |k, l| if (l and k.kind == .block) {
        r.unlink(k.stem, .block);
        _ = r.blocks.remove(k.stem);
    };
    var gone: usize = 0;
    var wants: std.ArrayList(write.Want) = .empty;
    defer wants.deinit(r.gpa);
    var i: usize = 0;
    while (i < r.states.items.len) {
        const s = &r.states.items[i];
        var whole = !lacks(keys.items, lack, chain.stem(s.name), .state);
        wants.clearRetainingCapacity();
        r.wantsOf(r.gpa, s, &wants) catch {
            whole = false;
        };
        for (wants.items) |w| whole = whole and r.blocks.contains(chain.stem(w.name));
        if (whole) {
            i += 1;
            continue;
        }
        r.remove(i);
        gone += 1;
    }
    r.collect();
    try m.sweep(m.ctx);
    return gone;
}

fn lacks(keys: []const Key, lack: []const bool, stem: Stem, kind: file.Kind) bool {
    for (keys, lack) |k, l| if (k.kind == kind and std.mem.eql(u8, &k.stem, &stem)) return l;
    return false;
}

/// A job that failed or was cancelled: every other rank removes the parts it staged that rank 0's store does not hold.
pub fn forget(r: *Rows, j: *const write.io.Job) void {
    const m = r.mirror orelse return;
    for (j.blocks()) |f| {
        const s = chain.stem(f.trailer.name);
        if (!r.blocks.contains(s)) m.unlink(m.ctx, .{ .stem = s, .kind = .block });
    }
    if (r.find(j.name) == null) m.unlink(m.ctx, .{ .stem = chain.stem(j.name), .kind = .state });
}
