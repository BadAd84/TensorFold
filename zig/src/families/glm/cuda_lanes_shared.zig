//! The lane backend's shared rounds: several streams' windows in one forward (Engine.verifyShared), their commits.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const Prof = @import("cuda_prof.zig").Prof;
const cl = @import("cuda_lanes.zig");
const Lanes = cl.Lanes;
const max_rows = cl.max_rows;

const be = lanes.backend;

/// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
pub fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
    if (windows.len > engine.max_shared) return error.WindowTooWide;
    var parts: [engine.max_shared]engine.Part = undefined;
    var ids: [engine.max_shared][max_rows]u32 = undefined;
    var sampled: [engine.max_shared][max_rows]u32 = undefined;
    var lane: [engine.max_shared]*cl.Lane = undefined;
    var row: usize = 0; // the windows' rows so far: each one's grammar rows start there in the bits
    var structured = false;
    for (windows, 0..) |w, k| {
        if (w.parents != null) return error.TreesNotBuilt;
        const l = try self.bind(w.stream, null);
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        if (w.held > l.held_n) return error.NoHeldDrafts;
        const pos = self.x.e.pos();
        for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
        ids[k][0] = w.pending;
        const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
        @memcpy(ids[k][1..][0..drafts.len], drafts);
        const bits = try self.putBits(row, rows, w.masks);
        structured = structured or bits != 0;
        parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows], .bits = bits };
        lane[k] = l;
        row += rows;
    }
    self.x.e.structured = structured; // one window under a grammar: the whole step draws from whole rows
    defer self.x.e.structured = false;
    try self.x.verifyShared(parts[0..windows.len]);
    self.shared += 1;
    for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
        const rows = p.ids.len;
        @memcpy(o.sampled[0..rows], p.out);
        @memcpy(o.drafts[0 .. rows - 1], p.ids[1..rows]);
        l.pending_rows = rows;
    }
}

/// TF_GLM_OVN_SHARED_PROF: the commits `items`' streams owe (two or more with rows pending) begin, else null.
pub fn commitProf(self: *Lanes, items: anytype) !?*Prof {
    const sp = self.x.e.shared_prof orelse return null;
    var pending: usize = 0;
    for (items) |it| {
        if (self.lanes.getPtr(it.stream)) |l| pending += @intFromBool(l.pending_rows != null);
    }
    if (pending < 2) return null;
    return try sp.commit(self.x.e.stream, @min(pending, engine.max_shared));
}

/// The commits' span ends (commitProf's profiler, if any).
pub fn commitEnd(self: *Lanes, prof: ?*Prof) void {
    const p = prof orelse return;
    p.mark(self.x.e.stream, .commit);
    p.end();
}

/// TF_GLM_OVN_SEG_COMMIT: the commits `items`' streams owe (`kept[i]` rows; null: all their window's) as one step
/// when two or more are pending; each `bind` after it then finds nothing to commit.
pub fn commitShared(self: *Lanes, items: anytype, kept: []const ?usize) !void {
    if (items.len > engine.max_shared) return; // more than a round holds: each commits through bind as before
    var list: [engine.max_shared]@import("cuda_commit_seg.zig").Kept = undefined;
    var held: [engine.max_shared]*cl.Lane = undefined;
    var n: usize = 0;
    for (items, kept) |it, k| {
        const l = self.lanes.getPtr(it.stream) orelse return error.UnknownStream;
        const rows = l.pending_rows orelse continue;
        list[n] = .{ .slot = l.slot, .n = @intCast(k orelse rows) };
        held[n] = l;
        n += 1;
    }
    if (n < 2) return; // one alone commits through bind as before
    try self.x.keepShared(list[0..n]);
    for (held[0..n]) |l| l.pending_rows = null;
}

/// keepFn's paths, each checked first, then their rows through commitShared.
pub fn keepAll(self: *Lanes, windows: []const be.Window, paths: []const []const u32) !void {
    var kept: [engine.max_shared]?usize = undefined;
    if (windows.len > kept.len) return;
    for (paths, 0..) |path, i| {
        if (path.len == 0) return error.EmptyPath;
        for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
        kept[i] = path.len;
    }
    try commitShared(self, windows, kept[0..windows.len]);
}

/// draftBlock's requests' kept rows (null: the whole window's) through commitShared.
pub fn keepRequests(self: *Lanes, requests: []const be.DraftRequest) !void {
    var kept: [engine.max_shared]?usize = undefined;
    if (requests.len > kept.len) return;
    for (requests, 0..) |r, i| kept[i] = if (r.rows) |rows| rows.len else null;
    try commitShared(self, requests, kept[0..requests.len]);
}

comptime {
    std.debug.assert(max_rows <= engine.shared_rows);
}
