//! TF_GLM_OVN_REPLY_PREFILL (ReplyPrefill-1b): a reply's background pass (server/warm.zig) in short, yielding chunks.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const kda_chunk = @import("cuda_kda_chunk.zig");

/// Milliseconds a pass waits for quiet first (production's idle poll); a request in that time drops it untouched.
pub const default_quiet_ms: u64 = 10;
/// Rows of a pass's chunk: a request that arrives waits for one at most (0: the chunk grid's whole chunks).
pub const default_rows: u64 = 256;

pub const Settings = struct { quiet_ns: u64, rows: usize };

/// The knob's settings (null: off); refused: _QUIET_MS 0 or over a minute, _ROWS off the 32-row grid or over a chunk.
pub fn settings() error{BadReplyPrefill}!?Settings {
    if (!cuda.ovn.on("REPLY_PREFILL")) return null;
    const ms = cuda.ovn.int("REPLY_PREFILL_QUIET_MS", default_quiet_ms) catch return error.BadReplyPrefill;
    const rows = cuda.ovn.int("REPLY_PREFILL_ROWS", default_rows) catch return error.BadReplyPrefill;
    // KDA_CHUNKED keeps states on 32-row sub-chunks: a capped chunk ends on that grid; 0 ms would skip the quiet wait
    if (ms == 0 or ms > 60_000 or rows % kda_chunk.sub_rows != 0 or rows > engine.prompt_rows) return error.BadReplyPrefill;
    return .{ .quiet_ns = ms * std.time.ns_per_ms, .rows = @intCast(rows) };
}

/// The lane host's warm_quiet for GLM (cuda_native.warmQuietNow): null while the knob is off or refused.
pub fn quietNs() ?u64 {
    const s = (settings() catch return null) orelse return null;
    return s.quiet_ns;
}

/// A chunk's end for a pass capped at `rows`: `end`, or sooner at the next multiple of `rows` past `start` (0: `end`).
pub fn capEnd(start: usize, end: usize, rows: usize) usize {
    if (rows == 0) return end;
    return @min(end, (start / rows + 1) * rows);
}

/// The cap on `s`'s chunks: the knob's rows for a background pass, else 0 (its chunks as before).
pub fn rowsOf(s: *const lanes.Stream) usize {
    if (s.yield_check == null) return 0;
    const x = (settings() catch return 0) orelse return 0;
    return x.rows;
}

/// A chunk gate's check for `s`: `cancelled`, or for a background pass also real work waiting (it steps aside).
pub fn gate(s: *lanes.Stream, cancelled: *const fn (?*anyopaque) bool) engine.Cancel {
    return .{ .ptr = s, .check = if (s.yield_check != null) stops else cancelled };
}

/// A background pass stops at its next gate: cancelled, or real work waits (LaneHost.yieldsTo).
fn stops(ptr: ?*anyopaque) bool {
    const s: *lanes.Stream = @ptrCast(@alignCast(ptr.?));
    return s.isCancelled() or s.yields();
}

test "reply prefill: a capped chunk ends at the next multiple of its rows, never past its own end, 0 uncapped" {
    try std.testing.expectEqual(@as(usize, 256), capEnd(0, 2048, 256));
    try std.testing.expectEqual(@as(usize, 512), capEnd(300, 2048, 256)); // a resume off the cap's grid realigns
    try std.testing.expectEqual(@as(usize, 400), capEnd(300, 400, 256)); // a mark or the prompt's end comes first
    try std.testing.expectEqual(@as(usize, 768), capEnd(512, 2048, 256)); // on the grid: a whole cap
    try std.testing.expectEqual(@as(usize, 2048), capEnd(300, 2048, 0));
}

test "reply prefill: off unset, defaults when on, and bad rows or a 0 ms wait refused" {
    const ovn = cuda.ovn;
    defer ovn.fake("REPLY_PREFILL", null);
    defer ovn.fake("REPLY_PREFILL_QUIET_MS", null);
    defer ovn.fake("REPLY_PREFILL_ROWS", null);
    ovn.fake("REPLY_PREFILL", null);
    try std.testing.expectEqual(@as(?Settings, null), try settings());
    try std.testing.expectEqual(@as(?u64, null), quietNs());
    ovn.fake("REPLY_PREFILL", "1");
    ovn.fake("REPLY_PREFILL_QUIET_MS", null);
    ovn.fake("REPLY_PREFILL_ROWS", null);
    try std.testing.expectEqual(Settings{ .quiet_ns = 10 * std.time.ns_per_ms, .rows = 256 }, (try settings()).?);
    ovn.fake("REPLY_PREFILL_ROWS", "0");
    try std.testing.expectEqual(@as(usize, 0), (try settings()).?.rows);
    ovn.fake("REPLY_PREFILL_ROWS", "100");
    try std.testing.expectError(error.BadReplyPrefill, settings());
    ovn.fake("REPLY_PREFILL_ROWS", std.fmt.comptimePrint("{d}", .{engine.prompt_rows + 32}));
    try std.testing.expectError(error.BadReplyPrefill, settings());
    ovn.fake("REPLY_PREFILL_ROWS", "64");
    ovn.fake("REPLY_PREFILL_QUIET_MS", "0");
    try std.testing.expectError(error.BadReplyPrefill, settings());
    try std.testing.expectEqual(@as(?u64, null), quietNs());
    ovn.fake("REPLY_PREFILL_QUIET_MS", "60001");
    try std.testing.expectError(error.BadReplyPrefill, settings());
    ovn.fake("REPLY_PREFILL_QUIET_MS", "3");
    try std.testing.expectEqual(@as(?u64, 3 * std.time.ns_per_ms), quietNs());
}

test "reply prefill: a background pass's gate also yields and its chunks are capped; a request's are as before" {
    const ovn = cuda.ovn;
    defer ovn.fake("REPLY_PREFILL", null);
    ovn.fake("REPLY_PREFILL", "1");
    const Flags = struct {
        cancel: bool = false,
        yield: bool = false,
        fn cancelled(p: *anyopaque) bool {
            return @as(*@This(), @ptrCast(@alignCast(p))).cancel;
        }
        fn yielding(p: *anyopaque) bool {
            return @as(*@This(), @ptrCast(@alignCast(p))).yield;
        }
        fn plain(p: ?*anyopaque) bool {
            return @as(*lanes.Stream, @ptrCast(@alignCast(p.?))).isCancelled();
        }
    };
    var f: Flags = .{};
    var s: lanes.Stream = undefined;
    s.cancel_check = .{ .ptr = &f, .check = Flags.cancelled };
    s.yield_check = null;
    const request = gate(&s, Flags.plain);
    try std.testing.expectEqual(@as(usize, 0), rowsOf(&s));
    f.yield = true;
    try std.testing.expect(!request.check(request.ptr)); // a request's gate never yields
    s.yield_check = .{ .ptr = &f, .check = Flags.yielding };
    const pass = gate(&s, Flags.plain);
    try std.testing.expectEqual(@as(usize, default_rows), rowsOf(&s));
    try std.testing.expect(pass.check(pass.ptr)); // real work waits: it steps aside
    f.yield = false;
    try std.testing.expect(!pass.check(pass.ptr));
    f.cancel = true;
    try std.testing.expect(pass.check(pass.ptr)); // and a cancel still stops it
    ovn.fake("REPLY_PREFILL", null);
    try std.testing.expectEqual(@as(usize, 0), rowsOf(&s)); // the knob off: whole chunks
}
