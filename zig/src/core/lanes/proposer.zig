//! Copy proposals from a stream's own context (Python SuffixLookupProposer) behind a small proposer interface.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// What the round loop asks of a proposer: a copied continuation, its evidence, and how it fared.
pub const Proposer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        propose: *const fn (ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32,
        propose_after: ?*const fn (ptr: *anyopaque, history: []const u32, tail: []const u32, max_draft: i64) anyerror![]const u32 = null,
        last_match: *const fn (ptr: *anyopaque) i64,
        last_end: ?*const fn (ptr: *anyopaque) usize = null,
        observe: ?*const fn (ptr: *anyopaque, proposed: i64, accepted: i64) void = null,
        /// The proposer prices its own drafts: `propose`'s result is taken as is, no width, length or match gate.
        priced: bool = false,
        /// Every non-forced round of the stream, whatever drafted it: the window's rows and its drafts kept.
        round: ?*const fn (ptr: *anyopaque, rows: u32, kept: u32) void = null,
    };

    /// Tokens valid until the next `propose`.
    pub fn propose(p: Proposer, context: []const u32, max_draft: i64) ![]const u32 {
        return p.vtable.propose(p.ptr, context, max_draft);
    }

    /// Tail tokens are a query, never new history for the proposer's index.
    pub fn proposeAfter(p: Proposer, history: []const u32, tail: []const u32, max_draft: i64) ![]const u32 {
        if (p.vtable.propose_after) |call| return call(p.ptr, history, tail, max_draft);
        return if (tail.len == 0) p.propose(history, max_draft) else &.{};
    }

    pub fn lastMatch(p: Proposer) i64 {
        return p.vtable.last_match(p.ptr);
    }

    /// Where the last match's continuation starts in the history it searched (null: the proposer does not say).
    pub fn lastEnd(p: Proposer) ?usize {
        return if (p.vtable.last_end) |f| f(p.ptr) else null;
    }

    pub fn observe(p: Proposer, proposed: i64, accepted: i64) void {
        if (p.vtable.observe) |f| f(p.ptr, proposed, accepted);
    }

    pub fn round(p: Proposer, rows: u32, kept: u32) void {
        if (p.vtable.round) |f| f(p.ptr, rows, kept);
    }
};

pub const max_ngram = 8;
const Key = [max_ngram]u32;

pub const Params = struct {
    ngram: usize = 3,
    min_match: i64 = 6,
    max_extension: i64 = 64,
    /// Asks (`propose` or `proposeAfter` calls with room) left empty after `window` rejected proposals in a row.
    silence_rounds: i64 = 16,
    window: usize = 4,
    confident_match: i64 = 24,
    /// After a copy window misses, propose the source where it resumes past the edit, with no new suffix match.
    resync: bool = false,
    /// Resyncs that land nothing before the pending miss is dropped.
    resync_tries: u32 = 2,
    /// Source tokens past the miss searched for the last two tokens the target wrote.
    resync_span: usize = 24,
    /// ... and searched for the last token alone (an insertion, or a replacement longer than one token).
    resync_near: usize = 4,
    /// Tokens the target may write past a miss before the resync is dropped.
    resync_wait: usize = 32,
    /// Rows the missed copy must have landed to arm a resync: a copy that was copying, not a coincidence.
    resync_min_kept: i64 = 2,
    /// Only a copy whose source lies before this history index arms a resync (the host passes the prompt's length).
    resync_source_end: usize = std.math.maxInt(usize),
};

/// Continuations backed by enough matching context; silent for `silence_rounds` asks (not rounds) after rejections.
pub const SuffixLookup = struct {
    gpa: Allocator,
    params: Params,
    index: std.AutoHashMapUnmanaged(Key, std.ArrayList(u32)) = .empty,
    indexed: usize = 0,
    last_key: Key = undefined,
    recent: std.ArrayList(i64) = .empty,
    silent_for: i64 = 0,
    out: std.ArrayList(u32) = .empty,
    proposals: i64 = 0,
    proposed_tokens: i64 = 0,
    judged_tokens: i64 = 0,
    accepted_tokens: i64 = 0,
    silenced_rounds: i64 = 0, // asks the silence answered empty (Python's name: a round may ask more than once)
    last_match: i64 = 0,
    last_confident: bool = false,
    last_end: usize = 0, // the history index the last match's continuation starts at
    last_from: usize = 0, // where the last proposal that returned tokens read them from
    last_len: usize = 0, // the history's length at that proposal
    last_out: bool = false, // the latest ask returned tokens
    last_resync: bool = false, // ... and a resync, not a suffix match, placed them
    resync_miss: ?usize = null, // the history index of the first draft the target rejected (null: none pending)
    resync_base: usize = 0, // the history's length right after that round
    resync_tried: u32 = 0, // resyncs since that miss that landed nothing
    resyncs: i64 = 0, // proposals a resync placed

    pub fn init(gpa: Allocator, params: Params) !SuffixLookup {
        if (params.ngram < 1 or params.ngram > max_ngram) return error.BadNgram;
        if (params.min_match < @as(i64, @intCast(params.ngram))) return error.MinMatchBelowNgram;
        var p = params;
        p.window = @max(1, p.window);
        return .{ .gpa = gpa, .params = p };
    }

    pub fn deinit(s: *SuffixLookup) void {
        s.clearIndex();
        s.index.deinit(s.gpa);
        s.recent.deinit(s.gpa);
        s.out.deinit(s.gpa);
    }

    pub fn proposer(s: *SuffixLookup) Proposer {
        return .{ .ptr = s, .vtable = &vtable };
    }

    pub const vtable: Proposer.VTable = .{ .propose = proposeFn, .propose_after = afterFn, .last_match = lastMatchFn, .last_end = lastEndFn, .observe = observeFn };

    fn proposeFn(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.propose(context, max_draft);
    }

    fn afterFn(ptr: *anyopaque, history: []const u32, tail: []const u32, max_draft: i64) anyerror![]const u32 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.proposeAfter(history, tail, max_draft);
    }

    fn lastMatchFn(ptr: *anyopaque) i64 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.last_match;
    }

    fn lastEndFn(ptr: *anyopaque) usize {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.last_end;
    }

    fn observeFn(ptr: *anyopaque, proposed: i64, accepted: i64) void {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        s.observe(proposed, accepted) catch {};
    }

    fn clearIndex(s: *SuffixLookup) void {
        var it = s.index.valueIterator();
        while (it.next()) |list| list.deinit(s.gpa);
        s.index.clearRetainingCapacity();
    }

    fn keyAt(s: *const SuffixLookup, context: []const u32, end: usize) Key {
        var key: Key = @splat(0xFFFF_FFFF);
        const n = s.params.ngram;
        @memcpy(key[0..n], context[end - n .. end]);
        return key;
    }

    fn extend(s: *SuffixLookup, context: []const u32) !void {
        const n = s.params.ngram;
        var position = @max(s.indexed, n - 1);
        while (position < context.len) : (position += 1) {
            const got = try s.index.getOrPut(s.gpa, s.keyAt(context, position + 1));
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(s.gpa, @intCast(position));
        }
        s.indexed = context.len;
    }

    /// The copied continuation is borrowed until the next proposal.
    pub fn propose(s: *SuffixLookup, history: []const u32, max_draft: i64) ![]const u32 {
        return s.proposeAfter(history, &.{}, max_draft);
    }

    /// Match history followed by a drafted tail; only history supplies candidates and copied tokens.
    pub fn proposeAfter(s: *SuffixLookup, history: []const u32, tail: []const u32, max_draft: i64) ![]const u32 {
        const n = s.params.ngram;
        s.out.clearRetainingCapacity();
        s.last_match = 0;
        s.last_confident = false;
        s.last_out = false;
        s.last_resync = false;
        const changed = s.indexed > history.len or (s.indexed != 0 and !std.mem.eql(u32, &s.keyAt(history, s.indexed), &s.last_key));
        if (changed) {
            s.clearIndex();
            s.indexed = 0;
            s.resync_miss = null;
        }
        if (max_draft <= 0 or history.len < n + 1) return s.out.items;
        if (s.silent_for > 0) {
            s.silent_for -= 1;
            s.silenced_rounds += 1;
            return s.out.items;
        }
        try s.extend(history);
        s.last_key = s.keyAt(history, s.indexed);
        const len = std.math.add(usize, history.len, tail.len) catch return error.ContextTooLong;
        var key: Key = @splat(0xFFFF_FFFF);
        for (key[0..n], len - n..) |*k, i| k.* = joined(history, tail, i);
        var best_end: usize = 0;
        var best_len: i64 = 0;
        if (s.index.get(key)) |list| {
            const positions = list.items;
            var i = positions.len;
            while (i > 0) {
                i -= 1;
                const end: usize = positions[i] + 1;
                if (end >= history.len) continue;
                var length: i64 = 0;
                while (length < s.params.max_extension and length < end and history[end - 1 - @as(usize, @intCast(length))] == joined(history, tail, len - 1 - @as(usize, @intCast(length)))) length += 1;
                if (length > best_len) {
                    best_len = length;
                    best_end = end;
                    if (length >= s.params.max_extension) break;
                }
            }
        }
        s.last_match = best_len;
        s.last_end = best_end;
        if (best_len < s.params.min_match) {
            // no suffix match right after an edit: the source's continuation past it, as if it had matched
            best_end = s.resyncFrom(history, tail) orelse return s.out.items;
            s.last_resync = true;
            s.last_match = s.params.confident_match;
            s.last_end = best_end;
        } else s.last_confident = best_len >= s.params.confident_match;
        const count = @min(history.len - best_end, @as(usize, @intCast(max_draft)));
        try s.out.appendSlice(s.gpa, history[best_end..][0..count]);
        if (s.out.items.len > 0) {
            s.proposals += 1;
            s.proposed_tokens += @intCast(s.out.items.len);
            s.resyncs += @intFromBool(s.last_resync);
            s.last_out = true;
            s.last_from = best_end;
            s.last_len = history.len;
        }
        return s.out.items;
    }

    /// Where the source resumes past the pending miss: a deletion, a substitution, or the tokens written matched.
    fn resyncFrom(s: *const SuffixLookup, history: []const u32, tail: []const u32) ?usize {
        if (!s.params.resync or tail.len != 0) return null;
        const miss = s.resync_miss orelse return null;
        if (history.len < s.resync_base or history.len - s.resync_base >= s.params.resync_wait) return null;
        if (s.resync_tried >= s.params.resync_tries) return null;
        const written = history[s.resync_base - 1 ..];
        if (written.len == 0) return null;
        const last = written[written.len - 1];
        const end = history.len - written.len;
        if (written.len >= 2) {
            const prev = written[written.len - 2];
            var j = miss;
            while (j + 1 < end and j <= miss + s.params.resync_span) : (j += 1) {
                if (j >= 1 and history[j - 1] == prev and history[j] == last) return j + 1;
            }
        }
        var j = miss;
        while (j + 1 < end and j <= miss + s.params.resync_near) : (j += 1) {
            if (history[j] == last) return j + 1;
        }
        return if (written.len == 1 and miss + 1 < history.len) miss + 1 else null;
    }

    /// A missed copy that was landing arms a resync; false: a resync landed nothing, so it spends a try alone.
    fn observeResync(s: *SuffixLookup, proposed: i64, accepted: i64) bool {
        if (accepted >= proposed) {
            s.resync_miss = null;
            return true;
        }
        if (s.last_resync and accepted == 0) {
            s.resync_tried += 1;
            return false;
        }
        s.resync_miss = null;
        if (accepted >= s.params.resync_min_kept and s.last_from < s.params.resync_source_end) {
            s.resync_miss = s.last_from + @as(usize, @intCast(accepted));
            s.resync_base = s.last_len + @as(usize, @intCast(accepted)) + 1;
            s.resync_tried = 0;
        }
        return true;
    }

    fn joined(history: []const u32, tail: []const u32, i: usize) u32 {
        return if (i < history.len) history[i] else tail[i - history.len];
    }

    pub fn observe(s: *SuffixLookup, proposed: i64, accepted: i64) !void {
        s.judged_tokens += proposed;
        s.accepted_tokens += accepted;
        if (proposed <= 0) return;
        if (s.params.resync and !s.observeResync(proposed, accepted)) return;
        try s.recent.append(s.gpa, accepted);
        if (s.recent.items.len > s.params.window) {
            const drop = s.recent.items.len - s.params.window;
            std.mem.copyForwards(i64, s.recent.items, s.recent.items[drop..]);
            s.recent.shrinkRetainingCapacity(s.params.window);
        }
        if (s.recent.items.len >= s.params.window and std.mem.max(i64, s.recent.items) == 0) {
            s.silent_for = s.params.silence_rounds;
            s.recent.clearRetainingCapacity();
        }
    }
};

test "suffix lookup copies the continuation after a repeated n-gram" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2 });
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3 };
    const got = try s.propose(&ctx, 3);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, got);
    try std.testing.expectEqual(@as(i64, 3), s.last_match);
}

test "a resync proposes the source past an edit, realigns on what the target wrote, and gives up bounded" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2, .resync = true });
    defer s.deinit();
    var ctx: std.ArrayList(u32) = .empty;
    defer ctx.deinit(gpa);
    try ctx.appendSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3 });
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7, 8 }, try s.propose(ctx.items, 5));
    try std.testing.expect(!s.last_resync);
    // the target kept 4 and 5, then said 50 for 6 (a substitution): the context grew by the kept rows and its own
    try s.observe(5, 2);
    try ctx.appendSlice(gpa, &.{ 4, 5, 50 });
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9, 1, 2 }, try s.propose(ctx.items, 5));
    try std.testing.expect(s.last_resync and s.last_match >= 24);
    try std.testing.expectEqual(@as(usize, 6), s.last_end);
    // the same proposal again on the same context, at the width asked
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, try s.propose(ctx.items, 2));
    try s.observe(2, 2);
    try ctx.appendSlice(gpa, &.{ 7, 8, 60 });
    try std.testing.expectEqual(@as(usize, 0), (try s.propose(ctx.items, 5)).len);
    try std.testing.expectEqual(@as(?usize, null), s.resync_miss);
    // a deletion: copying 1 2 3, the target skips 4 and says 5 at once, so the source resumes at 6
    var t = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2, .resync = true, .resync_min_kept = 0 });
    defer t.deinit();
    var c2: std.ArrayList(u32) = .empty;
    defer c2.deinit(gpa);
    try c2.appendSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3 });
    _ = try t.propose(c2.items, 3);
    try t.observe(3, 0);
    try c2.append(gpa, 5);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7, 8 }, try t.propose(c2.items, 3));
    // a resync that lands nothing spends a try; while the target is still inside its edit it gets no proposal
    try t.observe(3, 0);
    try std.testing.expectEqual(@as(u32, 1), t.resync_tried);
    try c2.appendSlice(gpa, &.{ 70, 71 });
    try std.testing.expectEqual(@as(usize, 0), (try t.propose(c2.items, 3)).len);
    try c2.append(gpa, 6);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, try t.propose(c2.items, 3));
    try std.testing.expect(t.last_resync);
    // two resyncs that land nothing end this miss, and none of them silences the proposer
    try t.observe(3, 0);
    try c2.append(gpa, 80);
    try std.testing.expectEqual(@as(usize, 0), (try t.propose(c2.items, 3)).len);
    try std.testing.expectEqual(@as(i64, 0), t.silent_for);
}

test "a resync is armed only by a copy that was landing, and only out of the prompt" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2, .resync = true });
    defer s.deinit();
    var ctx: std.ArrayList(u32) = .empty;
    defer ctx.deinit(gpa);
    try ctx.appendSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3 });
    // a coincidental match the target leaves at once: nothing pends, the next ask needs a suffix match again
    _ = try s.propose(ctx.items, 3);
    try s.observe(3, 1);
    try ctx.appendSlice(gpa, &.{ 4, 50 });
    try std.testing.expectEqual(@as(?usize, null), s.resync_miss);
    try std.testing.expectEqual(@as(usize, 0), (try s.propose(ctx.items, 3)).len);
    // a reply repeating its own lines (its source past the prompt) never resyncs
    var t = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2, .resync = true, .resync_source_end = 2 });
    defer t.deinit();
    var c2: std.ArrayList(u32) = .empty;
    defer c2.deinit(gpa);
    try c2.appendSlice(gpa, &.{ 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3 });
    _ = try t.propose(c2.items, 3);
    try t.observe(3, 2);
    try c2.appendSlice(gpa, &.{ 4, 5, 50 });
    try std.testing.expectEqual(@as(?usize, null), t.resync_miss);
    try std.testing.expectEqual(@as(usize, 0), (try t.propose(c2.items, 3)).len);
    // the same copy out of the prompt does arm one
    var u = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2, .resync = true, .resync_source_end = 11 });
    defer u.deinit();
    var c3: std.ArrayList(u32) = .empty;
    defer c3.deinit(gpa);
    try c3.appendSlice(gpa, &.{ 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3 });
    _ = try u.propose(c3.items, 3);
    try u.observe(3, 2);
    try c3.appendSlice(gpa, &.{ 4, 5, 50 });
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, try u.propose(c3.items, 3));
}

test "resync off leaves every proposal and the silence rule as they were" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2 });
    defer s.deinit();
    var ctx: std.ArrayList(u32) = .empty;
    defer ctx.deinit(gpa);
    try ctx.appendSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3 });
    _ = try s.propose(ctx.items, 5);
    try s.observe(5, 2);
    try ctx.appendSlice(gpa, &.{ 4, 5, 50 });
    try std.testing.expectEqual(@as(usize, 0), (try s.propose(ctx.items, 5)).len);
    try std.testing.expectEqual(@as(?usize, null), s.resync_miss);
    try std.testing.expect(!s.last_resync);
}

test "a drafted tail continues the history's copy, and history alone is indexed" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4 });
    defer s.deinit();
    const history = [_]u32{ 7, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1 };
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 8 }, try s.proposeAfter(&history, &.{ 2, 3, 4 }, 3));
    try std.testing.expectEqual(@as(i64, 5), s.last_match);
    try std.testing.expectEqual(history.len, s.indexed);
    try std.testing.expectEqual(@as(usize, 0), (try s.proposeAfter(&history, &.{ 2, 3, 9 }, 3)).len);
}

test "silence lasts silence_rounds asks however many a round makes; round() and an ask without room leave it" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2, .silence_rounds = 3, .window = 2 });
    defer s.deinit();
    const p = s.proposer();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3 };
    try s.observe(3, 0);
    try s.observe(3, 0);
    try std.testing.expectEqual(@as(i64, 3), s.silent_for);
    p.round(4, 0); // the round hook: SuffixLookup has none, so rounds never shorten the silence
    try std.testing.expectEqual(@as(usize, 0), (try p.proposeAfter(&ctx, &.{}, 0)).len); // no room: not an ask
    try std.testing.expectEqual(@as(i64, 3), s.silent_for);
    try std.testing.expectEqual(@as(usize, 0), (try p.proposeAfter(&ctx, &.{3}, 3)).len); // root copies' first ask
    try std.testing.expectEqual(@as(usize, 0), (try p.propose(&ctx, 3)).len);
    p.round(4, 0);
    try std.testing.expectEqual(@as(usize, 0), (try p.propose(&ctx, 3)).len);
    try std.testing.expectEqual(@as(i64, 3), s.silenced_rounds);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, try p.propose(&ctx, 3));
}
