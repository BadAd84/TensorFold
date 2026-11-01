//! GLM's prompt-lookup ("copy") drafts, ported from our glm-cuda family's `cuda/glm_copy.zig` (fork main 55d5636d,
//! itself from v1.8's glm5_next/cuda/copy_drafts.py, patches 0007, 0013 and 0032): when the context's last `match`
//! tokens (the pending token included) occurred before, the tokens that followed their latest earlier occurrence
//! are verified in place of the DFlash2 head's drafts. They only propose; verification keeps the serial sample.
//!
//! Upstream's lanes core already carries a copy proposer (`lanes.proposer.SuffixLookup`, a 3-gram index with a
//! silence rule) and the family hook for one (`LaneHost.proposers`, as deepseek_v41's priced lookup uses). GLM's
//! five rules differ from it, so this family brings its own behind the same `Proposer` interface:
//! - which occurrence: the latest one with `k` tokens after it, else the earliest (not the longest match);
//! - an occurrence that starts inside the reply must match `reply_match` tokens (COPY_CODE: 16), not `match` (8):
//!   code replies repeat short boilerplate whose continuations differ;
//! - a window of 9 to 15 rows is padded to the CUDA-graph width (16) by repeating its last draft, when room allows
//!   (TF_GLM_COPY_PAD; off by default, the GLM family runs no CUDA-graph replay so the pad is off unless asked);
//! - TF_GLM_OVN_COPY_AGREE (off): the occurrence agreeing furthest back past the match, capped when agreement is weak.
//! - TF_GLM_OVN_COPY_MISS_MAX (off): after a copy round that kept fewer than all its drafts, the next caps at K.
//! No silence, and a one-token copy counts (the proposer is `priced`), so the round loop takes its drafts as given.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");

/// The most drafts a round: a 16-row window (the GLM family's max_rows - 1).
pub const max_drafts = 15;

const agree_hits = 64; // TF_GLM_OVN_COPY_AGREE L=1: the latest this many occurrences with room are compared (AGREE_HITS)

pub const Settings = struct {
    on: bool = true, // TF_GLM_COPY_DRAFTS (production: 1)
    match: usize = 8, // TF_GLM_COPY_MATCH
    most: usize = max_drafts, // TF_GLM_COPY_MAX (production: 15; Python's own default is 5)
    reply_match: usize = 16, // TF_GLM_COPY_REPLY_MATCH (production COPY_CODE=1: 16; 0: the prompt's rule everywhere)
    pad: usize = 0, // TF_GLM_COPY_PAD (the old graph width, 16; 0: no padding, upstream's family runs no graphs)
    miss_most: usize = 0, // TF_GLM_OVN_COPY_MISS_MAX: a copy round after one that missed proposes at most this many
    agree: usize = 0, // TF_GLM_OVN_COPY_AGREE LO: agreement tokens past the match trusted (0: off, today's choice)
    agree_most: usize = 0, // K: a proposal agreeing for under LO tokens proposes at most this many drafts
    agree_longest: bool = true, // L: propose the occurrence agreeing furthest back (0: keep the baseline occurrence)

    /// The production settings, from the TF_GLM_COPY_* variables.
    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        const on = std.c.getenv("TF_GLM_COPY_DRAFTS");
        if (on) |v| {
            const t = std.mem.trim(u8, std.mem.span(v), " ");
            if (t.len == 0 or std.mem.eql(u8, t, "0")) s.on = false else if (!std.mem.eql(u8, t, "1")) {
                std.log.err("TF_GLM_COPY_DRAFTS is 0 or 1, not {s}", .{t});
                return error.BadCopySettings;
            }
        }
        s.match = try number("TF_GLM_COPY_MATCH", s.match, 2, 64);
        s.most = try number("TF_GLM_COPY_MAX", s.most, 1, max_drafts);
        s.reply_match = try number("TF_GLM_COPY_REPLY_MATCH", s.reply_match, 0, 64);
        s.pad = try number("TF_GLM_COPY_PAD", s.pad, 0, max_drafts + 1);
        s.miss_most = try number("TF_GLM_OVN_COPY_MISS_MAX", 0, 0, s.most);
        if (s.reply_match != 0 and s.reply_match <= s.match) {
            std.log.err("TF_GLM_COPY_REPLY_MATCH is 0 (off) or more than TF_GLM_COPY_MATCH ({d}), not {d}", .{ s.match, s.reply_match });
            return error.BadCopySettings;
        }
        if (std.c.getenv("TF_GLM_OVN_COPY_AGREE")) |v| {
            const t = std.mem.trim(u8, std.mem.span(v), " ");
            const a = parseAgreeSpec(t, s.most) catch {
                std.log.err("TF_GLM_OVN_COPY_AGREE is 0 (off) or LO:K[:L], LO 1 to 64, K 1 to TF_GLM_COPY_MAX, L 0 or 1, not {s}", .{t});
                return error.BadCopySettings;
            };
            if (a) |spec| {
                s.agree = spec.lo;
                s.agree_most = spec.k;
                s.agree_longest = spec.longest;
            }
        }
        return s;
    }

    fn number(name: [*:0]const u8, default: usize, low: usize, high: usize) !usize {
        const v = std.c.getenv(name) orelse return default;
        const t = std.mem.trim(u8, std.mem.span(v), " ");
        if (t.len == 0) return default;
        const n = std.fmt.parseInt(usize, t, 10) catch 0;
        if (n < low or n > high) {
            std.log.err("{s} is {d} to {d}, not {s}", .{ name, low, high, t });
            return error.BadCopySettings;
        }
        return n;
    }

    const AgreeSpec = struct { lo: usize, k: usize, longest: bool };

    /// Parse a trimmed TF_GLM_OVN_COPY_AGREE value: null when off ("" or "0"), else LO:K[:L] (copy_drafts.py:77-85).
    fn parseAgreeSpec(t: []const u8, most: usize) !?AgreeSpec {
        if (t.len == 0 or std.mem.eql(u8, t, "0")) return null;
        var it = std.mem.splitScalar(u8, t, ':');
        const lo = decimalPart(it.next()) orelse return error.BadCopySettings;
        const k = decimalPart(it.next()) orelse return error.BadCopySettings;
        var longest = true;
        if (it.next()) |l| {
            if (std.mem.eql(u8, l, "0")) longest = false else if (!std.mem.eql(u8, l, "1")) return error.BadCopySettings;
        }
        if (it.next() != null) return error.BadCopySettings; // LO:K[:L] is two or three fields, no more
        if (lo < 1 or lo > 64 or k < 1 or k > most) return error.BadCopySettings;
        return .{ .lo = lo, .k = k, .longest = longest };
    }

    /// A non-empty run of ASCII digits as a usize (Python's str.isdecimal), else null: no sign, separator or empty.
    fn decimalPart(part: ?[]const u8) ?usize {
        const t = part orelse return null;
        if (t.len == 0) return null;
        for (t) |c| if (c < '0' or c > '9') return null;
        return std.fmt.parseInt(usize, t, 10) catch null;
    }

    /// Whether `ctx[i .. i + match]` is an earlier occurrence of the context's last `match` tokens (`prompt` leading
    /// tokens are the prompt's: an occurrence starting past them also needs the tokens before it to match).
    fn occurs(s: Settings, ctx: []const u32, prompt: usize, i: usize) bool {
        const n = s.match;
        const len = ctx.len;
        if (!std.mem.eql(u32, ctx[i .. i + n], ctx[len - n ..])) return false;
        if (s.reply_match > n and i >= prompt) {
            const extra = s.reply_match - n;
            if (i < extra or len - n < extra) return false;
            for (1..extra + 1) |k| if (ctx[i - k] != ctx[len - n - k]) return false;
        }
        return true;
    }

    /// Whether TF_GLM_OVN_COPY_AGREE is active: the agreement rule acts only with a cap set (copy_drafts.py:125).
    fn agreeOn(s: Settings) bool {
        return s.agree != 0 and s.agree_most != 0;
    }

    /// Tokens before `start` equal to those before the matched suffix, back from the match up to `agree` (py:135-151).
    fn agreement(s: Settings, ctx: []const u32, start: usize, len: usize) usize {
        const n = s.match;
        var got: usize = 0;
        var j: usize = 1;
        while (j <= s.agree) : (j += 1) {
            if (len < n + j or start < j) break;
            if (ctx[start - j] != ctx[len - n - j]) break;
            got += 1;
        }
        return got;
    }

    /// The occurrence to copy from and the draft cap when `agree` is on (copy_drafts.py:169-175).
    fn agreeChoose(s: Settings, ctx: []const u32, prompt: usize, last: u32, k: *usize, baseline: usize, has_room: bool, len: usize) usize {
        const n = s.match;
        var start = baseline;
        var best_ag = s.agreement(ctx, baseline, len);
        if (s.agree_longest and has_room) {
            // the latest agree_hits occurrences with room: the one agreeing furthest back, ties the latest
            var count: usize = 1; // the baseline is the latest occurrence with room
            var i = baseline;
            while (count < agree_hits and i > 0) {
                i -= 1;
                if (ctx[i + n - 1] == last and s.occurs(ctx, prompt, i)) {
                    count += 1;
                    const ag = s.agreement(ctx, i, len);
                    if (ag > best_ag) { // strictly greater, so a tie keeps the later occurrence
                        best_ag = ag;
                        start = i;
                    }
                }
            }
        }
        if (best_ag < s.agree) k.* = @min(k.*, s.agree_most);
        return start;
    }

    /// CopyDrafts.propose: up to `min(most, room)` drafts, what followed the latest earlier occurrence that has that many
    /// tokens after it, else the most after any occurrence (the earliest); none when the suffix never occurred before. A
    /// window of more than `graphed` (8) rows short of the `pad` width is filled up to it by repeating the last draft,
    /// when `room` allows. `ctx` is the prompt, the reply so far and the pending token; the result lives in `buf`.
    pub fn propose(s: Settings, ctx: []const u32, prompt: usize, room: usize, buf: *[max_drafts]u32) []const u32 {
        var k = @min(s.most, room);
        const n = s.match;
        const len = ctx.len;
        if (!s.on or k < 1 or len <= n) return buf[0..0];
        const last = ctx[len - 1];
        var found: ?usize = null;
        var has_room = false; // the latest start found has k tokens after its match (production's full.size > 0)
        if (len >= n + k) { // the latest start with k tokens after its match
            var i = len - n - k + 1;
            while (i > 0) {
                i -= 1;
                if (ctx[i + n - 1] == last and s.occurs(ctx, prompt, i)) {
                    found = i;
                    has_room = true;
                    break;
                }
            }
        }
        if (found == null) { // none: the earliest start, which leaves fewer than k tokens
            var i = if (len >= n + k) len - n - k + 1 else 0;
            while (i < len - n) : (i += 1) {
                if (ctx[i + n - 1] == last and s.occurs(ctx, prompt, i)) {
                    found = i;
                    break;
                }
            }
        }
        var start = found orelse return buf[0..0];
        if (s.agreeOn()) start = s.agreeChoose(ctx, prompt, last, &k, start, has_room, len);
        const at = start + n;
        const count = @min(k, len - at);
        @memcpy(buf[0..count], ctx[at..][0..count]);
        const rows = 1 + count;
        const graphed = 8;
        if (s.pad > graphed and rows > graphed and rows < s.pad and s.pad - 1 <= room) {
            @memset(buf[count .. s.pad - 1], buf[count - 1]);
            return buf[0 .. s.pad - 1];
        }
        return buf[0..count];
    }
};

/// One request's proposer: the family's rules over the stream's own context, behind lanes.proposer.Proposer.
/// The lanes core hands `proposeAfter` the history and the pending token as a query; this proposer rejoins
/// them into the `ctx` its rules expect (the reply's tail inclusive) and reports `match` as its evidence.
pub const GlmProposer = struct {
    settings: Settings,
    prompt: usize, // the request's prompt tokens, the reply boundary `reply_match` needs
    buf: [max_drafts]u32 = undefined,
    out: std.ArrayList(u32) = .empty,
    gpa: Allocator,
    matched: i64 = 0, // the last proposal's match length (0: none)
    missed: bool = false, // the last settled copy round kept fewer than all its drafts (TF_GLM_OVN_COPY_MISS_MAX)

    pub fn deinit(p: *GlmProposer) void {
        p.out.deinit(p.gpa);
    }

    fn proposeTokens(p: *GlmProposer, history: []const u32, tail: []const u32, max_draft: i64) ![]const u32 {
        const room: usize = if (max_draft < 0) 0 else @intCast(@min(max_draft, max_drafts));
        const n = p.settings.match;
        if (n == 0 or n > history.len + tail.len or tail.len >= n) return error.BadCopyContext;
        // the context the rules see: the history with the pending token (and any forced tail) joined on,
        // so the reply's tail is the context's tail as CopyDrafts.propose expects
        const ctx = try p.gpa.alloc(u32, history.len + tail.len);
        defer p.gpa.free(ctx);
        @memcpy(ctx[0..history.len], history);
        @memcpy(ctx[history.len..], tail);
        var eff = p.settings;
        if (p.missed and eff.miss_most != 0) eff.most = @min(eff.most, eff.miss_most); // miss cap: narrow after a miss
        const got = eff.propose(ctx, p.prompt, room, &p.buf);
        p.matched = if (got.len > 0) @intCast(n) else 0;
        p.out.clearRetainingCapacity();
        try p.out.appendSlice(p.gpa, got);
        return p.out.items;
    }

    fn proposeFn(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const p: *GlmProposer = @ptrCast(@alignCast(ptr));
        return p.proposeTokens(context, &.{}, max_draft);
    }

    fn afterFn(ptr: *anyopaque, history: []const u32, tail: []const u32, max_draft: i64) anyerror![]const u32 {
        const p: *GlmProposer = @ptrCast(@alignCast(ptr));
        return p.proposeTokens(history, tail, max_draft);
    }

    fn lastMatchFn(ptr: *anyopaque) i64 {
        const p: *GlmProposer = @ptrCast(@alignCast(ptr));
        return p.matched;
    }

    fn observeFn(ptr: *anyopaque, proposed: i64, accepted: i64) void {
        const p: *GlmProposer = @ptrCast(@alignCast(ptr));
        // the engine settles a copy round here (p.context_copy): it missed when fewer drafts were kept than proposed
        if (proposed > 0) p.missed = accepted < proposed;
    }

    pub const vtable: lanes.proposer.Proposer.VTable = .{
        .propose = proposeFn,
        .propose_after = afterFn,
        .last_match = lastMatchFn,
        .priced = true, // the rules set their own length: a one-token copy is a copy, no core gate
        .observe = observeFn,
    };

    pub fn proposer(p: *GlmProposer) lanes.proposer.Proposer {
        return .{ .ptr = p, .vtable = &vtable };
    }
};

/// LaneHost.proposers: a GlmProposer per request, freed with it (lane_stream.zig's freeProposer).
fn makeProposer(ctx: *anyopaque, gpa: Allocator, prompt: []const u32, _: []const u32) anyerror!lanes.proposer.Proposer {
    const s: *const Settings = @ptrCast(@alignCast(ctx));
    const p = try gpa.create(GlmProposer);
    p.* = .{ .settings = s.*, .prompt = prompt.len, .gpa = gpa };
    return p.proposer();
}

fn freeProposer(_: *anyopaque, p: lanes.proposer.Proposer) void {
    const glm_p: *GlmProposer = @ptrCast(@alignCast(p.ptr));
    const gpa = glm_p.gpa;
    glm_p.deinit();
    gpa.destroy(glm_p);
}

/// The factory `cuda_native.copyProposers` hands `native/cuda.zig`'s Host.serve. The glm_cuda module
/// cannot name the core's Proposers type (no engine_api dep), so this mirrors its three fields; the
/// anonymous literal serve builds from them coerces (function pointer types match structurally).
pub const Factory = struct {
    ctx: *anyopaque,
    make: *const fn (ctx: *anyopaque, gpa: Allocator, prompt: []const u32, eos: []const u32) anyerror!lanes.proposer.Proposer,
    free: *const fn (ctx: *anyopaque, p: lanes.proposer.Proposer) void,
};

/// The family's copy-proposer factory for `native/cuda.zig`'s Host.serve (null: TF_GLM_COPY_DRAFTS=0).
pub fn proposers() ?Factory {
    const s = Settings.fromEnv() catch |e| {
        std.log.err("glm copy drafts: bad TF_GLM_COPY_* settings ({s})", .{@errorName(e)});
        return null;
    };
    if (!s.on) return null;
    settings = s;
    return .{ .ctx = &settings, .make = makeProposer, .free = freeProposer };
}

var settings: Settings = .{};

test "the latest occurrence with room, else the earliest" {
    const s: Settings = .{ .match = 2, .most = 3, .reply_match = 0, .pad = 0 };
    var buf: [max_drafts]u32 = undefined;
    // the suffix (1 2) occurred at 0 and 5, both with 3 tokens after them: the later one
    const ctx = [_]u32{ 1, 2, 5, 6, 7, 1, 2, 8, 9, 1, 2 };
    try std.testing.expectEqualSlices(u32, &.{ 8, 9, 1 }, s.propose(&ctx, 0, 10, &buf));
    // room 1 is one draft, from the same occurrence
    try std.testing.expectEqualSlices(u32, &.{8}, s.propose(&ctx, 0, 1, &buf));
    // (3 4) occurred once, at 0, with only 3 tokens after it where 4 are wanted: the earliest start, short
    const wide: Settings = .{ .match = 2, .most = 4, .reply_match = 0, .pad = 0 };
    const short = [_]u32{ 3, 4, 7, 3, 4 };
    try std.testing.expectEqualSlices(u32, &.{ 7, 3, 4 }, wide.propose(&short, 0, 10, &buf));
    // never seen
    const fresh = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 0), s.propose(&fresh, 0, 10, &buf).len);
}

test "an occurrence inside the reply needs the reply's longer match" {
    const s: Settings = .{ .match = 2, .most = 2, .reply_match = 4, .pad = 0 };
    var buf: [max_drafts]u32 = undefined;
    // prompt 9 9, reply 1 2 5 7 1 2: the occurrence of (1 2) at 2 starts in the reply, and the token before it (9) is not
    // the one before the suffix (7)
    const ctx = [_]u32{ 9, 9, 1, 2, 5, 7, 1, 2 };
    try std.testing.expectEqual(@as(usize, 0), s.propose(&ctx, 2, 10, &buf).len);
    // with the tokens before it equal, it copies
    const same = [_]u32{ 9, 9, 6, 6, 1, 2, 5, 6, 6, 1, 2 };
    try std.testing.expectEqualSlices(u32, &.{ 5, 6 }, s.propose(&same, 2, 10, &buf));
    // an occurrence in the prompt keeps the short match
    try std.testing.expectEqualSlices(u32, &.{ 5, 7 }, s.propose(&.{ 1, 2, 5, 7, 4, 1, 2 }, 6, 10, &buf));
}

test "a window of 9 to 15 rows is padded to the graph width when room allows" {
    const s: Settings = .{ .match = 2, .most = 15, .reply_match = 0, .pad = 16 };
    var buf: [max_drafts]u32 = undefined;
    var ctx: [30]u32 = undefined;
    for (&ctx, 0..) |*t, i| t.* = @intCast(100 + i);
    ctx[28] = 100;
    ctx[29] = 101; // the suffix (100 101) occurred at 0
    // 28 tokens follow the match at 0, so 15 drafts are plain
    try std.testing.expectEqual(@as(usize, 15), s.propose(&ctx, 0, 40, &buf).len);
    // room 10: ten drafts, 11 rows, below the pad's room (15): not padded
    try std.testing.expectEqual(@as(usize, 10), s.propose(&ctx, 0, 10, &buf).len);
    // a match with only 14 tokens after it (earliest, short): 15 rows, padded to 16 by repeating the last draft
    var short: [16]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = @intCast(200 + i);
    short[14] = 200;
    short[15] = 201;
    const got = s.propose(&short, 0, 40, &buf);
    try std.testing.expectEqual(@as(usize, 15), got.len);
    try std.testing.expectEqual(@as(u32, 201), got[14]); // 14 copied tokens end in 201, the pad repeats it
    try std.testing.expectEqual(got[13], got[14]);
}

test "the proposer rejoins the history and the pending token into one context" {
    var p: GlmProposer = .{ .settings = .{ .match = 2, .most = 3, .reply_match = 0, .pad = 0 }, .prompt = 2, .gpa = std.testing.allocator };
    defer p.deinit();
    // prompt 1 2, reply 5 6 7, pending 5: the context's tail (2 5) occurred at 1, and 6 7 1 follow it
    const got = try p.proposer().proposeAfter(&.{ 1, 2, 5, 6, 7, 1, 2 }, &.{5}, 10);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7, 1 }, got);
    try std.testing.expectEqual(@as(i64, 2), p.matched);
    // no occurrence: no drafts, matched 0
    const none = try p.proposer().proposeAfter(&.{ 5, 6, 7, 8 }, &.{9}, 10);
    try std.testing.expectEqual(@as(usize, 0), none.len);
    try std.testing.expectEqual(@as(i64, 0), p.matched);
}

// Shared agree vector: (1 2) occurs at 3 (preceded by 7 8 9, agrees 3 back) and 9 (preceded by 40, agrees 0 back).
const agree_ctx = [_]u32{ 7, 8, 9, 1, 2, 30, 31, 32, 40, 1, 2, 33, 34, 7, 8, 9, 1, 2 };

test "TF_GLM_OVN_COPY_AGREE off reproduces today's choice exactly" {
    var buf: [max_drafts]u32 = undefined;
    const off: Settings = .{ .match = 2, .most = 5, .reply_match = 0 };
    try std.testing.expectEqualSlices(u32, &.{ 33, 34, 7, 8, 9 }, off.propose(&agree_ctx, 0, 32, &buf));
    // agree 0 is off even with a cap set: the baseline (latest, start 9) is kept whole
    const lo_zero: Settings = .{ .match = 2, .most = 5, .reply_match = 0, .agree = 0, .agree_most = 7, .agree_longest = true };
    try std.testing.expectEqualSlices(u32, &.{ 33, 34, 7, 8, 9 }, lo_zero.propose(&agree_ctx, 0, 32, &buf));
    // a zero cap is also off (copy_drafts.py:125 gates agree on agree_most): no re-choice, no cap
    const cap_zero: Settings = .{ .match = 2, .most = 5, .reply_match = 0, .agree = 3, .agree_most = 0, .agree_longest = true };
    try std.testing.expectEqualSlices(u32, &.{ 33, 34, 7, 8, 9 }, cap_zero.propose(&agree_ctx, 0, 32, &buf));
}

test "TF_GLM_OVN_COPY_AGREE proposes the occurrence agreeing furthest back" {
    var buf: [max_drafts]u32 = undefined;
    const off: Settings = .{ .match = 2, .most = 5, .reply_match = 0 };
    try std.testing.expectEqualSlices(u32, &.{ 33, 34, 7, 8, 9 }, off.propose(&agree_ctx, 0, 32, &buf)); // latest, start 9
    // lo 3: start 3 agrees 3 (>= lo), so it is chosen and left uncapped; its drafts are 30 31 32 40 1
    const on: Settings = .{ .match = 2, .most = 5, .reply_match = 0, .agree = 3, .agree_most = 3, .agree_longest = true };
    try std.testing.expectEqualSlices(u32, &.{ 30, 31, 32, 40, 1 }, on.propose(&agree_ctx, 0, 32, &buf));
    // room 2 keeps the furthest-back choice (start 3) and trims the draft to the round's room
    try std.testing.expectEqualSlices(u32, &.{ 30, 31 }, on.propose(&agree_ctx, 0, 2, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 33, 34 }, off.propose(&agree_ctx, 0, 2, &buf));
}

test "TF_GLM_OVN_COPY_AGREE caps a weakly agreeing occurrence" {
    var buf: [max_drafts]u32 = undefined;
    // L=0 keeps the baseline occurrence (start 9, agrees 0 < 3) but caps its draft at agree_most 2
    const l0: Settings = .{ .match = 2, .most = 5, .reply_match = 0, .agree = 3, .agree_most = 2, .agree_longest = false };
    try std.testing.expectEqualSlices(u32, &.{ 33, 34 }, l0.propose(&agree_ctx, 0, 32, &buf));
    // longest among two weak occurrences (both under lo 5): start 3 wins and is still capped at 2
    const weak: Settings = .{ .match = 2, .most = 5, .reply_match = 0, .agree = 5, .agree_most = 2, .agree_longest = true };
    try std.testing.expectEqualSlices(u32, &.{ 30, 31 }, weak.propose(&agree_ctx, 0, 32, &buf));
}

test "TF_GLM_OVN_COPY_AGREE breaks agreement ties in favour of the latest" {
    var buf: [max_drafts]u32 = undefined;
    // (1 2) occurs at 2 and 8, each preceded by 9 (agreement 1); the tie goes to the later occurrence (start 8)
    const tie = [_]u32{ 9, 1, 2, 70, 71, 72, 73, 9, 1, 2, 80, 81, 9, 1, 2 };
    const on: Settings = .{ .match = 2, .most = 3, .reply_match = 0, .agree = 1, .agree_most = 1, .agree_longest = true };
    try std.testing.expectEqualSlices(u32, &.{ 80, 81, 9 }, on.propose(&tie, 0, 32, &buf));
}

test "TF_GLM_OVN_COPY_AGREE reproduces production's 16:7 longest choice" {
    var ctx: [130]u32 = undefined;
    var i: usize = 0;
    var x: u32 = 50;
    while (x < 100) : (x += 1) { // deep lead-in 50..99 (matches the final suffix's lead-in, so it agrees 50 back)
        ctx[i] = x;
        i += 1;
    }
    x = 100;
    while (x < 108) : (x += 1) { // the deep occurrence of the 8-token match 100..107
        ctx[i] = x;
        i += 1;
    }
    for ([_]u32{ 777, 778, 779, 900, 901 }) |t| {
        ctx[i] = t;
        i += 1;
    }
    x = 100;
    while (x < 108) : (x += 1) { // the near occurrence of the match, preceded by 901 (agrees 0 back)
        ctx[i] = x;
        i += 1;
    }
    ctx[i] = 888;
    i += 1;
    x = 50;
    while (x < 100) : (x += 1) { // the final suffix's lead-in 50..99
        ctx[i] = x;
        i += 1;
    }
    x = 100;
    while (x < 108) : (x += 1) { // the final suffix 100..107 (the query)
        ctx[i] = x;
        i += 1;
    }
    try std.testing.expectEqual(@as(usize, 130), i);
    var buf: [max_drafts]u32 = undefined;
    const off: Settings = .{ .match = 8, .most = 15, .reply_match = 16 };
    const on: Settings = .{ .match = 8, .most = 15, .reply_match = 16, .agree = 16, .agree_most = 7, .agree_longest = true };
    // off takes the latest occurrence (start 63); on takes the deep one (start 50, agreeing 16 or more back)
    try std.testing.expectEqualSlices(u32, &.{ 888, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 }, off.propose(&ctx, 130, 64, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 777, 778, 779, 900, 901, 100, 101, 102, 103, 104, 105, 106, 107, 888, 50 }, on.propose(&ctx, 130, 64, &buf));
}

test "TF_GLM_OVN_COPY_AGREE parsing matches production" {
    try std.testing.expect((try Settings.parseAgreeSpec("", 15)) == null);
    try std.testing.expect((try Settings.parseAgreeSpec("0", 15)) == null);
    // the production specs from test_copy_agree.py
    try std.testing.expectEqual(Settings.AgreeSpec{ .lo = 16, .k = 7, .longest = true }, (try Settings.parseAgreeSpec("16:7", 15)).?);
    try std.testing.expectEqual(Settings.AgreeSpec{ .lo = 16, .k = 7, .longest = false }, (try Settings.parseAgreeSpec("16:7:0", 15)).?);
    try std.testing.expectEqual(Settings.AgreeSpec{ .lo = 24, .k = 7, .longest = true }, (try Settings.parseAgreeSpec("24:7", 15)).?);
    try std.testing.expectEqual(Settings.AgreeSpec{ .lo = 4, .k = 3, .longest = true }, (try Settings.parseAgreeSpec("4:3", 15)).?);
    try std.testing.expectEqual(Settings.AgreeSpec{ .lo = 64, .k = 15, .longest = true }, (try Settings.parseAgreeSpec("64:15", 15)).?);
    try std.testing.expectEqual(Settings.AgreeSpec{ .lo = 16, .k = 1, .longest = true }, (try Settings.parseAgreeSpec("16:1", 15)).?);
    // the production refusals (len, decimal, LO, K and L bounds)
    for ([_][]const u8{ "16", "0:7", "65:7", "16:0", "16:16", "16:7:2", "a:b", "16:7:1:1", "-1:3", "00", "16:7:" }) |bad| {
        try std.testing.expectError(error.BadCopySettings, Settings.parseAgreeSpec(bad, 15));
    }
}

// Drive the GlmProposer round by round (test_copy_agree.drive): propose, keep the matching prefix, observe, advance.
fn driveMiss(s: Settings, prompt: []const u32, reply: []const u32, gpa: Allocator, props: *std.ArrayList(u32)) !void {
    var p: GlmProposer = .{ .settings = s, .prompt = prompt.len, .gpa = gpa };
    defer p.deinit();
    const pr = p.proposer();
    var pos: usize = 0;
    while (pos + 1 < reply.len) {
        const hist = try gpa.alloc(u32, prompt.len + pos);
        defer gpa.free(hist);
        @memcpy(hist[0..prompt.len], prompt);
        @memcpy(hist[prompt.len..], reply[0..pos]);
        const got = try pr.proposeAfter(hist, reply[pos..][0..1], 1 << 20);
        try props.appendSlice(gpa, got);
        var keep: usize = 1;
        for (got, 0..) |d, i| {
            if (pos + 1 + i >= reply.len or reply[pos + 1 + i] != d) break;
            keep += 1;
        }
        keep = @min(keep, reply.len - 1 - pos);
        pr.observe(@intCast(got.len), @intCast(keep - 1)); // settle the round: proposed against accepted drafts
        pos += keep;
    }
}

const miss_prompt = [_]u32{ 9, 1, 2, 7, 7, 3, 4, 4 };
const miss_reply = [_]u32{ 9, 1, 2, 5, 5, 5, 5, 9, 1, 2, 6, 7, 7, 3, 4, 4, 2, 2 };

test "TF_GLM_OVN_COPY_MISS_MAX narrows a copy round after a miss (proposals differ, replies do not)" {
    const gpa = std.testing.allocator;
    var props: std.ArrayList(u32) = .empty;
    defer props.deinit(gpa);
    // off: today's proposals over the reply
    try driveMiss(.{ .match = 2, .most = 5, .reply_match = 0, .miss_most = 0 }, &miss_prompt, &miss_reply, gpa, &props);
    try std.testing.expectEqualSlices(u32, &.{ 2, 7, 7, 3, 4, 5, 2, 5, 5, 5, 5, 3, 4, 4, 9, 1 }, props.items);
    // miss 2: a copy round after one that missed proposes at most 2, so the sequence differs
    props.clearRetainingCapacity();
    try driveMiss(.{ .match = 2, .most = 5, .reply_match = 0, .miss_most = 2 }, &miss_prompt, &miss_reply, gpa, &props);
    try std.testing.expectEqualSlices(u32, &.{ 2, 7, 7, 3, 4, 5, 2, 5, 5, 5, 5, 3, 4, 9, 1, 2, 5, 5 }, props.items);
}

test "TF_GLM_OVN_COPY_MISS_MAX composes with TF_GLM_OVN_COPY_AGREE" {
    const gpa = std.testing.allocator;
    var props: std.ArrayList(u32) = .empty;
    defer props.deinit(gpa);
    const ag: Settings = .{ .match = 2, .most = 5, .reply_match = 0, .agree = 3, .agree_most = 3, .agree_longest = true };
    // agree alone over the reply
    try driveMiss(ag, &miss_prompt, &miss_reply, gpa, &props);
    try std.testing.expectEqualSlices(u32, &.{ 2, 7, 7, 5, 2, 5, 5, 3, 4, 4 }, props.items);
    // agree plus miss 2: the miss cap (2) binds tighter than agree's cap (3) at the capped round
    props.clearRetainingCapacity();
    var am = ag;
    am.miss_most = 2;
    try driveMiss(am, &miss_prompt, &miss_reply, gpa, &props);
    try std.testing.expectEqualSlices(u32, &.{ 2, 7, 7, 5, 2, 5, 5, 3, 4, 9, 1, 2, 5, 5 }, props.items);
}

test "observe settles the miss flag; a non-copy round does not" {
    const gpa = std.testing.allocator;
    var p: GlmProposer = .{ .settings = .{ .match = 2, .most = 5, .miss_most = 2 }, .prompt = 0, .gpa = gpa };
    defer p.deinit();
    const pr = p.proposer();
    pr.observe(5, 3); // 3 of 5 drafts kept: missed
    try std.testing.expect(p.missed);
    pr.observe(5, 5); // all kept: cleared
    try std.testing.expect(!p.missed);
    pr.observe(5, 1); // missed again
    try std.testing.expect(p.missed);
    pr.observe(0, 0); // a non-copy round settles nothing: the flag persists
    try std.testing.expect(p.missed);
}

test "the miss cap applies only when missed and miss_most is set" {
    const gpa = std.testing.allocator;
    const hist = [_]u32{ 7, 8, 9, 1, 2, 30, 31, 32, 40, 1, 2, 33, 34, 7, 8, 9, 1 };
    var p: GlmProposer = .{ .settings = .{ .match = 2, .most = 5, .reply_match = 0, .miss_most = 2 }, .prompt = 0, .gpa = gpa };
    defer p.deinit();
    // not missed: the baseline proposal of 5 (latest occurrence, start 9)
    try std.testing.expectEqualSlices(u32, &.{ 33, 34, 7, 8, 9 }, try p.proposer().proposeAfter(&hist, &.{2}, 32));
    p.missed = true; // after a miss the same proposal caps at miss_most (2)
    try std.testing.expectEqualSlices(u32, &.{ 33, 34 }, try p.proposer().proposeAfter(&hist, &.{2}, 32));
    var q: GlmProposer = .{ .settings = .{ .match = 2, .most = 5, .reply_match = 0, .miss_most = 0 }, .prompt = 0, .gpa = gpa };
    defer q.deinit();
    q.missed = true; // miss_most 0 is off: a miss never caps
    try std.testing.expectEqualSlices(u32, &.{ 33, 34, 7, 8, 9 }, try q.proposer().proposeAfter(&hist, &.{2}, 32));
}
