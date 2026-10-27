// ! GLM's prompt-lookup ("copy") drafts, ported from our glm-cuda family's `cuda/glm_copy.zig` (fork main 55d5636d.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");

/// The most drafts a round: a 16-row window (the GLM family's max_rows - 1).
pub const max_drafts = 15;

pub const Settings = struct {
    on: bool = false, // TF_GLM_COPY_DRAFTS (off unless 1; the production stack runs it on)
    match: usize = 8, // TF_GLM_COPY_MATCH
    most: usize = max_drafts, // TF_GLM_COPY_MAX (production: 15; Python's own default is 5)
    reply_match: usize = 16, // TF_GLM_COPY_REPLY_MATCH (production COPY_CODE=1: 16; 0: the prompt's rule everywhere)

    /// The production settings, from the TF_GLM_COPY_* variables.
    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        const on = std.c.getenv("TF_GLM_COPY_DRAFTS");
        if (on) |v| {
            const t = std.mem.trim(u8, std.mem.span(v), " ");
            if (t.len == 0 or std.mem.eql(u8, t, "0")) s.on = false else if (std.mem.eql(u8, t, "1")) s.on = true else {
                std.log.err("TF_GLM_COPY_DRAFTS is 0 or 1, not {s}", .{t});
                return error.BadCopySettings;
            }
        }
        s.match = try number("TF_GLM_COPY_MATCH", s.match, 2, 64);
        s.most = try number("TF_GLM_COPY_MAX", s.most, 1, max_drafts);
        s.reply_match = try number("TF_GLM_COPY_REPLY_MATCH", s.reply_match, 0, 64);
        if (s.reply_match != 0 and s.reply_match <= s.match) {
            std.log.err("TF_GLM_COPY_REPLY_MATCH is 0 (off) or more than TF_GLM_COPY_MATCH ({d}), not {d}", .{ s.match, s.reply_match });
            return error.BadCopySettings;
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

    /// Whether `ctx[i .. i + match]` is an earlier occurrence of the context's last `match` tokens (`prompt` leading.
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

    /// CopyDrafts.propose: up to `min(most, room)` drafts.
    pub fn propose(s: Settings, ctx: []const u32, prompt: usize, room: usize, buf: *[max_drafts]u32) []const u32 {
        const k = @min(s.most, room);
        const n = s.match;
        const len = ctx.len;
        if (!s.on or k < 1 or len <= n) return buf[0..0];
        const last = ctx[len - 1];
        var found: ?usize = null;
        if (len >= n + k) { // the latest start with k tokens after its match
            var i = len - n - k + 1;
            while (i > 0) {
                i -= 1;
                if (ctx[i + n - 1] == last and s.occurs(ctx, prompt, i)) {
                    found = i;
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
        const at = (found orelse return buf[0..0]) + n;
        const count = @min(k, len - at);
        @memcpy(buf[0..count], ctx[at..][0..count]);
        return buf[0..count];
    }
};

/// One request's proposer: the family's rules over the stream's own context, behind lanes.proposer.Proposer..
pub const GlmProposer = struct {
    settings: Settings,
    prompt: usize, // the request's prompt tokens, the reply boundary `reply_match` needs
    buf: [max_drafts]u32 = undefined,
    out: std.ArrayList(u32) = .empty,
    gpa: Allocator,
    matched: i64 = 0, // the last proposal's match length (0: none)

    pub fn deinit(p: *GlmProposer) void {
        p.out.deinit(p.gpa);
    }

    fn proposeTokens(p: *GlmProposer, history: []const u32, tail: []const u32, max_draft: i64) ![]const u32 {
        const room: usize = if (max_draft < 0) 0 else @intCast(@min(max_draft, max_drafts));
        const n = p.settings.match;
        if (n == 0 or n > history.len + tail.len or tail.len >= n) return error.BadCopyContext;
        // the context the rules see: the history with the pending token (and any forced tail) joined on.
        const ctx = try p.gpa.alloc(u32, history.len + tail.len);
        defer p.gpa.free(ctx);
        @memcpy(ctx[0..history.len], history);
        @memcpy(ctx[history.len..], tail);
        const got = p.settings.propose(ctx, p.prompt, room, &p.buf);
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

    pub const vtable: lanes.proposer.Proposer.VTable = .{
        .propose = proposeFn,
        .propose_after = afterFn,
        .last_match = lastMatchFn,
        .priced = true, // the rules set their own length: a one-token copy is a copy, no core gate
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

/// The factory `cuda_native.copyProposers` hands `native/cuda.zig`'s Host.serve..
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
    const s: Settings = .{ .match = 2, .most = 3, .reply_match = 0 };
    var buf: [max_drafts]u32 = undefined;
    // the suffix (1 2) occurred at 0 and 5, both with 3 tokens after them: the later one
    const ctx = [_]u32{ 1, 2, 5, 6, 7, 1, 2, 8, 9, 1, 2 };
    try std.testing.expectEqualSlices(u32, &.{ 8, 9, 1 }, s.propose(&ctx, 0, 10, &buf));
    // room 1 is one draft, from the same occurrence
    try std.testing.expectEqualSlices(u32, &.{8}, s.propose(&ctx, 0, 1, &buf));
    // (3 4) occurred once, at 0, with only 3 tokens after it where 4 are wanted: the earliest start, short
    const wide: Settings = .{ .match = 2, .most = 4, .reply_match = 0 };
    const short = [_]u32{ 3, 4, 7, 3, 4 };
    try std.testing.expectEqualSlices(u32, &.{ 7, 3, 4 }, wide.propose(&short, 0, 10, &buf));
    // never seen
    const fresh = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 0), s.propose(&fresh, 0, 10, &buf).len);
}

test "an occurrence inside the reply needs the reply's longer match" {
    const s: Settings = .{ .match = 2, .most = 2, .reply_match = 4 };
    var buf: [max_drafts]u32 = undefined;
    // prompt 9 9, reply 1 2 5 7 1 2: the occurrence of (1 2) at 2 starts in the reply.
    const ctx = [_]u32{ 9, 9, 1, 2, 5, 7, 1, 2 };
    try std.testing.expectEqual(@as(usize, 0), s.propose(&ctx, 2, 10, &buf).len);
    // with the tokens before it equal, it copies
    const same = [_]u32{ 9, 9, 6, 6, 1, 2, 5, 6, 6, 1, 2 };
    try std.testing.expectEqualSlices(u32, &.{ 5, 6 }, s.propose(&same, 2, 10, &buf));
    // an occurrence in the prompt keeps the short match
    try std.testing.expectEqualSlices(u32, &.{ 5, 7 }, s.propose(&.{ 1, 2, 5, 7, 4, 1, 2 }, 6, 10, &buf));
}

test "a window of 9 to 15 rows is padded to the graph width when room allows" {
    const s: Settings = .{ .match = 2, .most = 15, .reply_match = 0 };
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
    var p: GlmProposer = .{ .settings = .{ .match = 2, .most = 3, .reply_match = 0 }, .prompt = 2, .gpa = std.testing.allocator };
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
