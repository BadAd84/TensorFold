//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const images = @import("images.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
/// How far a cut probe render runs past where it leaves the prompt: enough for the next template token to fix its ids.
const stop_slack = 4096;

/// With image input, `ids` holds each image's tokens, `images` the images and `cache_key` the prompt the cache matches.
pub const Rendered = struct { ids: []const u32, history_len: usize = 0, rewind_len: usize = 0, user_end: usize = 0, images: []const api.Image = &.{}, cache_key: []const u32 = &.{} };

/// Where lessons decide to act on a reply: the row before the generation prompt's `<think>`, else before its last.
pub fn decideOf(srv: *const Server, ids: []const u32) u32 {
    if (srv.text.tokenId("<think>")) |open| {
        var i = ids.len;
        while (i > ids.len -| 8) {
            i -= 1;
            if (ids[i] == open and i > 0) return @intCast(i - 1);
        }
    }
    return @intCast(ids.len -| 2);
}

pub const isTitle = messages_mod.isTitleRequest;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    var problem: []const u8 = "";
    const r = try rendered(srv, cx, messages, tools, thinking, effort, generation, null, null);
    const ids = r.encode(srv.text, cx.a, &problem) catch |e| return failed(cx, e, problem);
    const close = try closer(srv, cx, ids, thinking, generation);
    return if (close) |c| try std.mem.concat(cx.a, u32, &.{ ids, &.{c} }) else ids;
}

/// The template's text for `messages` (call arguments through `args`); with `against`, cut a little past leaving it.
fn rendered(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, args: ?*messages_mod.Args, against: ?[]const u8) errors.Refused!model_text.Rendered {
    const normalized = try messages_mod.toolArgumentsWith(cx, try messages_mod.normalizeWith(cx, messages, srv.late_system, srv.needs_user_after_tool, .of(srv.info.vision)), args);
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
        .clear_thinking = cx.clear_thinking,
        .stop = if (against) |t| .{ .against = t, .slack = stop_slack } else null,
    };
    var r = srv.text.render(cx.a, normalized, options, &problem) catch |e| return failed(cx, e, problem);
    // with image input, a marker the text spells is ordinary text (images.zig)
    if (srv.info.vision) |v| if (v.video) |o| if (o.video_text.len > 0) {
        const word = try srv.text.tokenString(cx.a, o.video_token);
        r = try r.replaceTemplate(cx.a, o.video_text, word);
    };
    return r;
}

fn failed(cx: *Cx, e: model_text.Error, problem: []const u8) errors.Refused {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.fail(.request, "{s}", .{problem}),
    };
}

/// The ``</think>`` id appended when thinking is off and a generation prompt's last token opens a thought.
fn closer(srv: *Server, cx: *Cx, ids: []const u32, thinking: bool, generation: bool) errors.Refused!?u32 {
    if (thinking or !generation or ids.len == 0) return null;
    const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
    if (!std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) return null;
    const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => &[_]u32{},
    };
    return if (close.len == 1) close[0] else null;
}

/// A probe's ids, `renderIds`' for its messages: pieces whose kept parts are the prompt's own, and a closer.
const Probe = struct {
    pieces: model_text.Pieces,
    close: ?u32,

    fn parts(p: *const Probe) [4][]const u32 {
        return .{ p.pieces.head, p.pieces.mid, p.pieces.tail, if (p.close) |*c| @as(*const [1]u32, c) else &.{} };
    }

    fn len(p: *const Probe) usize {
        return p.pieces.head.len + p.pieces.mid.len + p.pieces.tail.len + @intFromBool(p.close != null);
    }

    /// ``std.mem.indexOfDiff(u32, base, ids)`` for this probe's ids, without joining them.
    fn diff(p: *const Probe, base: []const u32) ?usize {
        var at: usize = 0;
        for (p.parts()) |part| {
            const n = @min(part.len, base.len - at);
            // a kept part is the prompt's own memory: equal wherever it sits at the same place
            if (part.ptr != base.ptr + at) if (std.mem.indexOfDiff(u32, base[at..][0..n], part[0..n])) |i| return at + i;
            if (n < part.len) return base.len;
            at += n;
        }
        return if (at == base.len) null else at;
    }
};

/// One request's conversation rendered once: the prompt's ids kept, so each probe encodes only what it changes.
const Once = struct {
    srv: *Server,
    tools: []const Value,
    thinking: bool,
    effort: ?[]const u8,
    args: messages_mod.Args = .empty,
    kept: model_text.Kept = .{ .ids = &.{} },
    /// The prompt's text up to its generation block: the history render's text, when the template shows it is.
    history_text: ?model_text.Rendered = null,
    /// The prompt's rendered text, which a cut probe render is watched against.
    text: []const u8 = "",

    /// The prompt's ids (generation prompt on), kept for the probes.
    fn prompt(o: *Once, cx: *Cx, messages: Value) errors.Refused![]const u32 {
        var problem: []const u8 = "";
        const r = try rendered(o.srv, cx, messages, o.tools, o.thinking, o.effort, true, &o.args, null);
        o.kept = r.keep(o.srv.text, cx.a, &problem) catch |e| return failed(cx, e, problem);
        o.text = r.text;
        if (r.history) |h| o.history_text = try r.slice(cx.a, 0, h);
        const close = try closer(o.srv, cx, o.kept.ids, o.thinking, true);
        return if (close) |c| try std.mem.concat(cx.a, u32, &.{ o.kept.ids, &.{c} }) else o.kept.ids;
    }

    /// `renderIds`' ids for `messages`, with the segments the kept prompt repeats taken from it.
    fn probe(o: *Once, cx: *Cx, messages: Value, generation: bool) errors.Refused!Probe {
        return o.encoded(cx, try rendered(o.srv, cx, messages, o.tools, o.thinking, o.effort, generation, &o.args, null), generation);
    }

    /// `probe(messages, true).diff(ids)`; TF_GLM_OVN_HOST_ONCE: from a render cut once its first ids are decided.
    fn diffGenerated(o: *Once, cx: *Cx, messages: Value, ids: []const u32) errors.Refused!?usize {
        const early = o.srv.host_once and o.kept.state != null and o.srv.info.vision == null;
        const r = try rendered(o.srv, cx, messages, o.tools, o.thinking, o.effort, true, &o.args, if (early) o.text else null);
        if (!r.cut) return (try o.encoded(cx, r, true)).diff(ids);
        var problem: []const u8 = "";
        const fixed = r.prefixPieces(o.srv.text, cx.a, o.kept, &problem) catch |e| return failed(cx, e, problem);
        if (fixed) |p| {
            const lead: Probe = .{ .pieces = p, .close = null };
            // decided: a mismatch inside the fixed ids, or the prompt ending before them
            if (lead.diff(ids)) |i| if (i != lead.len()) return i;
        }
        return (try o.probe(cx, messages, true)).diff(ids);
    }

    /// The prompt's messages without the generation prompt: `probe(messages, false)`, cut from the prompt's text.
    fn history(o: *Once, cx: *Cx, messages: Value) errors.Refused!Probe {
        if (o.history_text) |h| return o.encoded(cx, h, false);
        return o.probe(cx, messages, false);
    }

    fn encoded(o: *Once, cx: *Cx, r: model_text.Rendered, generation: bool) errors.Refused!Probe {
        var problem: []const u8 = "";
        const p = r.pieces(o.srv.text, cx.a, o.kept, &problem) catch |e| return failed(cx, e, problem);
        const last = if (p.tail.len > 0) p.tail else if (p.mid.len > 0) p.mid else p.head;
        return .{ .pieces = p, .close = try closer(o.srv, cx, last, o.thinking, generation) };
    }
};

/// The template's generation prompt (thinking on or off): what a probe conversation renders past its history.
fn generationPrompt(srv: *Server, cx: *Cx, thinking: bool) errors.Refused![]const u32 {
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "x" });
    const list = try cx.a.alloc(Value, 1);
    list[0] = .{ .object = user };
    const with = try renderIds(srv, cx, .{ .array = list }, &.{}, thinking, null, true);
    const without = try renderIds(srv, cx, .{ .array = list }, &.{}, thinking, null, false);
    if (without.len >= with.len or !std.mem.eql(u32, with[0..without.len], without)) return &.{};
    return with[without.len..];
}

/// Earlier replies' decoded rows: from each turn's generation prompt end through its closing stop token.
pub fn replySpans(srv: *Server, cx: *Cx, prompt: []const u32) errors.Refused![]const [2]u32 {
    if (srv.eos.len == 0) return &.{};
    var probe: Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking }; // a template that refuses the probe leaves the request without spans, not failed
    const openers = [2][]const u32{ generationPrompt(srv, &probe, false) catch &.{}, generationPrompt(srv, &probe, true) catch &.{} };
    var spans: std.ArrayList([2]u32) = .empty;
    var i: usize = 0;
    while (i < prompt.len) {
        var start: ?usize = null;
        var at_min: usize = prompt.len;
        for (openers) |o| if (o.len > 0) if (std.mem.indexOfPos(u32, prompt, i, o)) |at| if (at < at_min) {
            at_min = at;
            start = at + o.len;
        };
        const first = start orelse break;
        const end = std.mem.indexOfAnyPos(u32, prompt, first, srv.eos) orelse break;
        try spans.append(cx.a, .{ @intCast(first), @intCast(end + 1) });
        i = end + 1;
    }
    return spans.items;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    var once: Once = .{ .srv = srv, .tools = input.tools, .thinking = thinking, .effort = effort };
    const prompt = try once.prompt(cx, input.messages);
    const history = try once.history(cx, input.messages);
    const at = history.diff(prompt);
    var history_len: usize = 0;
    if (history.len() > 0 and at == history.len()) history_len = history.len(); // a strict prefix of the prompt
    if (history_len == 0 and prompt.len > 1 and at == null) history_len = prompt.len - 1; // a template with no generation suffix
    // either way the history's ids are the prompt's first `history.len()`
    if (history_len > 0) history_len = continuedLen(&once, cx, input.messages, prompt[0..history.len()], history_len);
    // --vision: placeholders expanded to image tokens before the window, chunks and cache see it; no rewind kept
    if (try images.expand(srv.info.vision, if (srv.config.vision_urls) images.Fetch.network(srv.io) else null, srv.text, cx, input.messages, prompt, history_len)) |x| return .{ .ids = x.ids, .history_len = x.history_len, .images = x.images, .cache_key = x.cache_key };
    const count = if (input.messages == .array) input.messages.array.len else 0;
    const rewind_len = if (srv.info.prompt_cache) rewindWith(&once, cx, input.messages, count, prompt) else 0; // one more render
    return .{ .ids = prompt, .history_len = history_len, .rewind_len = rewind_len, .user_end = userEnd(input.messages, history_len) };
}

/// The history's prefix the next turn repeats (a template may close a tool reply's turn only when nothing follows).
fn continuedLen(once: *Once, cx: *Cx, messages: Value, history: []const u32, history_len: usize) usize {
    if (messages != .array or messages.array.len == 0) return history_len;
    const role = messages.array[messages.array.len - 1].strField("role") orelse return history_len;
    if (!std.mem.eql(u8, role, "tool")) return history_len;
    const list = cx.a.alloc(Value, messages.array.len + 1) catch return history_len;
    @memcpy(list[0..messages.array.len], messages.array);
    const reply = json.newObject(cx.a) catch return history_len;
    reply.put(cx.a, "role", .{ .string = "assistant" }) catch return history_len;
    reply.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return history_len;
    list[messages.array.len] = .{ .object = reply };
    var scratch: Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking };
    const probe = once.probe(&scratch, .{ .array = list }, false) catch return history_len;
    const i = probe.diff(history) orelse return history_len;
    return if (i > 0 and i < history_len) i else history_len;
}

/// The history ends the last user only when that user is the final input message. Other ends stay unknown.
fn userEnd(messages: Value, history_len: usize) usize {
    if (messages != .array or messages.array.len == 0) return 0;
    const role = messages.array[messages.array.len - 1].strField("role") orelse return 0;
    return if (std.mem.eql(u8, role, "user")) history_len else 0;
}

test "interior prompt: user end excludes later tool and assistant text" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "user", "tool", "assistant" }) |role| {
        const m = try json.newObject(a);
        defer a.destroy(m);
        defer m.deinit(a);
        try m.put(a, "role", .{ .string = role });
        var list = [_]Value{.{ .object = m }};
        try std.testing.expectEqual(@as(usize, if (std.mem.eql(u8, role, "user")) 131072 else 0), userEnd(.{ .array = &list }, 131072));
    }
    try std.testing.expectEqual(@as(usize, 0), userEnd(.null, 131072));
}

/// Where the latest user message's text starts (of the first `real_count`): the prompt and a probe there agree to it.
pub fn rewindLen(srv: *Server, cx: *Cx, messages: Value, real_count: usize, tools: []const Value, prompt: []const u32, thinking: bool, effort: ?[]const u8) usize {
    var once: Once = .{ .srv = srv, .tools = tools, .thinking = thinking, .effort = effort }; // nothing kept: the probe encodes whole
    return rewindWith(&once, cx, messages, real_count, prompt);
}

fn rewindWith(once: *Once, cx: *Cx, messages: Value, real_count: usize, prompt: []const u32) usize {
    if (messages != .array or real_count > messages.array.len) return 0;
    const user_index = for (0..real_count) |n| {
        const i = real_count - 1 - n;
        const role = messages.array[i].strField("role") orelse continue;
        if (std.mem.eql(u8, role, "user")) break i;
    } else return 0;
    if (messages.array[user_index] != .object) return 0;
    const list = cx.a.dupe(Value, messages.array) catch return 0;
    const user = json.copyObject(cx.a, messages.array[user_index].object) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    list[user_index] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking };
    const i = (once.diffGenerated(&scratch, .{ .array = list }, prompt) catch return 0) orelse return 0;
    return if (i < prompt.len) i else 0;
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}
