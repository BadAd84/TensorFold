//! ``prompt_bench MODEL PROMPTS [rounds]``: the server's request preparation on the parity probe's r2, ms per site.
const std = @import("std");
const server = @import("server");
const api = @import("engine_api");
const json = server.json;
const Value = json.Value;
const prompt = server.prompt;
const chat = server.chat;
const Allocator = std.mem.Allocator;

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 21, .prompt_cache = true };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Gone = struct {
    pub fn check(_: Gone) bool {
        return false;
    }
};

fn object(a: Allocator, pairs: []const struct { []const u8, Value }) !Value {
    const o = try json.newObject(a);
    for (pairs) |p| try o.put(a, p[0], p[1]);
    return .{ .object = o };
}

fn message(a: Allocator, role: []const u8, content: []const u8) !Value {
    return object(a, &.{ .{ "role", .{ .string = role } }, .{ "content", .{ .string = content } } });
}

fn read(io: std.Io, a: Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, name }), a, .limited(16 << 20));
}

/// A chat body as a client sends it: the messages, optional tools, thinking off through chat_template_kwargs.
fn body(a: Allocator, messages: []const Value, tools: ?Value) ![]const u8 {
    const kwargs = try object(a, &.{.{ "enable_thinking", .{ .bool = false } }});
    const b = try object(a, &.{ .{ "model", .{ .string = "m" } }, .{ "messages", .{ .array = try a.dupe(Value, messages) } }, .{ "chat_template_kwargs", kwargs }, .{ "max_tokens", .{ .int = "64" } }, .{ "stream", .{ .bool = true } } });
    if (tools) |t| try b.object.put(a, "tools", t);
    return json.stringify(a, b, .{ .ascii = false });
}

/// An agent's conversation over `corpus`: one read_file call and its result per 4 KiB, ending on a tool result.
fn agent(a: Allocator, corpus: []const u8, bytes: usize) !struct { prior: []const u8, now: []const u8 } {
    const spec = try object(a, &.{ .{ "type", .{ .string = "function" } }, .{ "function", try object(a, &.{ .{ "name", .{ .string = "read_file" } }, .{ "description", .{ .string = "Read part of a file." } }, .{ "parameters", (try json.parse(a, "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":[\"path\"]}")).ok } }) } });
    const tools: Value = .{ .array = try a.dupe(Value, &.{spec}) };
    var list: std.ArrayList(Value) = .empty;
    try list.append(a, try message(a, "system", "You are a careful assistant. Answer briefly."));
    try list.append(a, try message(a, "user", "Read the files and report the three slowest batches."));
    var at: usize = 0;
    var n: usize = 0;
    var prior_len: usize = 0;
    while (at < bytes) : (n += 1) {
        const chunk = corpus[at % corpus.len ..][0..@min(4096, corpus.len - at % corpus.len)];
        at += chunk.len;
        const id = try std.fmt.allocPrint(a, "call_{d}", .{n});
        const args = try std.fmt.allocPrint(a, "{{\"path\": \"src/part_{d}.txt\", \"offset\": {d}}}", .{ n, at });
        const call = try object(a, &.{ .{ "id", .{ .string = id } }, .{ "type", .{ .string = "function" } }, .{ "function", try object(a, &.{ .{ "name", .{ .string = "read_file" } }, .{ "arguments", .{ .string = args } } }) } });
        const turn = try object(a, &.{ .{ "role", .{ .string = "assistant" } }, .{ "content", .{ .string = try std.fmt.allocPrint(a, "Reading part {d}.", .{n}) } }, .{ "tool_calls", .{ .array = try a.dupe(Value, &.{call}) } } });
        prior_len = list.items.len;
        try list.append(a, turn);
        try list.append(a, try object(a, &.{ .{ "role", .{ .string = "tool" } }, .{ "tool_call_id", .{ .string = id } }, .{ "content", .{ .string = try std.mem.concat(a, u8, &.{ try std.fmt.allocPrint(a, "[part {d}]\n", .{n}), chunk }) } } }));
    }
    return .{ .prior = try body(a, list.items[0..prior_len], tools), .now = try body(a, list.items, tools) };
}

const sites = [_][]const u8{
    "parse body",                     "plan (openai.chatInput)",        "chat.prepare",                 "  prompt.prepare",
    "  systemPrefixLen (chat 154)",   "  replySpans (chat 177)",        "  render prompt",              "  keep prompt",
    "  render history probe",         "  pieces history probe",         "  render continued probe",     "  pieces continued probe",
    "  render rewind probe",          "  pieces rewind probe",          "old: renderIds prompt",        "old: renderIds history",
    "old: renderIds continued probe", "old: rewindLen (full encode)",
};

/// A list with message `at` given the probe text, or a probe message of `role` appended when `at` is the end.
fn probeList(a: Allocator, messages: []const Value, at: usize, role: []const u8) ![]Value {
    const list = try a.alloc(Value, @max(messages.len, at + 1));
    @memcpy(list[0..messages.len], messages);
    const m = if (at < messages.len) try json.copyObject(a, messages[at].object) else try json.newObject(a);
    if (at == messages.len) try m.put(a, "role", .{ .string = role });
    try m.put(a, "content", .{ .string = "\u{2063}probe" });
    list[at] = .{ .object = m };
    return list;
}

/// The template's text for `messages`, as prompt.zig renders it for this bench's requests (thinking off).
fn render(srv: *server.Server, cx: *server.errors.Cx, tools: []const Value, messages: []const Value, generation: bool, args: *server.messages.Args) !server.model_text.Rendered {
    var problem: []const u8 = "";
    const normalized = try server.messages.toolArgumentsWith(cx, try server.messages.normalizeWith(cx, .{ .array = @constCast(messages) }, srv.late_system, srv.needs_user_after_tool, .of(srv.info.vision)), args);
    const t: ?Value = if (tools.len > 0) Value{ .array = @constCast(tools) } else null;
    return srv.text.render(cx.a, normalized, .{ .tools = t, .add_generation_prompt = generation }, &problem);
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const Out = struct { ids: usize, digest: u64, history: u32, rewind: u32, user_end: u32 };

fn load(gpa: Allocator, io: std.Io, dir: []const u8, script: *Script) !struct { hf: *server.hf_text.HfText, srv: *server.Server } {
    var problem: []const u8 = "";
    const hf = server.hf_text.HfText.load(gpa, io, dir, null, gpa, &problem) catch |e| {
        std.debug.print("load: {s}\n", .{problem});
        return e;
    };
    const srv = try server.Server.init(gpa, io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    return .{ .hf = hf, .srv = srv };
}

/// One request from its body bytes to a prepared request, as the chat route runs it.
fn prepareBody(srv: *server.Server, cx: *server.errors.Cx, bytes: []const u8) !chat.Prepared {
    const raw = (try json.parse(cx.a, bytes)).ok;
    const input = try server.openai.chatInput(srv, cx, raw);
    return chat.prepare(srv, cx, input, Gone{});
}

fn bench(gpa: Allocator, io: std.Io, dir: []const u8, name: []const u8, prior: []const u8, bytes: []const u8, rounds: usize) !void {
    var script: Script = .{};
    var cold_ns: i96 = 0;
    {
        // a fresh boot: nothing in the tokenizer's memo
        const x = try load(gpa, io, dir, &script);
        defer x.hf.deinit();
        defer x.srv.deinit();
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var cx: server.errors.Cx = .{ .a = arena.allocator() };
        const t = now(io);
        const p = try prepareBody(x.srv, &cx, bytes);
        cold_ns = now(io) - t;
        x.srv.release(p.preparing, p.request);
    }
    const x = try load(gpa, io, dir, &script);
    defer x.hf.deinit();
    defer x.srv.deinit();
    const srv = x.srv;
    {
        // the previous turn, as the client sent it before this one
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var cx: server.errors.Cx = .{ .a = arena.allocator() };
        const p = try prepareBody(srv, &cx, prior);
        srv.release(p.preparing, p.request);
    }
    const times = try gpa.alloc(i96, sites.len * rounds);
    defer gpa.free(times);
    var out: Out = undefined;
    var problem: []const u8 = "";
    for (0..rounds) |r| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var cx: server.errors.Cx = .{ .a = arena.allocator() };
        const a = cx.a;
        var at = now(io);
        const lap = struct {
            fn lap(i: std.Io, from: *i96) i96 {
                const t = now(i);
                defer from.* = t;
                return t - from.*;
            }
        }.lap;
        const row = times[r * sites.len ..][0..sites.len];
        @memset(row, 0);
        const raw = (try json.parse(a, bytes)).ok;
        row[0] = lap(io, &at);
        const input = try server.openai.chatInput(srv, &cx, raw);
        row[1] = lap(io, &at);
        const p = try chat.prepare(srv, &cx, input, Gone{});
        row[2] = lap(io, &at);
        srv.release(p.preparing, p.request);
        const effort = srv.effortFor(null);
        const prepared = try prompt.prepare(srv, &cx, input, false, effort);
        row[3] = lap(io, &at);
        _ = prompt.systemPrefixLen(srv, &cx, input.messages, input.tools, prepared.ids, false, effort);
        row[4] = lap(io, &at);
        _ = try prompt.replySpans(srv, &cx, prepared.ids);
        row[5] = lap(io, &at);
        // the new path's steps with the public pieces, as prompt.zig's Once takes them
        const msgs = input.messages.array;
        const last_tool = std.mem.eql(u8, msgs[msgs.len - 1].strField("role").?, "tool");
        const user_at = for (0..msgs.len) |n| {
            if (std.mem.eql(u8, msgs[msgs.len - 1 - n].strField("role").?, "user")) break msgs.len - 1 - n;
        } else unreachable;
        var args: server.messages.Args = .empty;
        const pr = try render(srv, &cx, input.tools, msgs, true, &args);
        row[6] = lap(io, &at);
        const k = try pr.keep(srv.text, a, &problem);
        row[7] = lap(io, &at);
        const hr = try render(srv, &cx, input.tools, msgs, false, &args);
        row[8] = lap(io, &at);
        _ = try hr.pieces(srv.text, a, k, &problem);
        row[9] = lap(io, &at);
        if (last_tool) {
            const cr = try render(srv, &cx, input.tools, try probeList(a, msgs, msgs.len, "assistant"), false, &args);
            row[10] = lap(io, &at);
            _ = try cr.pieces(srv.text, a, k, &problem);
            row[11] = lap(io, &at);
        }
        const rr = try render(srv, &cx, input.tools, try probeList(a, msgs, user_at, "user"), true, &args);
        row[12] = lap(io, &at);
        _ = try rr.pieces(srv.text, a, k, &problem);
        row[13] = lap(io, &at);
        // the old path, every probe encoded whole: the reference the request must equal
        const ids = try prompt.renderIds(srv, &cx, input.messages, input.tools, false, effort, true);
        row[14] = lap(io, &at);
        const history = try prompt.renderIds(srv, &cx, input.messages, input.tools, false, effort, false);
        row[15] = lap(io, &at);
        var history_len: usize = 0;
        if (history.len > 0 and history.len < ids.len and std.mem.eql(u32, ids[0..history.len], history)) history_len = history.len;
        if (history_len == 0 and ids.len > 1 and std.mem.eql(u32, history, ids)) history_len = ids.len - 1;
        if (history_len > 0 and last_tool) {
            var scratch: server.errors.Cx = .{ .a = a };
            if (prompt.renderIds(srv, &scratch, .{ .array = try probeList(a, msgs, msgs.len, "assistant") }, input.tools, false, effort, false)) |probe| {
                if (std.mem.indexOfDiff(u32, history, probe)) |i| history_len = if (i > 0 and i < history_len) i else history_len;
            } else |_| {}
        }
        row[16] = lap(io, &at);
        const rewind = prompt.rewindLen(srv, &cx, input.messages, msgs.len, input.tools, ids, false, effort);
        row[17] = lap(io, &at);
        const user_end = if (std.mem.eql(u8, msgs[msgs.len - 1].strField("role").?, "user")) history_len else 0;
        if (!std.mem.eql(u32, ids, p.request.prompt) or !std.mem.eql(u32, ids, prepared.ids)) return error.IdsDiffer;
        if (history_len != p.request.history_len or rewind != p.request.rewind_len or user_end != p.request.user_end) {
            std.debug.print("reference history {d} rewind {d} user_end {d}\n", .{ history_len, rewind, user_end });
            return error.MarksDiffer;
        }
        out = .{ .ids = ids.len, .digest = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids)), .history = p.request.history_len, .rewind = p.request.rewind_len, .user_end = p.request.user_end };
    }
    std.debug.print("\n{s}: {d} prompt ids, digest {x:0>16}, history {d}, rewind {d}, user_end {d}\n", .{ name, out.ids, out.digest, out.history, out.rewind, out.user_end });
    std.debug.print("{s:<34} {d:>9.2}\n{s:<34} {d:>9.2}\n", .{ "cold boot: body to chat.prepare", ms(cold_ns), "first warm turn: chat.prepare", ms(times[2]) });
    std.debug.print("{s:<34} {s:>9} {s:>9} ({d} rounds)\n", .{ "site", "median", "min", rounds });
    const column = try gpa.alloc(i96, rounds);
    defer gpa.free(column);
    for (sites, 0..) |site, i| {
        for (column, 0..) |*c, r| c.* = times[r * sites.len + i];
        std.mem.sort(i96, column, {}, std.sort.asc(i96));
        std.debug.print("{s:<34} {d:>9.2} {d:>9.2}\n", .{ site, ms(column[rounds / 2]), ms(column[0]) });
    }
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len < 3) {
        std.debug.print("usage: prompt_bench MODEL PROMPTS [rounds]\n", .{});
        return 2;
    }
    const rounds: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 7;
    server.log.init(io, false);
    const c0 = try read(io, a, argv[2], "c0.txt");
    const reply = try read(io, a, argv[2], "reply.txt");
    const tool = try read(io, a, argv[2], "tool_plain.txt");
    // the probe's r0 and r2, byte for byte (scripts/parity/perpatch_measure_ai.py R0, R2)
    const r0 = [_]Value{ try message(a, "system", "You are a careful assistant. Answer briefly."), try message(a, "user", try std.mem.concat(a, u8, &.{ c0, "\n\nRead the code above." })) };
    const r2 = r0 ++ [_]Value{ try message(a, "assistant", try std.mem.concat(a, u8, &.{ "Reply. ", reply })), try message(a, "user", try std.mem.concat(a, u8, &.{ tool, "\nTurn: name the three slowest batches." })) };
    try bench(gpa, io, argv[1], "r2", try body(a, &r0, null), try body(a, &r2, null), rounds);
    // r2m: the template's own tokens quoted in the turn (the masked route)
    const quoted = try read(io, a, argv[2], "tool_masked.txt");
    const r2m = r0 ++ [_]Value{ try message(a, "assistant", try std.mem.concat(a, u8, &.{ "Reply m. ", reply })), try message(a, "user", try std.mem.concat(a, u8, &.{ quoted, "\nTurn: name the three slowest batches." })) };
    try bench(gpa, io, argv[1], "r2m (masked)", try body(a, &r0, null), try body(a, &r2m, null), rounds);
    const corpus = try std.mem.concat(a, u8, &.{ c0, try read(io, a, argv[2], "c32.txt"), try read(io, a, argv[2], "fix40.txt") });
    const small = try agent(a, corpus, c0.len);
    try bench(gpa, io, argv[1], "agent ~200k, ends on a tool result", small.prior, small.now, rounds);
    var big_corpus: std.ArrayList(u8) = .empty;
    for (0..3) |k| for (0..corpus.len / 4096 + 1) |i| {
        try big_corpus.print(a, "<{d}.{d}>", .{ k, i });
        try big_corpus.appendSlice(a, corpus[i * 4096 .. @min(corpus.len, (i + 1) * 4096)]);
    };
    const big = try agent(a, big_corpus.items, big_corpus.items.len * 5 / 6);
    try bench(gpa, io, argv[1], "agent ~650k, ends on a tool result", big.prior, big.now, rounds);
    return 0;
}
