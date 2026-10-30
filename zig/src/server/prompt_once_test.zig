//! prepare renders once and reuses the prompt's ids in its probes; on random chats it must equal every full render.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const errors = @import("errors.zig");
const prompt_mod = @import("prompt.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 20, .prompt_cache = true };
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

// GLM-5.3's other template words as added tokens of the mini byte-level BPE, so each splits a segment as it does there
const extra = [_][]const u8{ "<|observation|>", "<tool_response>", "</tool_response>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tools>", "</tools>" };

fn tokenizer(a: Allocator) ![]const u8 {
    var parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"), .{});
    const list = &parsed.object.getPtr("added_tokens").?.array;
    for (extra, 0..) |word, i| {
        var o: std.json.ObjectMap = .empty;
        try o.put(a, "id", .{ .integer = @intCast(1100 + i) });
        try o.put(a, "content", .{ .string = word });
        try o.put(a, "special", .{ .bool = true });
        try o.put(a, "normalized", .{ .bool = false });
        try list.append(.{ .object = o });
    }
    return std.json.Stringify.valueAlloc(a, parsed, .{});
}

// request texts, the masked route among them: the template's own words, the probe's mark, long memo-sized runs
const texts = [_][]const u8{ "", "ok", "  ", "Read the code.", "<|user|>", "<|assistant|><think>", "</think>done", "<think>plan</think>answer", "<|observation|><tool_response>x</tool_response>", "\u{2063}probe", "caf\u{e9} \u{1f600}", "<arg_key>k</arg_key>" };

fn content(a: Allocator, r: std.Random) ![]const u8 {
    if (r.uintLessThan(u8, 6) > 0) return texts[r.uintLessThan(usize, texts.len)];
    var long: std.ArrayList(u8) = .empty;
    for (0..1100 + r.uintLessThan(usize, 600)) |i| try long.append(a, "abc <|user|> \n"[(i * 7 + r.uintLessThan(usize, 3)) % 14]);
    return long.items;
}

fn message(a: Allocator, r: std.Random, role: []const u8, n: usize) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = try content(a, r) });
    if (std.mem.eql(u8, role, "assistant")) {
        if (r.boolean()) try o.put(a, "reasoning_content", .{ .string = texts[r.uintLessThan(usize, texts.len)] });
        if (r.boolean()) {
            const args = [_][]const u8{ "{\"path\": \"a.txt\", \"n\": 3}", "", "{not json", "[1]" };
            const f = try json.newObject(a);
            try f.put(a, "name", .{ .string = "read_file" });
            try f.put(a, "arguments", .{ .string = args[r.uintLessThan(usize, args.len)] });
            const call = try json.newObject(a);
            try call.put(a, "id", .{ .string = try std.fmt.allocPrint(a, "c{d}", .{n}) });
            try call.put(a, "type", .{ .string = "function" });
            try call.put(a, "function", .{ .object = f });
            try o.put(a, "tool_calls", .{ .array = try a.dupe(Value, &.{.{ .object = call }}) });
        }
    }
    if (std.mem.eql(u8, role, "tool")) try o.put(a, "tool_call_id", .{ .string = try std.fmt.allocPrint(a, "c{d}", .{n -| 1}) });
    return .{ .object = o };
}

const Marks = struct { ids: []const u32, history_len: usize, rewind_len: usize, user_end: usize };

/// prepare as it was before rendering once: every probe a full renderIds.
fn reference(srv: *Server, cx: *errors.Cx, messages: []const Value, tools: []const Value, thinking: bool) errors.Refused!Marks {
    const list: Value = .{ .array = @constCast(messages) };
    const ids = try prompt_mod.renderIds(srv, cx, list, tools, thinking, null, true);
    const history = try prompt_mod.renderIds(srv, cx, list, tools, thinking, null, false);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < ids.len and std.mem.eql(u32, ids[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and ids.len > 1 and std.mem.eql(u32, history, ids)) history_len = ids.len - 1;
    const last = messages[messages.len - 1].strField("role").?;
    if (history_len > 0 and std.mem.eql(u8, last, "tool")) {
        const probe = try cx.a.alloc(Value, messages.len + 1);
        @memcpy(probe[0..messages.len], messages);
        const reply = try json.newObject(cx.a);
        try reply.put(cx.a, "role", .{ .string = "assistant" });
        try reply.put(cx.a, "content", .{ .string = "\u{2063}probe" });
        probe[messages.len] = .{ .object = reply };
        var scratch: errors.Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking };
        if (prompt_mod.renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, null, false)) |p| {
            if (std.mem.indexOfDiff(u32, history, p)) |i| if (i > 0 and i < history_len) {
                history_len = i;
            };
        } else |_| {}
    }
    // the old rewindLen: the last user's text replaced by the probe mark, the whole render compared
    var rewind: usize = 0;
    const at = for (0..messages.len) |n| {
        if (std.mem.eql(u8, messages[messages.len - 1 - n].strField("role").?, "user")) break messages.len - 1 - n;
    } else null;
    if (at) |u| {
        const list2 = try cx.a.dupe(Value, messages);
        const user = try json.copyObject(cx.a, messages[u].object);
        try user.put(cx.a, "content", .{ .string = "\u{2063}probe" });
        list2[u] = .{ .object = user };
        var scratch: errors.Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking };
        if (prompt_mod.renderIds(srv, &scratch, .{ .array = list2 }, tools, thinking, null, true)) |p| {
            if (std.mem.indexOfDiff(u32, ids, p)) |i| rewind = if (i < ids.len) i else 0;
        } else |_| {}
    }
    return .{ .ids = ids, .history_len = history_len, .rewind_len = rewind, .user_end = if (std.mem.eql(u8, last, "user")) history_len else 0 };
}

fn check(source: []const u8, split: bool, host_once: bool, rounds: usize, seed: u64) !usize {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = try tokenizer(a) });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = source });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, try tmp.dir.realPathFileAlloc(std.testing.io, ".", a), null, a, &problem);
    defer hf.deinit();
    try std.testing.expectEqual(split, hf.compiled.?.split != null);
    var script: Script = .{};
    const srv = try Server.init(a, std.testing.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    srv.host_once = host_once;
    const spec = (try json.parse(a, "{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"type\":\"object\"}}}")).ok;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const roles = [_][]const u8{ "user", "assistant", "tool", "tool", "system" };
    var compared: usize = 0;
    for (0..rounds) |_| {
        var list: std.ArrayList(Value) = .empty;
        if (r.boolean()) try list.append(a, try message(a, r, "system", 0));
        for (0..1 + r.uintLessThan(usize, 7)) |n| try list.append(a, try message(a, r, roles[r.uintLessThan(usize, roles.len)], n));
        const tools: []const Value = if (r.boolean()) &.{spec} else &.{};
        const thinking = r.boolean();
        const clear: ?bool = switch (r.uintLessThan(u8, 3)) {
            0 => null,
            1 => true,
            else => false,
        };
        var cx: errors.Cx = .{ .a = a, .clear_thinking = clear };
        const want = reference(srv, &cx, list.items, tools, thinking);
        var cx2: errors.Cx = .{ .a = a, .clear_thinking = clear };
        const got = prompt_mod.prepare(srv, &cx2, .{ .messages = .{ .array = list.items }, .tools = tools, .fields = .null }, thinking, null);
        if (want) |w| {
            const g = try got;
            try std.testing.expectEqualSlices(u32, w.ids, g.ids);
            try std.testing.expectEqual(w.history_len, g.history_len);
            try std.testing.expectEqual(w.rewind_len, g.rewind_len);
            try std.testing.expectEqual(w.user_end, g.user_end);
            compared += 1;
        } else |err| {
            try std.testing.expectError(err, got);
            try std.testing.expectEqualStrings(cx.message, cx2.message);
        }
    }
    return compared;
}

test "prompt once: GLM-5.3's template, the history cut from the prompt's render, equals every full render" {
    const source = @embedFile("fixtures/glm53/chat_template.jinja");
    try std.testing.expect(try check(source, true, false, 400, 0x676c6d) > 300);
}

test "prompt once: a template whose history must be rendered on its own equals every full render" {
    // text after the generation block: no mark, so the history is a probe render with reused pieces
    const source = @embedFile("fixtures/glm53/chat_template.jinja") ++ "{{ ' ' }}";
    try std.testing.expect(try check(source, false, false, 200, 0x6e6f6d) > 150);
}

test "prompt once: TF_GLM_OVN_HOST_ONCE's rewind probe, cut past where it leaves the prompt, equals every full render" {
    try std.testing.expect(try check(@embedFile("fixtures/glm53/chat_template.jinja"), true, true, 400, 0x637574) > 300);
}
