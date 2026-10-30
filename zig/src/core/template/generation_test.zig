//! A final generation block's mark: the flag-off render is the flag-on text up to it, provenance included.
const std = @import("std");
const template = @import("template.zig");
const t = std.testing;

const glm_like =
    \\[gMASK]<sop>{%- for m in messages -%}{%- if m.role == 'user' -%}<|user|>{{ m.content }}
    \\{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}{% if loop.last %}<|end|>{% endif %}
    \\{%- else -%}<|{{ m.role }}|>{{ m.content }}{%- endif -%}{%- endfor -%}
    \\{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}
    \\
;

fn render(a: std.mem.Allocator, compiled: *const template.Compiled, messages: std.json.Value, generation: bool, history: *?usize) !template.Str {
    var more: template.More = .{};
    defer history.* = more.history;
    return template.renderMore(a, compiled, messages, .null, .null, .null, generation, null, &more);
}

fn chat(a: std.mem.Allocator, r: std.Random) !std.json.Value {
    const roles = [_][]const u8{ "user", "assistant", "tool", "system" };
    const texts = [_][]const u8{ "", "x", "<|user|>", "a\nb", " ", "<think>" };
    var list = std.json.Array.init(a);
    for (0..r.uintLessThan(usize, 6)) |_| {
        var m: std.json.ObjectMap = .empty;
        try m.put(a, "role", .{ .string = roles[r.uintLessThan(usize, roles.len)] });
        try m.put(a, "content", .{ .string = texts[r.uintLessThan(usize, texts.len)] });
        try list.append(.{ .object = m });
    }
    return .{ .array = list };
}

test "generation split: the flag-off text and spans are the flag-on render's up to the mark" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var compiled = try template.compile(a, glm_like, null);
    defer compiled.deinit();
    try t.expect(compiled.split != null);
    var prng = std.Random.DefaultPrng.init(0x67656e);
    const r = prng.random();
    for (0..200) |_| {
        const messages = try chat(a, r);
        var mark: ?usize = null;
        const on = try render(a, &compiled, messages, true, &mark);
        var none: ?usize = null;
        const off = try render(a, &compiled, messages, false, &none);
        try t.expectEqual(@as(?usize, null), none);
        const h = mark.?;
        try t.expectEqualStrings(off.s, on.s[0..h]);
        var kept: usize = 0;
        while (kept < on.spans.len and on.spans[kept][0] < h) kept += 1;
        try t.expectEqualSlices(template.Span, off.spans, on.spans[0..kept]);
        for (on.spans[0..kept]) |p| try t.expect(p[1] <= h);
    }
}

test "generation split: any other read of the flag, an else, or output after the block keeps every render" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "{% if add_generation_prompt %}x{% endif %}{{ messages|length }}",
        "{{ add_generation_prompt }}{% if add_generation_prompt %}x{% endif %}",
        "{% if add_generation_prompt %}x{% else %}y{% endif %}",
        "{% if add_generation_prompt %}x{% elif true %}y{% endif %}",
        "{% set g = add_generation_prompt %}{% if g %}x{% endif %}",
        "{% if add_generation_prompt and true %}x{% endif %}",
        "{% macro f(add_generation_prompt) %}{% endmacro %}{% if add_generation_prompt %}x{% endif %}",
        "{% for add_generation_prompt in [1] %}{% endfor %}{% if add_generation_prompt %}x{% endif %}",
        "{% if add_generation_prompt %}x{% endif %} ",
        "{% if add_generation_prompt %}{% if add_generation_prompt %}x{% endif %}{% endif %}",
    }) |source| {
        var compiled = try template.compile(a, source, null);
        defer compiled.deinit();
        try t.expectEqual(@as(?usize, null), compiled.split);
        var mark: ?usize = 7;
        _ = try render(a, &compiled, .{ .array = std.json.Array.init(a) }, true, &mark);
        try t.expectEqual(@as(?usize, null), mark);
    }
    var trailing = try template.compile(a, "a{% if add_generation_prompt %}x{% endif %}\n", null);
    defer trailing.deinit();
    try t.expectEqual(@as(?usize, 1), trailing.split);
}

test "render watch: a render stopped past its difference is a prefix of the full one, text and spans" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var compiled = try template.compile(a, glm_like, null);
    defer compiled.deinit();
    var prng = std.Random.DefaultPrng.init(0x7761);
    const r = prng.random();
    var stopped: usize = 0;
    for (0..400) |_| {
        var ref_more: template.More = .{};
        const reference = try template.renderMore(a, &compiled, try chat(a, r), .null, .null, .null, true, null, &ref_more);
        const messages = try chat(a, r);
        var full_more: template.More = .{};
        const full = try template.renderMore(a, &compiled, messages, .null, .null, .null, true, null, &full_more);
        var watch: template.Watch = .{ .against = reference.s, .slack = r.uintLessThan(usize, 12) };
        var more: template.More = .{ .watch = &watch };
        const cut = try template.renderMore(a, &compiled, messages, .null, .null, .null, true, null, &more);
        try t.expectEqual(more.cut, watch.stopped);
        try t.expect(std.mem.startsWith(u8, full.s, cut.s));
        try t.expectEqualSlices(template.Span, full.spans[0..cut.spans.len], cut.spans);
        if (!more.cut) {
            try t.expectEqualStrings(full.s, cut.s);
            try t.expectEqual(full_more.history, more.history);
            continue;
        }
        stopped += 1;
        try t.expectEqual(@as(?usize, null), more.history);
        // stopped only once `slack` bytes past where it left the reference
        const d = watch.diverged.?;
        try t.expect(cut.s.len >= d + watch.slack);
        try t.expectEqual(d, std.mem.indexOfDiff(u8, reference.s, cut.s).?);
    }
    try t.expect(stopped > 100);
}
