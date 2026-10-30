//! Kept pieces against a fresh encodeMasked on random texts: masked added spellings, strips, merges, refusals.
const std = @import("std");
const tok = @import("tokenizer.zig");
const kept = @import("kept.zig");
const t = std.testing;

// Merges span a whole gap; "<V>" is vocabulary (a request spelling it is refused); "<S>>" lengthens "<S>".
const source =
    \\{"added_tokens":[{"content":"<S>","special":true,"normalized":false},
    \\{"content":"<S>>","special":true,"normalized":false},
    \\{"content":"<L>","special":true,"normalized":false,"lstrip":true},
    \\{"content":"<R>","special":true,"normalized":false,"rstrip":true},
    \\{"content":"<N>","special":true,"normalized":true},
    \\{"content":"<V>","special":true,"normalized":false}],
    \\"model":{"type":"BPE","unk_token":"?","vocab":{"a":0,"b":1,"c":2," ":3,"<":4,">":5,"S":6,"L":7,"R":8,"N":9,
    \\"V":10,"\n":11,"ab":12,"abc":13,"?":14,"<V":15,"<V>":16,"bc":17},"merges":["a b","ab c","b c","< V","<V >"]}}
;

const words = [_][]const u8{ "<S>", "<L>", "<R>", "<N>", "<V>", " ", "  ", "a", "b", "c", "ab", "bc", "\n", "<", "S>", ">", "\u{e9}", "\xff" };

const Part = struct { word: u8, request: bool };

fn text(a: std.mem.Allocator, parts: []const Part) !struct { text: []u8, spans: []tok.Span } {
    var out: std.ArrayList(u8) = .empty;
    var spans: std.ArrayList(tok.Span) = .empty;
    for (parts) |p| {
        const at = out.items.len;
        try out.appendSlice(a, words[p.word]);
        if (!p.request) continue;
        // adjacent request parts are one span, as a template writes one string
        if (spans.items.len > 0 and spans.items[spans.items.len - 1][1] == at) spans.items[spans.items.len - 1][1] = out.items.len else try spans.append(a, .{ at, out.items.len });
    }
    return .{ .text = out.items, .spans = spans.items };
}

fn part(r: std.Random, invalid: bool) Part {
    const n: u8 = if (invalid) words.len else words.len - 1;
    return .{ .word = r.uintLessThan(u8, n), .request = r.boolean() };
}

fn same(a: std.mem.Allocator, tokenizer: *const tok.Tokenizer, k: *const kept.Kept, probe: []const u8, spans: []const tok.Span) !void {
    const want = tokenizer.encodeMasked(a, probe, spans);
    const got = kept.pieces(tokenizer, a, k, probe, spans);
    if (want) |ids| {
        const p = try got;
        try t.expectEqualSlices(u32, ids, try std.mem.concat(a, u32, &.{ p.head, p.mid, p.tail }));
    } else |err| try t.expectError(err, got);
}

test "kept pieces: a segment starting the text encodes apart from the same bytes later in it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try tok.parse(a,
        \\{"added_tokens":[{"content":"<S>","special":true,"normalized":false}],
        \\"pre_tokenizer":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"first","split":false},
        \\"model":{"type":"WordLevel","unk_token":"?","vocab":{"?":0,"ab":1,"\u2581ab":2}}}
    );
    defer tokenizer.deinit();
    const held = try kept.keep(&tokenizer, a, "<S>ab", &.{.{ 3, 5 }});
    const want = try tokenizer.encodeMasked(a, "ab", &.{.{ 0, 2 }});
    // the same two bytes, but first in the text: Metaspace prepends to them only there
    try t.expect(!std.mem.eql(u32, want, held.ids[1..]));
    const got = try kept.pieces(&tokenizer, a, &held, "ab", &.{.{ 0, 2 }});
    try t.expectEqualSlices(u32, want, try std.mem.concat(a, u32, &.{ got.head, got.mid, got.tail }));
}

test "kept pieces: random edits of a masked text give encodeMasked's ids or its error" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try tok.parse(a, source);
    defer tokenizer.deinit();
    tokenizer.memo = try tok.Memo.create(a, 1 << 20);
    var prng = std.Random.DefaultPrng.init(0x6b657074);
    const r = prng.random();
    var reused: usize = 0;
    for (0..3000) |round| {
        const invalid = round % 10 == 0;
        var base: std.ArrayList(Part) = .empty;
        for (0..r.uintLessThan(usize, 40)) |_| try base.append(a, part(r, invalid));
        const b = try text(a, base.items);
        const want = tokenizer.encodeMasked(a, b.text, b.spans);
        const k = kept.keep(&tokenizer, a, b.text, b.spans);
        if (want) |ids| try t.expectEqualSlices(u32, ids, (try k).ids) else |err| {
            try t.expectError(err, k);
            continue;
        }
        const held = try k;
        var probe = try base.clone(a);
        for (0..1 + r.uintLessThan(usize, 3)) |_| {
            const at = if (probe.items.len == 0) 0 else r.uintLessThan(usize, probe.items.len + 1);
            switch (r.uintLessThan(u8, 6)) {
                0 => try probe.insert(a, at, part(r, invalid)),
                1 => if (at < probe.items.len) {
                    _ = probe.orderedRemove(at);
                },
                2 => if (at < probe.items.len) {
                    probe.items[at] = part(r, invalid);
                },
                3 => if (at < probe.items.len) {
                    probe.items[at].request = !probe.items[at].request;
                },
                4 => try probe.append(a, part(r, invalid)),
                else => probe.shrinkRetainingCapacity(at),
            }
        }
        const p = try text(a, probe.items);
        try same(a, &tokenizer, &held, p.text, p.spans);
        const got = kept.pieces(&tokenizer, a, &held, p.text, p.spans) catch continue;
        if (got.head.len + got.tail.len > 0) reused += 1;
    }
    // the reuse itself must be exercised, or the comparison above proves nothing about it
    try t.expect(reused > 500);
}

test "kept pieces: a cut text's fixed ids lead every longer text's encoding, errors included" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try tok.parse(a, source);
    defer tokenizer.deinit();
    var prng = std.Random.DefaultPrng.init(0x637574);
    const r = prng.random();
    var fixed: usize = 0;
    for (0..3000) |round| {
        const invalid = round % 10 == 0;
        var base: std.ArrayList(Part) = .empty;
        for (0..r.uintLessThan(usize, 30)) |_| try base.append(a, part(r, false));
        const b = try text(a, base.items);
        const held = kept.keep(&tokenizer, a, b.text, b.spans) catch continue;
        // a longer text sharing a random lead with the kept one, cut anywhere between its parts
        var full = try a.dupe(Part, base.items[0..r.uintLessThan(usize, base.items.len + 1)]);
        for (0..r.uintLessThan(usize, 30)) |_| full = try std.mem.concat(a, Part, &.{ full, &.{part(r, invalid)} });
        const f = try text(a, full);
        const p = try text(a, full[0..r.uintLessThan(usize, full.len + 1)]);
        const got = kept.prefix(&tokenizer, a, &held, p.text, p.spans);
        const want = tokenizer.encodeMasked(a, f.text, f.spans);
        const lead = got catch |err| {
            try t.expectError(err, want);
            continue;
        } orelse continue;
        const ids = want catch continue; // a later segment the cut text does not reach may refuse
        try t.expect(std.mem.startsWith(u32, ids, try std.mem.concat(a, u32, &.{ lead.head, lead.mid })));
        fixed += 1;
    }
    try t.expect(fixed > 500);
}

test "kept pieces: a long kept prefix and suffix around one changed request, and a refused one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try tok.parse(a, source);
    defer tokenizer.deinit();
    const head = "<S>abc abc<L>  ab<R>  ab<S>";
    const tail = "<S>abc<N>tail";
    const base = head ++ "request one" ++ tail;
    const held = try kept.keep(&tokenizer, a, base, &.{.{ head.len, head.len + 11 }});
    const changed = head ++ "changed text" ++ tail;
    const spans = [_]tok.Span{.{ head.len, head.len + 12 }};
    const got = try kept.pieces(&tokenizer, a, &held, changed, &spans);
    try t.expect(got.head.len > 0 and got.tail.len > 0);
    try t.expectEqualSlices(u32, try tokenizer.encodeMasked(a, changed, &spans), try std.mem.concat(a, u32, &.{ got.head, got.mid, got.tail }));
    const refused = head ++ "changed <V> text" ++ tail;
    const quoted = [_]tok.Span{.{ head.len, head.len + 16 }};
    try t.expectError(error.UnsafeAddedTokenMask, tokenizer.encodeMasked(a, refused, &quoted));
    try t.expectError(error.UnsafeAddedTokenMask, kept.pieces(&tokenizer, a, &held, refused, &quoted));
}
