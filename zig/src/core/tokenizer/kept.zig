//! Ids kept with their added-token segments: a later text reuses the ids of the segments it repeats, unencoded.
const std = @import("std");
const tok = @import("tokenizer.zig");
const steps = @import("steps.zig");
const memo = @import("memo.zig");
const Allocator = std.mem.Allocator;
const Span = tok.Span;
const Segment = tok.Segment;

/// `ids` equal encodeMasked(text, spans); no segments when nothing can be reused (no spans, or lossy UTF-8).
pub const Kept = struct {
    text: []const u8,
    spans: []const Span,
    segs: []const Segment = &.{},
    starts: []const usize = &.{},
    ids: []const u32,
};

/// Ids whose concatenation `head ++ mid ++ tail` is encodeMasked(text, spans): kept, encoded, kept.
pub const Pieces = struct {
    head: []const u32 = &.{},
    mid: []const u32 = &.{},
    tail: []const u32 = &.{},
};

/// Only valid UTF-8 with request spans splits as encodeMasked does without a lossy copy.
fn reusable(text: []const u8, spans: []const Span) bool {
    return spans.len > 0 and std.unicode.utf8ValidateSlice(text);
}

/// encodeMasked(text, spans) with the segments and each segment's first id index; slices of `text` and `spans` kept.
pub fn keep(t: *const tok.Tokenizer, a: Allocator, text: []const u8, spans: []const Span) tok.Error!Kept {
    if (!reusable(text, spans)) return .{ .text = text, .spans = spans, .ids = try t.encodeMasked(a, text, spans) };
    try tok.checkSpans(text, spans, null);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
    const segs = try a.dupe(Segment, try t.segments(&ctx, text, spans));
    const starts = try a.alloc(usize, segs.len + 1);
    var ids: std.ArrayList(u32) = .empty;
    try t.encodeSegments(&ctx, text, spans, segs, &ids, starts, null);
    return .{ .text = text, .spans = spans, .segs = segs, .starts = starts, .ids = try a.dupe(u32, ids.items) };
}

/// encodeMasked(text, spans) as pieces: the leading and trailing segments `kept` repeats are its ids, not re-encoded.
pub fn pieces(t: *const tok.Tokenizer, a: Allocator, kept: *const Kept, text: []const u8, spans: []const Span) tok.Error!Pieces {
    if (kept.segs.len == 0 or !reusable(text, spans)) return .{ .mid = try t.encodeMasked(a, text, spans) };
    try tok.checkSpans(text, spans, null);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
    const segs = try t.segments(&ctx, text, spans);
    const both = @min(kept.segs.len, segs.len);
    var head: usize = 0;
    while (head < both and same(kept, kept.segs[head], text, spans, segs[head])) head += 1;
    var tail: usize = 0;
    while (tail < both - head and same(kept, kept.segs[kept.segs.len - 1 - tail], text, spans, segs[segs.len - 1 - tail])) tail += 1;
    var ids: std.ArrayList(u32) = .empty;
    try t.encodeSegments(&ctx, text, spans, segs[head .. segs.len - tail], &ids, null, null);
    return .{ .head = kept.ids[0..kept.starts[head]], .mid = try a.dupe(u32, ids.items), .tail = kept.ids[kept.starts[kept.segs.len - tail]..] };
}

/// A cut text's leading segment ids, as every longer text with its bytes and clipped spans encodes them, or null.
pub fn prefix(t: *const tok.Tokenizer, a: Allocator, kept: *const Kept, text: []const u8, spans: []const Span) tok.Error!?Pieces {
    if (kept.segs.len == 0 or !reusable(text, spans)) return null;
    try tok.checkSpans(text, spans, null);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
    const segs = try t.segments(&ctx, text, spans);
    // an added token no later byte can lengthen (its trie walk ends inside the text) fixes every segment up to it
    var reach: usize = 0;
    for (t.added) |x| reach = @max(reach, x.content.len);
    var fixed = segs.len;
    while (fixed > 0 and !(segs[fixed - 1].id != null and segs[fixed - 1].start + reach < text.len)) fixed -= 1;
    if (fixed == 0) return null;
    for (segs[0 .. fixed - 1], segs[1..fixed]) |x, y| if (x.end > y.start) return null;
    var head: usize = 0;
    const both = @min(kept.segs.len, fixed);
    while (head < both and same(kept, kept.segs[head], text, spans, segs[head])) head += 1;
    var ids: std.ArrayList(u32) = .empty;
    try t.encodeSegments(&ctx, text, spans, segs[head..fixed], &ids, null, null);
    return .{ .head = kept.ids[0..kept.starts[head]], .mid = try a.dupe(u32, ids.items) };
}

/// Whether two segments encode alike: the same id, or the same bytes, start-of-text flag and clipped request spans.
fn same(kept: *const Kept, k: Segment, text: []const u8, spans: []const Span, p: Segment) bool {
    if (k.id != null or p.id != null) return k.id != null and p.id != null and k.id.? == p.id.?;
    if (k.end - k.start != p.end - p.start or (k.start == 0) != (p.start == 0)) return false;
    if (!memo.equal(kept.text[k.start..k.end], text[p.start..p.end])) return false;
    var i = firstAfter(kept.spans, k.start);
    var j = firstAfter(spans, p.start);
    while (true) : ({
        i += 1;
        j += 1;
    }) {
        const in_k = i < kept.spans.len and kept.spans[i][0] < k.end;
        const in_p = j < spans.len and spans[j][0] < p.end;
        if (in_k != in_p) return false;
        if (!in_k) return true;
        const a = clip(kept.spans[i], k);
        const b = clip(spans[j], p);
        if (a[0] != b[0] or a[1] != b[1]) return false;
    }
}

/// The first span ending after `at` (spans are ordered and disjoint).
fn firstAfter(spans: []const Span, at: usize) usize {
    var lo: usize = 0;
    var hi = spans.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (spans[mid][1] <= at) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// A span clipped to a segment, relative to its start.
fn clip(p: Span, s: Segment) Span {
    return .{ @max(s.start, p[0]) - s.start, @min(s.end, p[1]) - s.start };
}

test {
    _ = @import("kept_test.zig");
}
