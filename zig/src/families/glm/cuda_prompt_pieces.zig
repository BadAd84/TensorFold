//! TF_GLM_OVN_MULTI_FILL: one prompt pass over several streams' pieces, each from where its prompt stands (0, a kept
//! state it resumes, or its last piece's end) to its next mark or its end: Engine.prefillChunk's steps, batched.
//! A piece starting at a mark fires it first (rank 0's hook keeps the state, the gate gives every rank its copy).
const std = @import("std");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const snapshots = @import("cuda_snapshots.zig");
const Engine = engine.Engine;

/// One stream's piece: rows from `start` of its `len`-id prompt; `part` those ids and the next one (if any).
pub const Piece = struct {
    slot: u32,
    start: usize,
    rows: usize,
    len: usize,
    part: []const u32,
    from: ?engine.Resume = null, // its first piece: the kept state it resumes (from.at == start)
    mark: bool = false, // start is one of its marks: the state there is kept before the pass
};

/// Header words a piece takes on the wire: slot, start, rows, len, the resumed id's two halves, its position, flags.
const head_words = 8;
/// The most words a pass's step carries: every header, a chunk's rows and each piece's next id.
pub const words = engine.max_together * (head_words + 1) + engine.prompt_rows;

/// Whether `pieces` go in one pass: 1 to max_together on distinct slots, prompt_rows in all, each a valid step.
pub fn fits(pieces: []const Piece, capacity: usize) bool {
    if (pieces.len == 0 or pieces.len > engine.max_together) return false;
    var rows: usize = 0;
    for (pieces, 0..) |p, i| {
        if (p.rows == 0 or p.start + p.rows > p.len or p.len + fw.window_rows >= capacity) return false;
        if (p.part.len != @min(p.len - p.start, p.rows + 1)) return false;
        if (p.from) |r| if (r.at != p.start or r.at == 0) return false;
        for (pieces[0..i]) |q| if (q.slot == p.slot) return false;
        rows += p.rows;
    }
    return rows <= fw.prefill_rows;
}

pub fn encode(pieces: []const Piece, capacity: usize, buf: []u32) ![]u32 {
    if (!fits(pieces, capacity)) return error.BadPrompts;
    var n: usize = head_words * pieces.len;
    for (pieces) |p| n += p.part.len;
    if (buf.len < n) return error.BadStep;
    var at: usize = head_words * pieces.len;
    for (pieces, 0..) |p, i| {
        const r = p.from orelse engine.Resume{ .id = 0, .at = 0 };
        buf[head_words * i ..][0..head_words].* = .{ p.slot, @intCast(p.start), @intCast(p.rows), @intCast(p.len), @truncate(r.id), @intCast(r.id >> 32), @intCast(r.at), @as(u32, @intFromBool(p.mark)) | @as(u32, @intFromBool(p.from != null)) << 1 };
        @memcpy(buf[at..][0..p.part.len], p.part);
        at += p.part.len;
    }
    return buf[0..n];
}

pub fn decode(ids: []const u32, count: usize, capacity: usize, out: *[engine.max_together]Piece) ![]const Piece {
    if (count == 0 or count > out.len or ids.len < head_words * count) return error.BadStep;
    var at: usize = head_words * count;
    for (out[0..count], 0..) |*p, i| {
        const w = ids[head_words * i ..][0..head_words];
        const start: usize = w[1];
        const rows: usize = w[2];
        const len: usize = w[3];
        if (start >= len or rows > len - start) return error.BadStep;
        const n = @min(len - start, rows + 1);
        if (n > ids.len - at) return error.BadStep;
        const from: ?engine.Resume = if (w[7] & 2 != 0) .{ .id = @as(u64, w[5]) << 32 | w[4], .at = w[6] } else null;
        p.* = .{ .slot = w[0], .start = start, .rows = rows, .len = len, .part = ids[at..][0..n], .from = from, .mark = w[7] & 1 != 0 };
        at += n;
    }
    if (at != ids.len or !fits(out[0..count], capacity)) return error.BadStep;
    return out[0..count];
}

/// The pass on every rank: resumed pieces restored and marks fired (rank 0's `marks[k]`), the gate, the shared
/// prompt forward, then each piece's drafter rows, its first token at its prompt's end (`firsts[k]`), head, commit.
pub fn run(e: *Engine, pieces: []const Piece, marks: ?[]const ?engine.Marks, firsts: []?u32) !void {
    if (!fits(pieces, e.capacity) or firsts.len < pieces.len) return error.BadPrompts;
    for (pieces, 0..) |p, k| {
        if (p.slot >= e.seqs.items.len or e.seqs.items[p.slot] == null) return error.BadPrompts;
        try e.bind(p.slot);
        if (p.from) |r| { // Engine.prefillStart's resume
            try e.restoreState(r.id);
            if (e.drafter) |dr| try dr.restart(p.slot, r.at);
        }
        if (e.cur.st.pos != p.start or e.cur.st.pending != 0) return error.BadPrompts;
        if (!p.mark) continue;
        if (marks) |m| if (m[k]) |x| x.call(x.ctx, @intCast(p.start)); // Engine.prefillChunk's mark at a chunk start
        try snapshots.gate(e); // the kept state's id to the other ranks, which copy their bound sequence now
    }
    var segs: [engine.max_together]fw.Seg = undefined;
    var ids: [fw.prefill_rows]u32 = undefined;
    var rows: usize = 0;
    for (pieces, 0..) |p, k| {
        try e.bind(p.slot);
        segs[k] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.rows };
        @memcpy(ids[rows..][0..p.rows], p.part[0..p.rows]);
        rows += p.rows;
    }
    try e.gate(null);
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = &e.bp, .segs = segs[0..pieces.len] }};
    var f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = &e.pieces, .split = e.settings.split, .ring = e.settings.ring, .prx = @import("cuda_forward_prx.zig").takes(false), .split_min = e.settings.split_min, .kda_wide = e.settings.kda_wide, .taps = if (e.drafter) |dr| dr.taps(true) else null };
    try f.run(@ptrCast(ids[0..rows]));
    if (e.drafter) |dr| for (pieces, segs[0..pieces.len]) |p, g|
        try dr.promptSegment(e, p.slot, p.start, p.rows, p.len, g.row0);
    const D = e.c.hidden;
    for (pieces, segs[0..pieces.len], 0..) |p, g, k| {
        firsts[k] = null;
        if (p.start + p.rows != p.len) continue;
        try e.bind(p.slot);
        var t: [1]u32 = undefined;
        try e.draw(e.bp.logits + k * e.share.vocab.len() * 2, 1, p.len, 0, &t);
        firsts[k] = t[0];
        try e.holdPrompt(e.bp.fnormed + (g.row0 + g.rows - 1) * D * 2);
    }
    f.pieces = null;
    f.taps = null;
    for (pieces, segs[0..pieces.len]) |p, g| {
        one[0] = .{ .w = &e.w, .st = g.st, .b = &e.bp };
        const nxt: []const u32 = p.part[1..]; // decode._absorb_rows: each row's next token, as prefillChunk's head pass
        if (e.head and nxt.len > 0) {
            try f.mtp(@ptrCast(nxt), &.{e.bp.fnormed + g.row0 * D * 2});
            try g.st.setMtpLen(e.stream, g.st.mtp_len + nxt.len);
        }
        try f.commit(g.rows);
    }
}

test "a pass's pieces cross the wire whole: resumed ids past 32 bits, marks, a piece ending inside its prompt" {
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const pieces = [_]Piece{
        .{ .slot = 3, .start = 0, .rows = 4, .len = 9, .part = ids[0..5], .mark = false },
        .{ .slot = 5, .start = 4, .rows = 5, .len = 9, .part = ids[4..9], .from = .{ .id = (7 << 32) | 11, .at = 4 }, .mark = true },
    };
    var buf: [words]u32 = undefined;
    const wire = try encode(&pieces, 1 << 16, &buf);
    var out: [engine.max_together]Piece = undefined;
    const back = try decode(wire, pieces.len, 1 << 16, &out);
    for (pieces, back) |a, b| {
        try std.testing.expectEqual(a.slot, b.slot);
        try std.testing.expectEqual(a.start, b.start);
        try std.testing.expectEqual(a.rows, b.rows);
        try std.testing.expectEqual(a.len, b.len);
        try std.testing.expectEqual(a.from, b.from);
        try std.testing.expectEqual(a.mark, b.mark);
        try std.testing.expectEqualSlices(u32, a.part, b.part);
    }
    const bad = [_]Piece{.{ .slot = 1, .start = 2, .rows = 3, .len = 9, .part = ids[2..5] }}; // part one id short
    try std.testing.expect(!fits(&bad, 1 << 16));
}

test "multifill fits: per-piece descriptors - rows, bounds, part length, resume row, distinct slots, capacity" {
    const cap = 1 << 16;
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    // a resumed piece: start 3, four rows, its ids from 3 plus the one lookahead; a kept state at its start row
    try std.testing.expect(fits(&.{.{ .slot = 0, .start = 3, .rows = 4, .len = 8, .part = ids[3..8], .from = .{ .id = 9, .at = 3 } }}, cap));
    // a piece that ends its prompt carries no lookahead id (part.len == rows)
    try std.testing.expect(fits(&.{.{ .slot = 0, .start = 4, .rows = 4, .len = 8, .part = ids[4..8] }}, cap));
    // rows == 0 is rejected
    try std.testing.expect(!fits(&.{.{ .slot = 0, .start = 0, .rows = 0, .len = 8, .part = ids[0..1] }}, cap));
    // start + rows past the prompt is rejected
    try std.testing.expect(!fits(&.{.{ .slot = 0, .start = 6, .rows = 4, .len = 8, .part = ids[6..8] }}, cap));
    // the part must be exactly the rows plus the one lookahead id (here one short)
    try std.testing.expect(!fits(&.{.{ .slot = 0, .start = 0, .rows = 4, .len = 8, .part = ids[0..4] }}, cap));
    // a resume row must equal the piece's start, and never be 0 (position 0 is a fresh prompt, not a resume)
    try std.testing.expect(!fits(&.{.{ .slot = 0, .start = 3, .rows = 4, .len = 8, .part = ids[3..8], .from = .{ .id = 9, .at = 2 } }}, cap));
    try std.testing.expect(!fits(&.{.{ .slot = 0, .start = 0, .rows = 4, .len = 8, .part = ids[0..5], .from = .{ .id = 9, .at = 0 } }}, cap));
    // two pieces may not share a stream slot
    const two = [_]Piece{
        .{ .slot = 1, .start = 0, .rows = 2, .len = 8, .part = ids[0..3] },
        .{ .slot = 1, .start = 0, .rows = 2, .len = 8, .part = ids[0..3] },
    };
    try std.testing.expect(!fits(&two, cap));
    // a prompt whose kept state plus a decode window would not fit the capacity is rejected
    try std.testing.expect(!fits(&.{.{ .slot = 0, .start = 0, .rows = 4, .len = 8, .part = ids[0..5] }}, 8 + fw.window_rows));
}
