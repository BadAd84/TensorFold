//! TF_GLM_OVN_FILL_SLICE_MS=<ms> and _FILL_FAST_MS=<ms> (production's TF_GLM_FILL_BUDGET_MS with FILL_DRAFTS=1, and
//! ots0122's FILL_FAST_MS): a prompt chunk beside decoding streams runs in layer slices of about the budget, a decode
//! round between them; a chunk that ends its prompt and whose layers left fit FAST_MS runs them at once. The chunk
//! waits in the prompt buffers (Engine.bp) between slices; decode rounds run on their own buffers (bd, bs, bm).
//! Also the chunk's tail after its last layer (Engine.prefillChunk's, moved here so both paths share it).
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const passes = @import("cuda_forward_slices.zig");
const prof_mod = @import("cuda_prof.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const dflash = @import("dflash.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// The chunk part-way through its layers: its sequence's slot, its rows of a `len`-id prompt, its ids (and the next
/// one) in Slices.ids, the indexer split it took, and the next layer it runs.
const Pending = struct { slot: u32, start: usize, rows: usize, len: usize, n: usize, split: bool, layer: usize };

/// A slice whose GPU time rank 0 has not read yet: its rows and layers, and whether the chunk's tail followed it.
const Seen = struct { rows: usize, layers: usize, last: bool };

pub const Slices = struct {
    budget_ms: f64,
    fast_ms: f64, // 0: off
    hi: usize = 0, // the next chunk step's first slice end (rank 0's chunkOf, the others' from the step); 0: whole
    pending: ?Pending = null,
    ids: [fw.prefill_rows + 1]u32 = undefined,
    row_layer_ms: f64, // rank 0: one layer's ms a row, learned from each slice's GPU time
    tail_ms: f64 = 50, // rank 0: the final norm, head, drafter ingest, draw, MTP head and commit after the last layer
    events: [3]cuda.Event, // before a slice's layers, after them, after the chunk's tail
    seen: ?Seen = null,

    /// Made at load when TF_GLM_OVN_FILL_SLICE_MS is set (every rank: the same environment).
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, budget_ms: u64, fast_ms: u64, layers: usize) !*Slices {
        const s = try gpa.create(Slices);
        errdefer gpa.destroy(s);
        // the prior: a 4,096-row chunk's ~2.35 s (SPEED-REPORT's prompt chunks at any depth) over its layers
        s.* = .{ .budget_ms = @floatFromInt(budget_ms), .fast_ms = @floatFromInt(fast_ms), .row_layer_ms = 2350.0 / (4096.0 * @as(f64, @floatFromInt(@max(layers, 1)))), .events = undefined };
        var made: usize = 0;
        errdefer for (s.events[0..made]) |*ev| ev.deinit();
        for (&s.events) |*ev| {
            ev.* = try cuda.Event.init(d, true);
            made += 1;
        }
        return s;
    }

    pub fn destroy(s: *Slices, gpa: std.mem.Allocator) void {
        for (&s.events) |*ev| ev.deinit();
        gpa.destroy(s);
    }

    /// The step's first slice end, once: the chunk it runs is the step's own.
    pub fn take(s: *Slices) usize {
        const h = s.hi;
        s.hi = 0;
        return h;
    }

    /// The last unread slice's GPU time into the costs (its decode round has synchronized the stream since).
    fn observe(s: *Slices) void {
        const v = s.seen orelse return;
        s.seen = null;
        s.events[1].synchronize() catch return;
        const ms: f64 = cuda.Event.elapsedMs(s.events[0], s.events[1]) catch return;
        const per = ms / @as(f64, @floatFromInt(v.rows * v.layers));
        if (per > 0 and std.math.isFinite(per)) s.row_layer_ms = 0.8 * s.row_layer_ms + 0.2 * per;
        if (!v.last) return;
        s.events[2].synchronize() catch return;
        const tail_ms: f64 = cuda.Event.elapsedMs(s.events[1], s.events[2]) catch return;
        if (tail_ms >= 0 and std.math.isFinite(tail_ms)) s.tail_ms = 0.8 * s.tail_ms + 0.2 * tail_ms;
    }

    /// Where a slice of a `rows`-row chunk from layer `lo` ends: every layer left when they fit FAST_MS and the chunk
    /// ends its prompt (`ends`), else the fewest slices of at most the budget (one layer at least), split evenly.
    fn end(s: *Slices, rows: usize, lo: usize, layers: usize, ends: bool) usize {
        s.observe();
        const left = layers - lo;
        const per = s.row_layer_ms * @as(f64, @floatFromInt(rows)); // one layer's ms
        if (ends and s.fast_ms > 0 and per * @as(f64, @floatFromInt(left)) + s.tail_ms <= s.fast_ms) return layers;
        const fit: usize = @intFromFloat(@min(@as(f64, @floatFromInt(left)), @max(1.0, @floor(s.budget_ms / per))));
        const count = (left + fit - 1) / fit;
        return lo + (left + count - 1) / count;
    }
};

/// Set when TF_GLM_OVN_FILL_SLICE_MS made an engine's Slices: with the knob off nothing reads Engine.slices at all
/// (host fakes leave the field undefined, as they leave every knob's state).
pub var on = false;

/// The engine's slices, null with the knob off.
pub fn of(e: anytype) ?*Slices {
    return if (on) e.slices else null;
}

/// Rank 0: the first slice's end for a chunk of `rows` rows (`ends`: it ends its prompt); 0: the chunk runs whole.
pub fn first(e: *Engine, rows: usize, ends: bool) usize {
    const s = of(e) orelse return 0;
    const layers = e.w.layers.len;
    const hi = s.end(rows, 0, layers, ends);
    return if (hi >= layers) 0 else hi;
}

/// Rank 0: the pending chunk's next slice end: every layer left unless `busy` (another stream decodes).
pub fn next(e: *Engine, busy: bool) usize {
    const s = of(e).?;
    const p = s.pending.?;
    const layers = e.w.layers.len;
    if (!busy) return layers;
    return s.end(p.rows, p.layer, layers, p.start + p.rows == p.len);
}

/// Whether a chunk waits part-way through its layers.
pub fn pending(e: *const Engine) bool {
    const s = of(e) orelse return false;
    return s.pending != null;
}

/// The pending chunk's next layer, null when none waits.
pub fn nextLayer(e: *const Engine) ?usize {
    const s = of(e) orelse return null;
    return if (s.pending) |p| p.layer else null;
}

/// The pending chunk's rows (0 when none waits).
pub fn pendingRows(e: *const Engine) usize {
    const s = of(e) orelse return 0;
    return if (s.pending) |p| p.rows else 0;
}

/// Refuses a prompt step while a chunk waits in the prompt buffers (it would overwrite them).
pub fn idle(e: *const Engine) !void {
    if (pending(e)) return error.ChunkPending;
}

/// A released slot's pending chunk is gone with its sequence (every rank releases alike).
pub fn forget(e: *Engine, slot: u32) void {
    const s = of(e) orelse return;
    if (s.pending) |p| if (p.slot == slot) {
        s.pending = null;
    };
}

/// Engine.prefillChunk with a first slice end `hi`: the rows in, layers 0..hi, the chunk then pending. Every rank
/// runs a chunk whole, as without the knob, if it has prompt logprobs, ends at the last layer or takes Forward.whole.
pub fn begin(e: *Engine, f: *fw.Forward, part: []const u32, start: usize, n: usize, len: usize, split: bool, hi: usize) !?u32 {
    const s = of(e) orelse return error.NoSlices;
    const ids: []const i32 = @ptrCast(part[0..n]);
    if (hi >= e.w.layers.len or e.prompt_logprobs != null or !passes.sliceable(f, n)) {
        try f.run(ids);
        return tail(e, f, part, start, n, len, split);
    }
    @memcpy(s.ids[0..part.len], part);
    s.pending = .{ .slot = e.slot, .start = start, .rows = n, .len = len, .n = part.len, .split = split, .layer = 0 };
    errdefer s.pending = null;
    try s.events[0].record(e.stream);
    try passes.enter(f, ids);
    try passes.layers(f, n, 0, hi);
    try s.events[1].record(e.stream);
    s.pending.?.layer = hi;
    s.seen = .{ .rows = n, .layers = hi, .last = false };
    return null;
}

/// The bound slot's pending chunk's layers lo..hi on every rank (cuda_follow's chunk_part); after its last layer, its
/// tail: the first token when the chunk ends its prompt.
pub fn slice(e: *Engine, lo: usize, hi: usize) !?u32 {
    const s = of(e) orelse return error.NoSlices;
    const p = s.pending orelse return error.NoPendingChunk;
    const layers = e.w.layers.len;
    if (lo != p.layer or hi <= lo or hi > layers or e.slot != p.slot) return error.BadSlice;
    errdefer s.pending = null;
    var one: [1]fw.Rank = undefined;
    var f = e.view(&e.bp, &one); // as prefillChunk's: the prompt buffers, their pieces, the drafter's prompt taps
    f.index_split = p.split;
    if (dflash.ingestsNone(p.start, p.rows, p.len)) f.taps = null;
    try s.events[0].record(e.stream);
    try passes.layers(&f, p.rows, lo, hi);
    try s.events[1].record(e.stream);
    s.seen = .{ .rows = p.rows, .layers = hi - lo, .last = hi == layers };
    if (hi < layers) {
        s.pending.?.layer = hi;
        return null;
    }
    s.pending = null;
    try f.finish(p.rows);
    const token = try tail(e, &f, s.ids[0..p.n], p.start, p.rows, p.len, p.split);
    try s.events[2].record(e.stream);
    return token;
}

/// Engine.prefillChunk after its pass: prompt rows' logprobs, the drafter's ingest, the first token at the prompt's
/// end, the MTP head's rows and the commit; `part` the chunk's ids and the next one.
pub fn tail(e: *Engine, f: *fw.Forward, part: []const u32, start: usize, n: usize, len: usize, split: bool) !?u32 {
    const as_i32: []const i32 = @ptrCast(part);
    e.index_splits += @intFromBool(split);
    var stamp = prof_mod.span(f.prof, e.stream, .prompt_score);
    if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
    stamp.end();
    stamp = prof_mod.span(f.prof, e.stream, .drafter);
    if (e.drafter) |dr| try dr.prompt(e, e.slot, start, n, len);
    stamp.end();
    stamp = prof_mod.span(f.prof, e.stream, .draw);
    const last = start + n == len;
    var token: ?u32 = null;
    if (last) {
        var t: [1]u32 = undefined;
        try e.draw(e.bp.logits, 1, len, 0, &t);
        token = t[0];
    }
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .draw);
    if (last) try e.holdPrompt(e.bp.fnormed + (n - 1) * e.c.hidden * 2);
    stamp = prof_mod.span(f.prof, e.stream, .mtp);
    if (e.head) {
        const nxt = as_i32[1..];
        if (nxt.len > 0) { // decode._absorb_rows
            const mp = f.prof;
            defer f.prof = mp;
            f.prof = null; // the head's pass as one section
            try f.mtp(nxt, &.{e.bp.fnormed});
            f.prof = mp;
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
        }
    }
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .mtp);
    stamp = prof_mod.span(f.prof, e.stream, .commit);
    try f.commit(n);
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .commit);
    return token;
}

test "slices: the fewest of at most the budget, split evenly; FAST_MS takes every layer left of a last chunk" {
    var s: Slices = .{ .budget_ms = 200, .fast_ms = 400, .row_layer_ms = 50.0 / 4096.0, .tail_ms = 50, .events = undefined };
    try std.testing.expectEqual(@as(usize, 4), s.end(4096, 0, 45, false)); // 45 layers of 50 ms: 12 slices, 3-4 each
    try std.testing.expectEqual(@as(usize, 41), s.end(4096, 37, 45, false)); // 8 left: 4 and 4
    try std.testing.expectEqual(@as(usize, 45), s.end(4096, 39, 45, true)); // 6 left: 300 + 50 ms fit 400
    try std.testing.expectEqual(@as(usize, 42), s.end(4096, 39, 45, false)); // the same chunk not ending its prompt
    try std.testing.expectEqual(@as(usize, 1), s.end(16 * 4096, 0, 45, false)); // a layer past the budget runs alone
}
