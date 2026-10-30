//! TF_GLM_OVN_ONE_CHUNK (production's ots0121): prompt chunks run through their marks, the states kept from inside.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const kda_chunk = @import("cuda_kda_chunk.zig");
const kda_block = @import("cuda_kda_block.zig");
const sparse = @import("cuda_sparse.zig");
const prof = @import("cuda_prof.zig");
const snapshots = @import("cuda_snapshots.zig");
pub const keep = @import("cuda_one_chunk_keep.zig");
pub const Capture = keep.Capture;

/// Set once at load when the knob is on (cuda_engine_ovn.open); off, no Engine field below is read.
pub var on = false;

/// The most knob-off chunk ends inside one step (the header's six free words carry them).
pub const max_cuts = 6;
/// The most marks one step keeps from inside it (each holds a capture slot of KDA states); more end the step there.
pub const slots = keep.slots;

/// A row inside a step where the knob-off path ends a chunk: the KDA chain splits there; `mark`: a state keeps there.
pub const Cut = struct { at: u32, mark: bool };

/// A step from `start` to `end` and the knob-off chunk ends inside it, in order.
pub const Plan = struct {
    start: usize,
    end: usize,
    cuts: [max_cuts]Cut = undefined,
    n: usize = 0,

    pub fn marks(p: *const Plan) usize {
        var k: usize = 0;
        for (p.cuts[0..p.n]) |c| k += @intFromBool(c.mark);
        return k;
    }
};

/// A step from `start`: to the prompt's end when it fits a chunk, else the grid; see the stops below.
pub fn cut(start: usize, len: usize, marks: []const u32, base: usize, ring_rows: usize, keeps: bool) Plan {
    const C = engine.prompt_rows;
    const whole = if (len - start <= C) len else (start / C + 1) * C;
    var p: Plan = .{ .start = start, .end = whole };
    var b = base;
    var x = start;
    var kept: usize = 0;
    while (true) {
        const y = engine.chunkEnd(x, len, marks); // the knob-off path's next chunk, [x, y)
        if (y > b + ring_rows) { // it rebases the index rings at x: the step does only when x is its own start
            if (x != start) { // x is the last cut: the step ends there and the next one starts (and rebases) at it
                p.n -= 1;
                p.end = x;
                return p;
            }
            b = x / sparse.pool * sparse.pool;
        }
        if (y >= whole) return p;
        const mark = std.mem.indexOfScalar(u32, marks, @intCast(y)) != null;
        if ((mark and (!keeps or kept == slots)) or p.n == max_cuts) { // a stop: the mark keeps at the next step's start
            p.end = y;
            return p;
        }
        p.cuts[p.n] = .{ .at = @intCast(y), .mark = mark };
        p.n += 1;
        kept += @intFromBool(mark);
        x = y;
    }
}

/// Rank 0's step from `start` under the knob (null: off, a paired step or a sliced one, which cut as before).
pub fn plan(e: *const engine.Engine, start: usize, len: usize, marks: []const u32, from: ?engine.Resume, apart: bool) ?Plan {
    if (!on or apart) return null;
    if (e.one == null) return null;
    const base = if (from) |r| (e.snaps.get(r.id) orelse return null).bases[0] else e.cur.st.base; // as restored
    return cut(start, len, marks, base, e.cur.st.ring_rows, keepsInside(e));
}

/// Marks stop steps when the MTP head absorbs prompt rows or with DRAFT_KEEPWIN: their state at a mark is not captured.
fn keepsInside(e: *const engine.Engine) bool {
    if (e.head) return false;
    if (cuda.ovn.on("DRAFT_KEEPWIN") and e.drafter != null) return false;
    return true;
}

/// The cuts' words in a chunk step's header: temperature, top_p and min_p, which a chunk step leaves 0 (0: no cut).
fn words(h: anytype) *[2 * 3]u32 {
    const H = @TypeOf(h.*);
    comptime std.debug.assert(@offsetOf(H, "min_p") == @offsetOf(H, "temperature") + 16);
    return @ptrCast(@alignCast(&h.temperature));
}

/// Rank 0: the step's cuts into its header (bit 31: a mark).
pub fn put(h: anytype, p: ?Plan) void {
    const w = words(h);
    @memset(w, 0);
    const q = p orelse return;
    for (q.cuts[0..q.n], 0..) |c, i| w[i] = c.at | @as(u32, @intFromBool(c.mark)) << 31;
}

/// A follower: the step's plan from its header (null: no cut).
pub fn read(h: anytype, start: usize, rows: usize) !?Plan {
    const w = words(@constCast(h));
    if (w[0] == 0) return null;
    var p: Plan = .{ .start = start, .end = start + rows };
    for (w) |v| {
        if (v == 0) break;
        const at = v & ~(@as(u32, 1) << 31);
        if (at <= start or at >= p.end or (p.n > 0 and at <= p.cuts[p.n - 1].at)) return error.BadStep;
        p.cuts[p.n] = .{ .at = at, .mark = v >> 31 != 0 };
        p.n += 1;
    }
    return p;
}

/// The next prefillChunk's plan on this rank (Leader.chunkOf's, or a follower's from the header).
pub fn arm(e: *engine.Engine, p: ?Plan) !void {
    if (!on) {
        if (p != null) return error.OneChunkKnobOff; // the ranks' environments disagree
        return;
    }
    const cap = e.one orelse return error.OneChunkKnobOff;
    cap.armed = p;
}

/// Before a chunk's forward: its cuts on `f`, the rings rebased as the forward would, then what its marks need kept.
pub fn enter(e: *engine.Engine, f: *fw.Forward, start: usize, rows: usize) !void {
    if (!on) return;
    const cap = e.one orelse return;
    cap.marks = 0;
    const p = cap.armed orelse return;
    cap.armed = null;
    if (p.start != start or p.end != start + rows) return; // left by a step cancelled at its gate: this one cuts nothing
    if (p.n == 0) return;
    cap.plan = p;
    for (p.cuts[0..p.n]) |c| if (c.mark) {
        cap.at[cap.marks] = c.at;
        cap.marks += 1;
    };
    f.one = cap;
    if (cap.marks == 0) return;
    try e.cur.st.ready(e.stream, rows);
    try keep.before(cap, e, &e.cur.st, start + rows);
}

/// After the chunk's commit: each mark kept in order (rank 0 asks its store), every rank at a gate for each.
pub fn leave(e: *engine.Engine, marks: ?engine.Marks) !void {
    if (!on) return;
    const cap = e.one orelse return;
    const n = cap.marks;
    cap.marks = 0;
    for (cap.at[0..n], 0..) |at, k| {
        cap.taking = k;
        defer cap.taking = null;
        if (marks) |m| m.call(m.ctx, @intCast(at));
        try snapshots.gate(e);
    }
}

/// The mark a keep now takes from inside the last chunk (null: a keep of the live state).
pub fn taking(e: anytype) ?usize {
    if (comptime !@hasField(@TypeOf(e.*), "one")) {
        return null;
    } else {
        if (!on) return null;
        const cap = e.one orelse return null;
        return cap.taking;
    }
}

/// Whether `at` is the mark a keep now takes from inside the last chunk (cuda_native.snap_save's position check).
pub fn inside(e: anytype, at: u32) bool {
    const k = taking(e) orelse return false;
    return e.one.?.at[k] == at;
}

/// Forward.mixer's KDA block on a chunk with cuts: the chain in the knob-off path's calls, a mark's state captured.
pub fn kdaBlock(f: *const fw.Forward, r: fw.Rank, k: kda_block.Layer, li: usize, rows: usize, out: ?u64, fronted: bool, cap: *Capture) !void {
    const c = &r.w.config;
    const st = r.st;
    var s = r.b.kda;
    s.p = r.b.kproj;
    var stamp = prof.span(f.prof, f.stream, .projections);
    if (fronted) try kda_block.dumpOnce(f.stream, s.p, k.matWidth(), rows) else try kda_block.project(f.set(), f.stream, k, s, r.b.normed, c.hidden, rows);
    stamp.end();
    stamp = prof.span(f.prof, f.stream, .kda);
    const n = k.heads * kda.dk;
    const w = k.matWidth();
    const conv = st.conv + li * 3 * st.conv_c * 2;
    const p = &cap.plan;
    var lo: usize = 0;
    var slot: usize = 0;
    for (0..p.n + 1) |i| { // each call as a chunk of its own: from the last's state and conv window, the layer committed
        const hi = if (i < p.n) p.cuts[i].at - st.pos else rows;
        const cur = st.cur[li];
        var v = s;
        v.p += lo * w * 2;
        v.ka += lo * n * 2;
        v.kg += lo * n * 2;
        v.saves.out += lo * n * 2;
        v.saves.state_out = st.recAt(1 - cur, li);
        try kda_block.chainRows(&f.m.kda, f.stream, k, v, hi - lo, conv, st.recAt(cur, li), false, kda_chunk.prompt(&f.m.chunked, st.pos + lo));
        try glue.convShift(f.set(), f.stream, conv, v.p, hi - lo, (hi - lo) * w, w, 1, 3 * n);
        st.cur[li] = 1 - cur;
        if (i < p.n and p.cuts[i].mark) { // the window and both states as the knob-off path holds them at the mark
            try keep.layer(cap, f.d, f.stream, st, li, slot);
            slot += 1;
        }
        lo = hi;
    }
    stamp.end();
    if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
}
