//! TF_GLM_OVN_ONE_CHUNK's states at marks inside a chunk: captured during it, assembled as the knob-off path keeps them
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const oc = @import("cuda_one_chunk.zig");
const sparse = @import("cuda_sparse.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const state = @import("cuda_state.zig");
const snapshots = @import("cuda_snapshots.zig");
const layout = @import("cuda_snapshot_layout.zig");
const pages = @import("cuda_snapshot_pages.zig");

/// Marks one chunk keeps from inside it, each with its own copy of every KDA layer's conv window and both states.
pub const slots = 2;

/// A pool's rows past a mark that a state there holds: its partial pool and the spare row (cuda_snapshot_layout).
const pool_rows = engine.prompt_rows / sparse.pool + 3;

/// The device copies a step's marks need, made at load (one allocation), and the step's marks.
pub const Capture = struct {
    gpa: std.mem.Allocator,
    buf: cuda.DeviceBuffer,
    cur: []u1, // [slots][kda layers]: each layer's parity at the mark
    nk: usize,
    nd: usize,
    rec_layer: usize,
    conv_layer: usize,
    ring_row: usize,
    pk_row: usize,
    hidden_bytes: usize,
    rec: u64, // [slots][2][kda layers] states, as State.rec
    conv: u64, // [slots][kda layers][3][conv_c] windows, as State.conv
    ring: u64, // [dsa layers][2][prompt_rows] index rows before the chunk, from ring_lo
    pooled: u64, // [dsa layers][pool_rows] pooled keys before the chunk, from pool_lo
    hidden: u64, // the first-draft row before the chunk
    armed: ?oc.Plan = null, // the next chunk's plan (oc.arm), taken by oc.enter
    plan: oc.Plan = undefined, // the chunk's, while its forward runs
    at: [slots]usize = undefined,
    marks: usize = 0, // marks of the last chunk whose states oc.leave keeps
    taking: ?usize = null, // the mark a keep takes now
    end: usize = 0,
    base: usize = 0,
    ring_lo: usize = 0,
    pool_lo: usize = 0,

    /// Sized from the bound sequence (every sequence has its shapes).
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, st: *const state.State, i_dim: usize, hidden: usize) !*Capture {
        const c = try gpa.create(Capture);
        errdefer gpa.destroy(c);
        const nk = st.kda_layers;
        const nd = st.caches.len;
        const pk_row = sparse.pooledRow(i_dim);
        const sizes = [_]usize{ slots * 2 * nk * st.rec_layer, slots * nk * 3 * st.conv_c * 2, nd * 2 * engine.prompt_rows * st.ring_row, nd * pool_rows * pk_row, hidden * 2 };
        var total: usize = 0;
        var at: [sizes.len]usize = undefined;
        for (sizes, &at) |n, *o| {
            o.* = total;
            total += std.mem.alignForward(usize, n, 256);
        }
        var buf = try cuda.DeviceBuffer.alloc(d, total);
        errdefer buf.free();
        const cur = try gpa.alloc(u1, slots * nk);
        @memset(cur, 0);
        c.* = .{ .gpa = gpa, .buf = buf, .cur = cur, .nk = nk, .nd = nd, .rec_layer = st.rec_layer, .conv_layer = 3 * st.conv_c * 2, .ring_row = st.ring_row, .pk_row = pk_row, .hidden_bytes = hidden * 2, .rec = buf.ptr + at[0], .conv = buf.ptr + at[1], .ring = buf.ptr + at[2], .pooled = buf.ptr + at[3], .hidden = buf.ptr + at[4] };
        std.log.info("glm: TF_GLM_OVN_ONE_CHUNK: marks kept from inside a prompt chunk, {d} MiB of captures", .{total >> 20});
        return c;
    }

    pub fn destroy(c: *Capture) void {
        c.buf.free();
        c.gpa.free(c.cur);
        c.gpa.destroy(c);
    }
};

fn copy(d: *const cuda.Driver, stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
    if (n == 0) return;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
}

/// At a mark's cut in KDA layer `li`: its conv window there and both its states (the mark's and the cut's before).
pub fn layer(c: *Capture, d: *const cuda.Driver, stream: cuda.Stream, st: *const state.State, li: usize, slot: usize) !void {
    if (st.kda_layers != c.nk or st.rec_layer != c.rec_layer) return error.SnapshotShape;
    const conv = st.conv + li * c.conv_layer;
    try copy(d, stream, c.conv + (slot * c.nk + li) * c.conv_layer, conv, c.conv_layer);
    for (0..2) |q| {
        const copy_q: u1 = @intCast(q);
        try copy(d, stream, c.rec + ((slot * 2 + q) * c.nk + li) * c.rec_layer, st.recAt(copy_q, li), c.rec_layer);
    }
    c.cur[slot * c.nk + li] = st.cur[li];
}

/// Pooled rows [lo, lo + n) of cache `x` and `flat` (a page's span at a time on a paged sequence), either way.
fn pooledRows(d: *const cuda.Driver, stream: cuda.Stream, st: *const state.State, x: dsa_block.Caches, pk_row: usize, lo: usize, n: usize, flat: u64, into_flat: bool) !void {
    const span = if (x.pt == 0) 0 else state.page_rows / sparse.pool * pk_row;
    const part: layout.Part = .{ .live = x.pk, .len = (lo + n) * pk_row, .span = span, .page = x.ps };
    const pl: layout.Plan = .{ .ids = st.pages };
    var done = lo * pk_row;
    while (done < part.len) {
        const p = pl.piece(part, done);
        const f = flat + (done - lo * pk_row);
        try copy(d, stream, if (into_flat) f else p.live, if (into_flat) p.live else f, p.len);
        done += p.len;
    }
}

/// Before the chunk to `end` (rings rebased): what it overwrites that its marks' states hold: ring, pool, draft rows.
pub fn before(c: *Capture, e: *engine.Engine, st: *const state.State, end: usize) !void {
    if (st.caches.len != c.nd or st.ring_row != c.ring_row) return error.SnapshotShape;
    const first = c.at[0];
    const last = c.at[c.marks - 1];
    c.end = end;
    c.base = st.base;
    c.ring_lo = first - st.base;
    const ring_hi = @min(end - st.base, st.ring_rows);
    if (c.ring_lo < ring_hi) for (0..c.nd) |j| for (0..2) |q| {
        const dst = c.ring + (j * 2 + q) * engine.prompt_rows * c.ring_row;
        try copy(e.d, e.stream, dst, st.rings[2 * j + q] + c.ring_lo * c.ring_row, (ring_hi - c.ring_lo) * c.ring_row);
    };
    c.pool_lo = first / sparse.pool;
    for (st.caches, 0..) |x, j| {
        const hi = @min(last / sparse.pool + 2, x.pk_rows);
        if (hi > c.pool_lo) try pooledRows(e.d, e.stream, st, x, c.pk_row, c.pool_lo, hi - c.pool_lo, c.pooled + j * pool_rows * c.pk_row, true);
    }
    try copy(e.d, e.stream, c.hidden, st.prompt_hidden, c.hidden_bytes);
}

/// Where a plan's part from `live` starts in its lease.
fn offsetOf(plan: *const layout.Plan, live: u64) !usize {
    var off: usize = 0;
    for (plan.parts[0..plan.count]) |p| {
        if (p.live == live) return off;
        off += p.len;
    }
    return error.SnapshotShape;
}

/// Mark `k`'s state under `id`: the live prefix to the mark, then its position, KDA, ring, pool and draft rows put back
pub fn keepAt(e: anytype, id: u64, k: usize) !void {
    const c = e.one.?;
    const st = &e.cur.st;
    const at = c.at[k];
    if (e.snaps.contains(id)) return error.StateKept;
    if (st.pending != 0 or k >= c.marks or c.end != st.pos) return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const plan = try layout.Plan.init(st, &e.c, at, if (e.mtp) st.mtp_len else null);
    var lease = try pool.take(plan.len);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, c.cur[k * c.nk ..][0..c.nk]);
    errdefer e.gpa.free(cur);
    try pool.move(e.stream, &lease, &plan, true);
    const arena = st.arena.buf.ptr;
    if (plan.parts[0].live != arena or st.pos_dev - arena + 4 > pages.page) return error.SnapshotShape;
    const held = (pool.arena orelse return error.NoSnapshotRoom).ptr + @as(u64, lease.ids.items[0]) * pages.page;
    try e.d.check(e.d.api.cuMemsetD32Async(held + (st.pos_dev - arena), @bitCast(@as(i32, @intCast(at))), 1, e.stream.handle), "cuMemsetD32Async");
    try snapshots.moveFlat(pool, e.stream, &lease, st.conv - arena, c.conv + k * c.nk * c.conv_layer, c.nk * c.conv_layer, true);
    try snapshots.moveFlat(pool, e.stream, &lease, st.rec - arena, c.rec + k * 2 * c.nk * c.rec_layer, 2 * c.nk * c.rec_layer, true);
    for (st.caches, 0..) |x, j| {
        const lo = at - c.base;
        const hi = @min(c.end - c.base, @min(at, st.ring_rows));
        if (lo < hi) for (0..2) |q| {
            const src = c.ring + ((j * 2 + q) * engine.prompt_rows + lo - c.ring_lo) * c.ring_row;
            try snapshots.moveFlat(pool, e.stream, &lease, try offsetOf(&plan, st.rings[2 * j + q]) + lo * c.ring_row, src, (hi - lo) * c.ring_row, true);
        };
        const p0 = at / sparse.pool;
        const p1 = @min(p0 + 2, x.pk_rows);
        const src = c.pooled + (j * pool_rows + p0 - c.pool_lo) * c.pk_row;
        if (p1 > p0) try snapshots.moveFlat(pool, e.stream, &lease, try offsetOf(&plan, x.pk) + p0 * c.pk_row, src, (p1 - p0) * c.pk_row, true);
    }
    const tail = try offsetOf(&plan, st.mtp_pos_dev);
    try snapshots.moveFlat(pool, e.stream, &lease, tail + (st.prompt_hidden - st.mtp_pos_dev), c.hidden, c.hidden_bytes, true);
    try e.stream.synchronize();
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = plan.len, .pos = at, .mtp_len = st.mtp_len, .cur = cur, .bases = .{ c.base, st.mtp_base } });
}
