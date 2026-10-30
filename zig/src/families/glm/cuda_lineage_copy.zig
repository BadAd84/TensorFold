//! TF_GLM_OVN_LINEAGE's copies (cuda_lineage tags the rows): a restore copies only the rows the live caches lack, and
//! a keep shares an earlier state's pages for the rows both hold. KDA states and conv windows, the index rings, the
//! head's caches, each cache's partial pool and spare row, and the fields around them are copied whole, as without it.
//! A kept state's lease is the knob-off one with each DSA layer's latent rows and pooled keys starting on a page.
const std = @import("std");
const cuda = @import("cuda");
const lin = @import("cuda_lineage.zig");
const types = @import("cuda_engine_types.zig");
const pages = @import("cuda_snapshot_pages.zig");
const layout = @import("cuda_snapshot_layout.zig");
const gather = @import("cuda_snapshot_gather.zig");
const sparse = @import("cuda_sparse.zig");
const image = @import("cuda_disk_image.zig");
const snapshots = @import("cuda_snapshots.zig");
const State = @import("cuda_state.zig").State;
const window_rows = @import("cuda_state.zig").window_rows;
const ring_bytes = @import("dflash.zig").ring_bytes;
const max_layers = @import("config.zig").max_layers;

const max_parts = @typeInfo(@FieldType(layout.Plan, "parts")).array.len;

/// Row ranges a restore copies (past them its last range runs to the state's end).
const max_ranges = 64;

/// Bytes the knob's keeps and restores copied, left in place, and shared since load (the GPU check reads them).
pub const Stats = struct { copied: u64 = 0, skipped: u64 = 0, shared: u64 = 0 };
pub var stats: Stats = .{};

const Kind = enum { whole, latent, scales, pooled };
const Which = struct { kind: Kind, layer: usize = 0 };

/// What a plan part from `live` is: a DSA layer's latent rows, their scales or pooled keys, or a part copied whole.
fn which(st: *const State, live: u64) Which {
    for (st.caches, 0..) |x, j| {
        if (live == x.lc) return .{ .kind = .latent, .layer = j };
        if (x.ls != 0 and live == x.ls) return .{ .kind = .scales, .layer = j };
        if (live == x.pk) return .{ .kind = .pooled, .layer = j };
    }
    return .{ .kind = .whole };
}

/// Where each plan part sits in a lease: back to back, or `padded` with latent rows and pooled keys on a new page.
pub const Lay = struct { at: [max_parts]usize = undefined, end: usize = 0 };

pub fn lay(plan: *const layout.Plan, st: *const State, padded: bool) Lay {
    var l: Lay = .{};
    for (plan.parts[0..plan.count], 0..) |p, i| {
        const k = which(st, p.live).kind;
        if (padded and (k == .latent or k == .pooled)) l.end = std.mem.alignForward(usize, l.end, pages.page);
        l.at[i] = l.end;
        l.end += p.len;
    }
    return l;
}

/// cuda_snapshots.planBytes under the knob: the padded lease's whole pages.
pub fn size(st: *const State, c: anytype, capacity: usize, at: usize, head: ?usize) u64 {
    const plan = layout.Plan.sized(st, c, capacity, at, head) catch return std.math.maxInt(u64);
    return pages.rounded(lay(&plan, st, true).end) catch std.math.maxInt(u64);
}

/// The copies of one keep or restore: a memcpy each, or (TF_GLM_OVN_SNAP_GATHER) tf_glm_snap_copy batches.
const Mover = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    g: ?*const gather.Gather,
    n: usize = 0,
    bytes: u64 = 0,

    fn of(pool: *const pages.Pool, stream: cuda.Stream) !Mover {
        const arena = pool.arena orelse return error.NoSnapshotRoom;
        return .{ .d = arena.d, .stream = stream, .g = if (pool.gather) |*g| g else null };
    }

    fn one(m: *Mover, src: u64, dst: u64, len: usize) !void {
        m.bytes += len;
        const g = m.g orelse return m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, m.stream.handle), "cuMemcpyDtoDAsync");
        g.host[m.n] = .{ .src = src, .dst = dst, .len = len };
        m.n += 1;
        if (m.n == gather.batch) try m.flush();
    }

    fn flush(m: *Mover) !void {
        if (m.n > 0) try m.g.?.flush(m.stream, m.n);
        m.n = 0;
    }
};

/// Bytes [lo, hi) of plan part `part`, at `at` in the lease: into the lease (`into`) or back out of it.
fn range(m: *Mover, pool: *const pages.Pool, lease: *const pages.Lease, plan: *const layout.Plan, part: layout.Part, at: usize, lo: usize, hi: usize, into: bool) !void {
    const arena = (pool.arena orelse return error.NoSnapshotRoom).ptr;
    if (hi > part.len or at + hi > lease.ids.items.len * pages.page) return error.SnapshotSize;
    var done = lo;
    while (done < hi) {
        const x = at + done;
        const held = arena + @as(u64, lease.ids.items[x / pages.page]) * pages.page + x % pages.page;
        const live = plan.piece(part, done);
        const n = @min(@min(live.len, pages.page - x % pages.page), hi - done);
        if (into) try m.one(live.live, held, n) else try m.one(held, live.live, n);
        done += n;
    }
}

/// A cache's pooled keys at `pos`: the whole pools a missing row falls in, then its partial pool and spare row.
fn pooled(m: *Mover, pool: *const pages.Pool, lease: *const pages.Lease, plan: *const layout.Plan, part: layout.Part, at: usize, lack: []const lin.Range, pos: usize, row: usize) !void {
    const whole = pos / sparse.pool;
    var next: usize = 0;
    for (lack) |r| {
        const lo = @max(next, r.lo / sparse.pool);
        const hi = @min(whole, (r.hi + sparse.pool - 1) / sparse.pool);
        if (lo < hi) try range(m, pool, lease, plan, part, at, lo * row, hi * row, false);
        next = @max(next, hi);
    }
    try range(m, pool, lease, plan, part, at, whole * row, part.len, false);
}

/// A lineage of rows `content` to `pos`, owned by `gpa`.
fn kept(gpa: std.mem.Allocator, content: []const lin.Seg, pos: usize, padded: bool) !*lin.Kept {
    var tmp: [lin.max_segs]lin.Seg = undefined;
    var o: lin.Out = .{ .buf = &tmp };
    o.copy(content, pos);
    const segs = try gpa.dupe(lin.Seg, o.done());
    errdefer gpa.free(segs);
    const k = try gpa.create(lin.Kept);
    k.* = .{ .segs = segs, .padded = padded };
    return k;
}

/// A state's lineage; one that has none (a disk read, a keep without it) takes a tag of its own over its rows.
fn keptOf(e: anytype, s: *types.Snap) !*lin.Kept {
    if (s.lin == null) s.lin = try kept(e.gpa, &.{}, 0, false);
    const k = s.lin.?;
    if (k.segs.len == 0 and s.pos > 0) {
        const one = try e.gpa.alloc(lin.Seg, 1);
        one[0] = .{ .end = @intCast(s.pos), .tag = lin.fresh() };
        e.gpa.free(k.segs);
        k.segs = one;
    }
    return k;
}

/// cuda_snapshots.free: the state's lineage goes with it.
pub fn forget(e: anytype, s: *types.Snap) void {
    const k = s.lin orelse return;
    e.gpa.free(k.segs);
    e.gpa.destroy(k);
    s.lin = null;
}

/// cuda_snapshots.restore under the knob: the rows the bound sequence holds of the state stay; the rest is copied.
pub fn restore(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.len == 0) return error.UnsealedState;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const st = &e.cur.st;
    const plan = try layout.Plan.init(st, &e.c, s.pos, if (e.mtp) s.mtp_len else null);
    if (plan.len != s.len) return error.SnapshotSize;
    const k = try keptOf(e, s);
    st.lin.settle(st.pos);
    var buf: [max_ranges]lin.Range = undefined;
    const lack = lin.missing(st.lin.content(), k.segs, s.pos, &buf);
    const l = lay(&plan, st, k.padded);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot restore did not drain");
        st.lin.forget(); // some rows are half copied
    }
    var m = try Mover.of(pool, e.stream);
    const pk_row = sparse.pooledRow(e.c.i_dim);
    for (plan.parts[0..plan.count], 0..) |part, i| switch (which(st, part.live).kind) {
        .latent, .scales => {
            const row = part.len / s.pos;
            for (lack) |r| try range(&m, pool, &s.lease, &plan, part, l.at[i], r.lo * row, r.hi * row, false);
        },
        .pooled => try pooled(&m, pool, &s.lease, &plan, part, l.at[i], lack, s.pos, pk_row),
        .whole => try range(&m, pool, &s.lease, &plan, part, l.at[i], 0, part.len, false),
    };
    try m.flush();
    try st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
    e.cur.in_head = 0;
    try e.stream.synchronize();
    st.lin.restored(k.segs, s.pos);
    stats.copied += m.bytes;
    stats.skipped += s.len - m.bytes;
}

/// The kept state whose pages hold the most of rows `segs` to `pos`, with where its parts sit.
const Source = struct { snap: *const types.Snap, rows: usize, lay: Lay, latent: [max_layers]usize, pooled: [max_layers]usize };

fn source(e: anytype, segs: []const lin.Seg, pos: usize) !?Source {
    var best: ?*const types.Snap = null;
    var best_id: u64 = 0;
    var rows: usize = 0;
    var it = e.snaps.iterator();
    while (it.next()) |x| {
        const s = x.value_ptr;
        const k = s.lin orelse continue;
        if (!k.padded or s.len == 0) continue;
        const d = lin.firstDiff(segs, k.segs, 0, @min(pos, s.pos));
        if (d > rows or (d == rows and d > 0 and x.key_ptr.* < best_id)) {
            best = s;
            best_id = x.key_ptr.*;
            rows = d;
        }
    }
    const s = best orelse return null;
    const st = &e.cur.st;
    const plan = try layout.Plan.sized(st, &e.c, e.capacity, s.pos, if (e.mtp) s.mtp_len else null);
    var out: Source = .{ .snap = s, .rows = rows, .lay = lay(&plan, st, true), .latent = undefined, .pooled = undefined };
    for (plan.parts[0..plan.count], 0..) |p, i| {
        const w = which(st, p.live);
        if (w.kind == .latent) out.latent[w.layer] = i;
        if (w.kind == .pooled) out.pooled[w.layer] = i;
    }
    return out;
}

/// Whole pages at a latent part's start (`row` bytes a row), or a pooled part's (`row` bytes a pool), that hold only
/// rows below `rows`: pools below it are whole, the one it falls in is not.
pub fn sharedPages(pools: bool, rows: usize, row: usize) usize {
    return (if (pools) rows / sparse.pool else rows) * row / pages.page;
}

/// cuda_snapshots.keepAs under the knob: whole pages an earlier state holds of these rows are shared, the rest copied.
pub fn keepAs(e: anytype, id: u64) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    const st = &e.cur.st;
    if (st.pending != 0 or st.mtp_len > st.pos + window_rows) return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const plan = try layout.Plan.init(st, &e.c, st.pos, if (e.mtp) st.mtp_len else null);
    const l = lay(&plan, st, true);
    const win = snapshots.window(e);
    const state_len = if (win != null) try pages.rounded(l.end) else l.end; // the ring starts on a page
    var lease = try pool.take(state_len + @as(usize, if (win != null) ring_bytes else 0));
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, st.cur);
    errdefer e.gpa.free(cur);
    st.lin.settle(st.pos);
    const k = try kept(e.gpa, st.lin.content(), st.pos, true);
    errdefer {
        e.gpa.free(k.segs);
        e.gpa.destroy(k);
    }
    var m = try Mover.of(pool, e.stream);
    const src = try source(e, k.segs, st.pos);
    const pk_row = sparse.pooledRow(e.c.i_dim);
    var shared: u64 = 0;
    for (plan.parts[0..plan.count], 0..) |part, i| {
        const w = which(st, part.live);
        var from: usize = 0;
        if (src) |x| if (w.kind == .latent or w.kind == .pooled) {
            const row = if (w.kind == .latent) part.len / st.pos else pk_row;
            const there = x.lay.at[if (w.kind == .latent) x.latent[w.layer] else x.pooled[w.layer]];
            const n = sharedPages(w.kind == .pooled, x.rows, row);
            for (0..n) |q| pool.share(&lease, l.at[i] / pages.page + q, x.snap.lease.ids.items[there / pages.page + q]);
            from = n * pages.page;
            shared += from;
        };
        try range(&m, pool, &lease, &plan, part, l.at[i], from, part.len, true);
    }
    try m.flush();
    if (win) |w| try snapshots.moveFlat(pool, e.stream, &lease, state_len, w.ring, ring_bytes, true);
    try e.stream.synchronize();
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = plan.len, .pos = st.pos, .mtp_len = st.mtp_len, .cur = cur, .bases = .{ st.base, st.mtp_base }, .win_from = if (win) |w| w.from else null, .lin = k });
    stats.copied += m.bytes;
    stats.shared += shared;
}

/// TF_GLM_OVN_ONE_CHUNK's keep from inside a chunk (a knob-off lease), with the rows it holds of the live lineage.
pub fn keepInside(e: anytype, id: u64, mark: usize) !void {
    try @import("cuda_one_chunk_keep.zig").keepAt(e, id, mark);
    const s = e.snaps.getPtr(id).?;
    e.cur.st.lin.settle(e.cur.st.pos);
    s.lin = kept(e.gpa, e.cur.st.lin.content(), s.pos, false) catch null; // without one its first restore copies all
}

/// cuda_snapshots.blankAs under the knob: a padded lease, its lineage unknown until its first restore.
pub fn blankAs(e: anytype, id: u64, at: usize) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    var lease = try pool.take(snapshots.planBytes(e, at)); // a disk image holds no drafter window
    errdefer pool.give(&lease);
    const cur = try e.gpa.alloc(u1, e.cur.st.cur.len);
    errdefer e.gpa.free(cur);
    @memset(cur, 0);
    const k = try kept(e.gpa, &.{}, 0, true);
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = 0, .pos = at, .mtp_len = 0, .cur = cur, .bases = .{ 0, 0 }, .lin = k });
}

/// The plan of kept state `s` (its parts' lengths) and its padding, when its lease is padded.
fn padding(e: anytype, s: *const types.Snap) !?struct { plan: layout.Plan, flat: Lay, pad: Lay } {
    const k = s.lin orelse return null;
    if (!k.padded) return null;
    const st = &e.cur.st;
    const plan = try layout.Plan.sized(st, &e.c, e.capacity, s.pos, if (e.mtp) s.mtp_len else null);
    return .{ .plan = plan, .flat = lay(&plan, st, false), .pad = lay(&plan, st, true) };
}

/// Where KEEPWIN's ring sits in a kept state's lease: on the page past its rows.
pub fn end(e: anytype, s: *const types.Snap) !usize {
    const p = try padding(e, s) orelse return pages.rounded(s.len);
    return pages.rounded(p.pad.end);
}

/// cuda_disk_ranks.copier: the image's lease segments (knob-off offsets) moved onto a padded lease.
pub fn remap(e: anytype, s: *const types.Snap, segs: *std.ArrayList(image.Seg)) !void {
    const p = try padding(e, s) orelse return;
    var out: std.ArrayList(image.Seg) = .empty;
    errdefer out.deinit(e.gpa);
    for (segs.items) |g| {
        const base = switch (g.from) {
            .lease => |x| x,
            else => {
                try out.append(e.gpa, g);
                continue;
            },
        };
        var done: u64 = 0;
        while (done < g.len) {
            const off = base + done;
            const i = for (p.plan.parts[0..p.plan.count], 0..) |part, j| {
                if (off >= p.flat.at[j] and off < p.flat.at[j] + part.len) break j;
            } else return error.SnapshotShape;
            const n = @min(g.len - done, p.flat.at[i] + p.plan.parts[i].len - off);
            try out.append(e.gpa, .{ .img = g.img + done, .len = n, .from = .{ .lease = p.pad.at[i] + (off - p.flat.at[i]) } });
            done += n;
        }
    }
    segs.deinit(e.gpa);
    segs.* = out;
}
