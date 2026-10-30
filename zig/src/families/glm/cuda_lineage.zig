//! TF_GLM_OVN_LINEAGE: which write left each cache row, so a restore copies only rows the live caches lack and a keep
//! shares the pages of an earlier state that holds the same rows (cuda_lineage_copy). Row r of two lineages holds the
//! same bytes when both name the same tag at r: a tag is one run of writes from one row on, and a committed row is
//! never written again under its tag (writes start at the position, which only a reset or a restore lowers).
const std = @import("std");
const kv = @import("core").kv_pool;

/// TF_GLM_OVN_LINEAGE=1, set once at engine init (cuda_engine_ovn); off: no lineage is read or kept anywhere.
pub var on = false;

/// Rows [previous end, end) hold the rows of write `tag` (0: unknown, which never matches); rows past the last end are
/// unknown.
pub const Seg = struct { end: u32, tag: u64 };

/// Rows a KV page holds (cuda_state.page_rows; latent.chunk).
pub const page_rows = 512;

/// Rows past a restore's position whose pooled keys it set: its partial pool and the spare one (a pass sets only
/// pools it completes, tf_glm_pool_keys: none past its rows).
pub const tail = 8;

/// Segments a sequence's lineage holds (past them its rows count as unknown, which only costs copies).
pub const max_segs = 128;

var next_tag = std.atomic.Value(u64).init(1);

/// A tag no row holds yet.
pub fn fresh() u64 {
    return next_tag.fetchAdd(1, .monotonic);
}

/// The tag of row `row`.
pub fn tagAt(segs: []const Seg, row: usize) u64 {
    for (segs) |s| if (row < s.end) return s.tag;
    return 0;
}

/// The rows a list covers (its last end).
pub fn reach(segs: []const Seg) usize {
    return if (segs.len == 0) 0 else segs[segs.len - 1].end;
}

/// A list written in row order into `buf` from row `base`: equal neighbours merge; a full buffer stops it.
pub const Out = struct {
    buf: []Seg,
    base: usize = 0,
    n: usize = 0,
    full: bool = false,

    fn end(o: *const Out) usize {
        return if (o.n == 0) o.base else o.buf[o.n - 1].end;
    }

    /// Rows up to `to` hold `tag`.
    pub fn push(o: *Out, to: usize, tag: u64) void {
        if (o.full or to <= o.end()) return;
        if (o.n > 0 and o.buf[o.n - 1].tag == tag) {
            o.buf[o.n - 1].end = @intCast(to);
            return;
        }
        if (o.n == o.buf.len) {
            o.full = true; // the rows from here on stay unknown
            return;
        }
        o.buf[o.n] = .{ .end = @intCast(to), .tag = tag };
        o.n += 1;
    }

    /// Rows [end, hi) as `src` holds them (unknown past its reach).
    pub fn copy(o: *Out, src: []const Seg, hi: usize) void {
        const lo = o.end();
        for (src) |s| {
            if (s.end <= lo) continue;
            o.push(@min(s.end, hi), s.tag);
            if (s.end >= hi) break;
        }
        o.push(hi, 0);
    }

    /// The list, without unknown rows at its end.
    pub fn done(o: *Out) []Seg {
        while (o.n > 0 and o.buf[o.n - 1].tag == 0) o.n -= 1;
        return o.buf[0..o.n];
    }
};

/// The first row in [lo, hi) where `a` and `b` differ or either is unknown (hi when none).
pub fn firstDiff(a: []const Seg, b: []const Seg, lo: usize, hi: usize) usize {
    var row = lo;
    var i: usize = 0;
    var j: usize = 0;
    while (row < hi) {
        while (i < a.len and a[i].end <= row) i += 1;
        while (j < b.len and b[j].end <= row) j += 1;
        const ta = if (i < a.len) a[i].tag else 0;
        const tb = if (j < b.len) b[j].tag else 0;
        if (ta == 0 or ta != tb) return row;
        row = @min(a[i].end, b[j].end);
    }
    return hi;
}

/// Rows [lo, hi).
pub const Range = struct { lo: usize, hi: usize };

/// The row ranges below `hi` where `have` lacks `want`'s rows, in order; when `out` (not empty) is full its last runs
/// to `hi`.
pub fn missing(have: []const Seg, want: []const Seg, hi: usize, out: []Range) []Range {
    var n: usize = 0;
    var row: usize = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (row < hi) {
        while (i < have.len and have[i].end <= row) i += 1;
        while (j < want.len and want[j].end <= row) j += 1;
        const ta = if (i < have.len) have[i].tag else 0;
        const tb = if (j < want.len) want[j].tag else 0;
        const to = @min(hi, @min(if (i < have.len) have[i].end else hi, if (j < want.len) want[j].end else hi));
        if (ta == 0 or ta != tb) {
            if (n > 0 and out[n - 1].hi == row) {
                out[n - 1].hi = to;
            } else if (n == out.len) {
                out[n - 1].hi = hi;
                return out[0..n];
            } else {
                out[n] = .{ .lo = row, .hi = to };
                n += 1;
            }
        }
        row = to;
    }
    return out[0..n];
}

/// What a sequence's rows hold now: `segs` as of its last settle, and the writes since (tag `tag` from `from` on,
/// none past `hw`; rows from the position to `hw` may hold uncommitted rows).
pub const Live = struct {
    segs: [max_segs]Seg = undefined,
    n: usize = 0,
    tag: u64 = 0,
    from: usize = 0,
    hw: usize = 0,

    /// Its rows as of the last settle.
    pub fn content(l: *const Live) []const Seg {
        return l.segs[0..l.n];
    }

    /// State.ready: a pass of `rows` rows from `pos` writes them (and the pooled keys of the pools it completes).
    pub fn wrote(l: *Live, pos: usize, rows: usize) void {
        l.hw = @max(l.hw, pos + rows);
    }

    fn assign(l: *Live, lo: usize, hi: usize, tag: u64) void {
        var tmp: [max_segs]Seg = undefined;
        var o: Out = .{ .buf = &tmp };
        o.copy(l.content(), lo);
        o.push(hi, tag);
        o.copy(l.content(), @max(hi, reach(l.content())));
        l.put(o.done());
    }

    fn put(l: *Live, segs: []const Seg) void {
        @memcpy(l.segs[0..segs.len], segs);
        l.n = segs.len;
    }

    /// The writes since the last settle folded in at position `pos`: committed rows are the tag's, the rest unknown.
    pub fn settle(l: *Live, pos: usize) void {
        if (l.hw > l.from) {
            if (pos > l.from) l.assign(l.from, pos, l.tag);
            l.assign(@max(pos, l.from), l.hw, 0);
        }
        l.from = pos;
        l.hw = pos;
    }

    /// Engine.begin (the position goes to 0): the rows stay as they are; the writes that follow take a new tag.
    pub fn reset(l: *Live, pos: usize) void {
        l.settle(pos);
        l.tag = fresh();
        l.from = 0;
        l.hw = 0;
    }

    /// A restore of rows `k` to `at` (settled before it): rows below `at` are k's, its boundary pools unknown.
    pub fn restored(l: *Live, k: []const Seg, at: usize) void {
        var tmp: [max_segs]Seg = undefined;
        var o: Out = .{ .buf = &tmp };
        o.copy(k, at);
        o.push(at + tail, 0);
        o.copy(l.content(), @max(at + tail, reach(l.content())));
        l.put(o.done());
        l.tag = fresh();
        l.from = at;
        l.hw = at;
    }

    /// Every row unknown (a failed restore left some rows half written).
    pub fn forget(l: *Live) void {
        l.n = 0;
        l.hw = l.from;
    }
};

/// What a free KV page holds: page `j`'s rows of a sequence's lineage (j maxInt: nothing known).
pub const PageTag = struct { j: u32 = std.math.maxInt(u32), n: u8 = 0, segs: [4]Seg = undefined };

/// A paged sequence gives its pages back: each page's rows as its settled lineage holds them (its next `mapping`
/// replaces its own).
pub fn closing(tags: []PageTag, l: *Live, pos: usize, ids: []const u32) void {
    l.settle(pos);
    for (ids, 0..) |id, j| {
        const lo = j * page_rows;
        var t: PageTag = .{ .j = @intCast(j) };
        var o: Out = .{ .buf = &t.segs, .base = lo };
        o.copy(l.content(), lo + page_rows);
        t.n = @intCast(o.done().len);
        tags[id] = t;
    }
}

/// A paged sequence maps `ids`: its rows are what those pages hold (none known when they were poisoned).
pub fn mapping(tags: []const PageTag, l: *Live, pos: usize, ids: []const u32, poisoned: bool) void {
    l.from = pos;
    l.hw = pos;
    l.n = 0;
    if (poisoned) return;
    var o: Out = .{ .buf = &l.segs };
    for (ids, 0..) |id, j| {
        const t = tags[id];
        if (t.j == j) o.copy(t.segs[0..t.n], (j + 1) * page_rows) else o.push((j + 1) * page_rows, 0);
    }
    l.n = o.done().len;
}

/// Rank 0: `n` pages for a paged slot whose next restore takes kept rows `k` to `at`; for each of the first rows' pages
/// the free page that holds most of them, else the free list's next.
pub fn takeFor(pool: *kv.Pool, lease: *kv.Lease, n: usize, tags: []const PageTag, k: []const Seg, at: usize) !void {
    const want = @min(n, (at + page_rows - 1) / page_rows);
    const best = try pool.gpa.alloc(u32, want);
    defer pool.gpa.free(best);
    const score = try pool.gpa.alloc(usize, want);
    defer pool.gpa.free(score);
    @memset(score, 0);
    for (pool.free[0..pool.nfree]) |id| {
        const t = tags[id];
        if (t.j >= want) continue;
        const lo = @as(usize, t.j) * page_rows;
        const rows = firstDiff(t.segs[0..t.n], k, lo, @min(lo + page_rows, at)) - lo;
        if (rows > score[t.j]) {
            score[t.j] = rows;
            best[t.j] = id;
        }
    }
    while (lease.ids.items.len < n) {
        const j = lease.ids.items.len;
        if (j < want and score[j] > 0 and try pool.takeFree(lease, best[j])) continue;
        _ = try pool.take(lease);
    }
}

/// A kept state's lineage: its rows to its position, and whether its lease is padded (cuda_lineage_copy.lay).
pub const Kept = struct { segs: []Seg, padded: bool };

test "lineage: settle folds committed rows in and leaves uncommitted ones unknown" {
    var l: Live = .{};
    l.reset(0);
    const t = l.tag;
    l.wrote(0, 100);
    l.settle(90);
    try std.testing.expectEqual(t, tagAt(l.content(), 0));
    try std.testing.expectEqual(t, tagAt(l.content(), 89));
    try std.testing.expectEqual(@as(u64, 0), tagAt(l.content(), 90));
    l.wrote(90, 4);
    l.settle(94);
    try std.testing.expectEqual(@as(usize, 1), l.n); // merged into one run
    try std.testing.expectEqual(@as(u32, 94), l.segs[0].end);
    try std.testing.expectEqual(@as(usize, 0), firstDiff(&.{}, &.{}, 0, 10)); // unknown rows never match
    const zero = [_]Seg{.{ .end = 10, .tag = 0 }};
    try std.testing.expectEqual(@as(usize, 0), firstDiff(&zero, &zero, 0, 10));
}

test "lineage: a restore keeps the kept rows, forgets its boundary pools and keeps the rows past them" {
    var l: Live = .{};
    l.reset(0);
    const a = l.tag;
    l.wrote(0, 1000);
    l.settle(1000);
    var k = [_]Seg{ .{ .end = 300, .tag = a }, .{ .end = 500, .tag = 77 } };
    l.restored(&k, 500);
    try std.testing.expectEqual(a, tagAt(l.content(), 299));
    try std.testing.expectEqual(@as(u64, 77), tagAt(l.content(), 499));
    try std.testing.expectEqual(@as(u64, 0), tagAt(l.content(), 500 + tail - 1));
    try std.testing.expectEqual(a, tagAt(l.content(), 500 + tail));
    try std.testing.expectEqual(a, tagAt(l.content(), 999));
    try std.testing.expectEqual(@as(u64, 0), tagAt(l.content(), 1000));
}

test "lineage: missing rows are the differing and unknown ones, merged" {
    const have = [_]Seg{ .{ .end = 100, .tag = 1 }, .{ .end = 200, .tag = 0 }, .{ .end = 300, .tag = 1 }, .{ .end = 400, .tag = 2 } };
    const want = [_]Seg{ .{ .end = 50, .tag = 1 }, .{ .end = 150, .tag = 3 }, .{ .end = 350, .tag = 1 }, .{ .end = 500, .tag = 2 } };
    var out: [8]Range = undefined;
    const m = missing(&have, &want, 500, &out);
    try std.testing.expectEqual(@as(usize, 3), m.len);
    try std.testing.expectEqual(Range{ .lo = 50, .hi = 200 }, m[0]);
    try std.testing.expectEqual(Range{ .lo = 300, .hi = 350 }, m[1]);
    try std.testing.expectEqual(Range{ .lo = 400, .hi = 500 }, m[2]);
    var one: [1]Range = undefined;
    try std.testing.expectEqual(Range{ .lo = 50, .hi = 500 }, missing(&have, &want, 500, &one)[0]);
    try std.testing.expectEqual(@as(usize, 0), missing(&want, &want, 500, &out).len);
}

test "lineage: pages keep their rows by page index, and a sequence maps only the ones in the same place" {
    var tags: [4]PageTag = @splat(.{});
    var l: Live = .{};
    l.reset(0);
    const a = l.tag;
    l.wrote(0, 700);
    l.settle(700);
    closing(&tags, &l, 700, &.{ 2, 0 });
    try std.testing.expectEqual(@as(u32, 0), tags[2].j);
    try std.testing.expectEqual(@as(u32, 1), tags[0].j);
    mapping(&tags, &l, 0, &.{ 2, 3, 0 }, false); // page 0 sits at index 2 now: its rows are not where they were
    try std.testing.expectEqual(a, tagAt(l.content(), 511));
    try std.testing.expectEqual(@as(u64, 0), tagAt(l.content(), 512));
    try std.testing.expectEqual(@as(u64, 0), tagAt(l.content(), 1100));
    mapping(&tags, &l, 0, &.{ 2, 0 }, true);
    try std.testing.expectEqual(@as(usize, 0), l.n);
}
