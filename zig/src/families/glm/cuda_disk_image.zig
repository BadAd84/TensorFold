//! A GLM kept state's disk image on one rank: its parts at a position, and where each image byte sits in its lease.
const std = @import("std");
const cuda = @import("cuda");
const st_mod = @import("cuda_state.zig");
const sparse = @import("cuda_sparse.zig");
const Config = @import("config.zig").Config;
const pages = @import("cuda_snapshot_pages.zig");
const State = st_mod.State;
const window_rows = st_mod.window_rows;

/// The image's first bytes: the format, the position, the head's rows, both ring bases and the KDA copy bits.
pub const Words = struct {
    at: u64,
    mtp_len: u64,
    base: u64,
    mtp_base: u64,
    cur: u64,

    pub const bytes: u64 = 48;
    /// 1: bf16 pooled keys; 2: TF_GLM_OVN_PK8's kv8 rows (a state saved in one is never read in the other).
    fn format() u64 {
        return if (sparse.pk8) 2 else 1;
    }

    pub fn encode(w: Words) [6]u64 {
        return .{ format(), w.at, w.mtp_len, w.base, w.mtp_base, w.cur };
    }

    /// Words read back: refused unless the format is this one and the state stands at `at`.
    pub fn decode(x: [6]u64, at: u32) !Words {
        if (x[0] != format() or x[1] != at) return error.DiskWords;
        return .{ .at = x[1], .mtp_len = x[2], .base = x[3], .mtp_base = x[4], .cur = x[5] };
    }
};

/// A part as the rows store takes it: `row` bytes for each `per` tokens from position 0, or the state's own (row 0).
pub const Part = struct { bytes: u64, row: u32 = 0, per: u32 = 1 };

/// `len` image bytes from `img`: the words, `lease` bytes into the state's lease, or zeros the lease does not hold.
pub const Seg = struct { img: u64, len: u64, from: union(enum) { words, lease: u64, zeros } };

/// What a rank's kept state holds, read from a sequence of the engine (its addresses are not used past here).
pub const Shape = struct {
    prefix: u64, // the sequence's own fields before its caches: KDA conv and states, a window's saved rows
    tail: u64, // its fields after them: the head's position and first-draft row
    layers: u32, // DSA layers, whose rows are shared by position
    latent: u32, // a latent row's bytes
    scales: bool, // fp8 latent rows: a 4-byte scale each
    ring_rows: u32,
    ring_row: u32,
    pk_rows: u32,
    pooled_row: u32,
    capacity: u32,
    head: bool, // the MTP head's caches are kept (drafting)
    kda: Kda = .{}, // the prefix's fields a state holds (canon): the rest is a decode window's scratch and padding
    hidden: u64 = 0, // the tail's first-draft row: its offset in the tail
    hidden_len: u64 = 0,

    pub fn of(st: *const State, c: *const Config, capacity: usize, head: bool) !Shape {
        if (st.rings.len == 0) return error.SnapshotShape;
        const base = st.arena.buf.ptr;
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        const end = if (st.table != 0) st.table else base + st.arena.buf.len;
        if (first < base or st.mtp_pos_dev < first or end < st.mtp_pos_dev) return error.SnapshotShape;
        const fp8 = st.mtp_caches.ls != 0;
        const kda: Kda = .{ .pos = st.pos_dev -% base, .conv = st.conv -% base, .conv_len = st.kda_layers * 3 * st.conv_c * 2, .rec = st.rec -% base, .rec_layer = st.rec_layer, .layers = @intCast(st.kda_layers) };
        return .{
            .prefix = first - base,
            .tail = end - st.mtp_pos_dev,
            .layers = @intCast(st.caches.len),
            .latent = c.kv_lora * @as(u32, if (fp8) 1 else 2),
            .scales = fp8,
            .ring_rows = @intCast(st.ring_rows),
            .ring_row = @intCast(st.ring_row),
            .pk_rows = @intCast(capacity / sparse.pool + 2),
            .pooled_row = @intCast(sparse.pooledRow(c.i_dim)),
            .capacity = @intCast(capacity),
            .head = head,
            .kda = kda,
            .hidden = st.prompt_hidden -% st.mtp_pos_dev,
            .hidden_len = c.hidden * 2,
        };
    }

    /// Rows the image keeps of the head's caches at `at`: a verify window past it, as the cache charges a state.
    pub fn headRows(sh: Shape, at: u32) u32 {
        return @min(sh.capacity, at + window_rows);
    }
};

/// One cache's parts in the lease's order, for `n` rows: latent, scales, the two index rings, the pooled keys.
fn cache(sh: Shape, n: u64) [5]u64 {
    return .{
        n * sh.latent,
        if (sh.scales) n * 4 else 0,
        @min(n, sh.ring_rows) * sh.ring_row,
        @min(n, sh.ring_rows) * sh.ring_row,
        @min(n / sparse.pool + 2, sh.pk_rows) * sh.pooled_row,
    };
}

/// A prefix's fields a state holds: its position word, the conv states, and both copies of each KDA layer's state.
pub const Kda = struct { pos: u64 = 0, conv: u64 = 0, conv_len: u64 = 0, rec: u64 = 0, rec_layer: u64 = 0, layers: u32 = 0 };

const Match = @import("lanes").rows_image.Match;

/// TF_GLM_OVN_DISK_CARRY's replay: which bytes of a stored image at `at` (words `s`) answer a fresh one's (`f`).
pub fn canon(sh: Shape, at: u32, s: Words, f: Words, out: *std.ArrayList(Match), a: std.mem.Allocator) !void {
    if (sh.tail < 4 or sh.hidden + sh.hidden_len > sh.tail) return error.SnapshotShape;
    for ([_]Words{ s, f }) |w| {
        if (!holds(w.base, at, @min(at, sh.ring_rows))) return error.DiskWords;
        if (sh.head and (w.mtp_len > sh.headRows(at) or !holds(w.mtp_base, w.mtp_len, @min(w.mtp_len, sh.ring_rows)))) return error.DiskWords;
    }
    // the words' format, position and head rows; the ring bases and KDA copy bits only say where a state's bytes sit
    try same(0, 24, out, a);
    if (s.mtp_len != f.mtp_len) return; // another head: its rows word differs, and nothing past it lines up
    try prefix(sh, Words.bytes, s.cur, f.cur, out, a);
    var img: u64 = Words.bytes + sh.prefix;
    for (0..sh.layers) |_| img = try cacheMatches(sh, img, cache(sh, at), at, s.base, f.base, out, a);
    if (sh.head) img = try cacheMatches(sh, img, cache(sh, sh.headRows(at)), s.mtp_len, s.mtp_base, f.mtp_base, out, a);
    try same(img, 4, out, a); // the head's position word
    try same(img + sh.hidden, sh.hidden_len, out, a); // the first-draft row; the rest of the tail is padding
}

/// The first held byte where two images of one state at `at` differ (all words, then canon's matches); null: none.
pub fn heldDiff(sh: Shape, at: u32, p: []const u8, q: []const u8, a: std.mem.Allocator) !?u64 {
    if (p.len != q.len or p.len < Words.bytes) return 0;
    if (std.mem.indexOfDiff(u8, p[0..Words.bytes], q[0..Words.bytes])) |i| return i;
    const w = try Words.decode(std.mem.bytesToValue([6]u64, p[0..Words.bytes]), at);
    var ms: std.ArrayList(Match) = .empty;
    defer ms.deinit(a);
    try canon(sh, at, w, w, &ms, a);
    var first: ?u64 = null;
    for (ms.items) |m| {
        if (m.off + m.len > p.len) return error.SnapshotShape;
        const x = p[@intCast(m.off)..][0..@intCast(m.len)];
        const i = std.mem.indexOfDiff(u8, x, q[@intCast(m.off)..][0..@intCast(m.len)]) orelse continue;
        if (first == null or m.off + i < first.?) first = m.off + i;
    }
    return first;
}

/// A prefix at image offset `img` (rank 0's, or another rank's in the store): its fields, each KDA layer's live copy.
pub fn prefix(sh: Shape, img: u64, s_cur: u64, f_cur: u64, out: *std.ArrayList(Match), a: std.mem.Allocator) !void {
    const k = sh.kda;
    if (k.layers > 64 or k.pos + 4 > sh.prefix or k.conv + k.conv_len > sh.prefix or k.rec + 2 * k.layers * k.rec_layer > sh.prefix) return error.SnapshotShape;
    try same(img + k.pos, 4, out, a);
    try same(img + k.conv, k.conv_len, out, a);
    // the live copy is the one a layer's last pass wrote (cuda_forward: state_out is the other, then `cur` flips to it)
    for (0..k.layers) |li| {
        const sb = (s_cur >> @intCast(li)) & 1;
        const fb = (f_cur >> @intCast(li)) & 1;
        try out.append(a, .{ .off = img + k.rec + (sb * k.layers + li) * k.rec_layer, .len = k.rec_layer, .fresh = img + k.rec + (fb * k.layers + li) * k.rec_layer });
    }
}

/// One cache at `img` (parts `size`) holding `n` rows: rows by position, the rings' rows both hold, completed pools.
fn cacheMatches(sh: Shape, img: u64, size: [5]u64, n: u64, s_base: u64, f_base: u64, out: *std.ArrayList(Match), a: std.mem.Allocator) !u64 {
    try same(img, n * sh.latent, out, a);
    try same(img + size[0], if (sh.scales) n * 4 else 0, out, a);
    // a ring holds position p at row p - base (State.setBases); pool_keys, its one reader, starts at a pass's own pool
    const lo = n / sparse.pool * sparse.pool; // so a completed pool's rows are never read again (both bases below)
    var ring = img + size[0] + size[1];
    for (0..2) |_| {
        if (lo < n) try out.append(a, .{ .off = ring + (lo - s_base) * sh.ring_row, .len = (n - lo) * sh.ring_row, .fresh = ring + (lo - f_base) * sh.ring_row });
        ring += size[2];
    }
    try same(ring, n / sparse.pool * sh.pooled_row, out, a); // pool_keys writes completed pools alone, none past them
    return ring + size[4];
}

/// Whether a ring based at `base` holds positions [base, n) within its `rows` rows, based on a pool (State.rebase).
fn holds(base: u64, n: u64, rows: u64) bool {
    return base % sparse.pool == 0 and base <= n and n - base <= rows;
}

/// `len` bytes at `off` answered at the same place.
fn same(off: u64, len: u64, out: *std.ArrayList(Match), a: std.mem.Allocator) !void {
    if (len > 0) try out.append(a, .{ .off = off, .len = len, .fresh = off });
}

/// The image at `at` (any row, as layout.Plan keeps it): words, prefix, each DSA layer's rows, the head, the tail.
pub fn parts(sh: Shape, at: u32, out: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    if (at == 0 or at > sh.capacity) return error.SnapshotShape;
    try out.appendSlice(a, &.{ .{ .bytes = Words.bytes }, .{ .bytes = sh.prefix } });
    const pooled: u64 = at / sparse.pool;
    for (0..sh.layers) |_| {
        const c = cache(sh, at);
        try out.append(a, .{ .bytes = c[0], .row = sh.latent });
        if (sh.scales) try out.append(a, .{ .bytes = c[1], .row = 4 });
        try out.appendSlice(a, &.{ .{ .bytes = c[2] }, .{ .bytes = c[3] } });
        // whole pools are rows by position; the partial pool (at % pool rows in) and the spare row stay the state's own
        try out.appendSlice(a, &.{ .{ .bytes = pooled * sh.pooled_row, .row = sh.pooled_row, .per = sparse.pool }, .{ .bytes = c[4] - pooled * sh.pooled_row } });
    }
    if (sh.head) for (cache(sh, sh.headRows(at))) |n| if (n > 0) try out.append(a, .{ .bytes = n });
    try out.append(a, .{ .bytes = sh.tail });
}

/// The image's segments in order for a state at `at` whose head holds `mtp_len` rows (0 without a head).
pub fn segments(sh: Shape, at: u32, mtp_len: u64, out: *std.ArrayList(Seg), a: std.mem.Allocator) !void {
    if (sh.head and mtp_len > sh.headRows(at)) return error.SnapshotShape;
    var img: u64 = Words.bytes;
    var lease: u64 = 0;
    try out.append(a, .{ .img = 0, .len = Words.bytes, .from = .words });
    const f = front(sh, at);
    try out.append(a, .{ .img = img, .len = f, .from = .{ .lease = 0 } });
    img += f;
    lease += f;
    if (sh.head) for (cache(sh, sh.headRows(at)), cache(sh, mtp_len)) |full, held| {
        if (held > 0) try out.append(a, .{ .img = img, .len = held, .from = .{ .lease = lease } });
        if (full > held) try out.append(a, .{ .img = img + held, .len = full - held, .from = .zeros });
        img += full;
        lease += held;
    };
    if (sh.tail > 0) try out.append(a, .{ .img = img, .len = sh.tail, .from = .{ .lease = lease } });
}

/// The image bytes after the words that sit in the lease as in the image: the prefix and every DSA layer's parts.
pub fn front(sh: Shape, at: u32) u64 {
    var n: u64 = sh.prefix;
    for (0..sh.layers) |_| for (cache(sh, at)) |x| {
        n += x;
    };
    return n;
}

/// The lease bytes of a state at `at` whose head holds `mtp_len` rows (layout.Plan's length for it).
pub fn leaseBytes(sh: Shape, at: u32, mtp_len: u64) u64 {
    var n: u64 = sh.prefix + sh.tail;
    for (0..sh.layers) |_| for (cache(sh, at)) |x| {
        n += x;
    };
    if (sh.head) for (cache(sh, mtp_len)) |x| {
        n += x;
    };
    return n;
}

/// Rank 0's image bytes at `at` (its head counted at a verify window past `at`).
pub fn imageBytes(sh: Shape, at: u32) u64 {
    var n: u64 = Words.bytes + front(sh, at) + sh.tail;
    if (sh.head) for (cache(sh, sh.headRows(at))) |x| {
        n += x;
    };
    return n;
}

/// `len` bytes at `off` of rank `rank`'s image; `shared`: past rank 0's prefix, the same bytes every rank holds.
pub const Piece = struct { rank: u32, off: u64, len: u64, shared: bool };

/// The store's image at `at`: rank 0's, then each other rank's prefix (rank 0's bytes stand for all the rest).
pub fn storeParts(sh: Shape, prefixes: []const u64, at: u32, out: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    try parts(sh, at, out, a);
    for (prefixes[1..]) |p| try out.append(a, .{ .bytes = p });
}

/// Store image bytes [off, off + len) split by rank: rank 0's image, then rank r's prefix after its words.
pub fn route(sh: Shape, prefixes: []const u64, at: u32, off: u64, len: u64, out: *std.ArrayList(Piece), a: std.mem.Allocator) !void {
    const own = Words.bytes + sh.prefix; // rank 0's bytes no other rank holds
    var start: u64 = 0; // the section's first store byte
    for (prefixes, 0..) |p, r| {
        const size = if (r == 0) imageBytes(sh, at) else p;
        defer start += size;
        const lo = @max(off, start);
        const hi = @min(off + len, start + size);
        if (lo >= hi) continue;
        if (r > 0) {
            try out.append(a, .{ .rank = @intCast(r), .off = Words.bytes + lo - start, .len = hi - lo, .shared = false });
            continue;
        }
        if (lo < own) try out.append(a, .{ .rank = 0, .off = lo, .len = @min(hi, own) - lo, .shared = false });
        if (hi > own) try out.append(a, .{ .rank = 0, .off = @max(lo, own), .len = hi - @max(lo, own), .shared = true });
    }
    if (off + len > start) return error.SnapshotShape;
}

/// Copies between `host` and image bytes [off, off + host.len) of a lease: out (lease to host) or in.
pub const Copy = struct {
    d: *const cuda.Driver,
    pool: *const pages.Pool,
    lease: *const pages.Lease,
    segs: []const Seg,
    words: *[6]u64,

    pub fn run(c: Copy, off: u64, host: []u8, out: bool) !void {
        const end = off + host.len;
        for (c.segs) |s| {
            const lo = @max(off, s.img);
            const hi = @min(end, s.img + s.len);
            if (lo >= hi) continue;
            const h = host[@intCast(lo - off)..][0..@intCast(hi - lo)];
            switch (s.from) {
                .words => {
                    const w = std.mem.asBytes(c.words)[@intCast(lo - s.img)..][0..h.len];
                    if (out) @memcpy(h, w) else @memcpy(w, h);
                },
                .zeros => if (out) @memset(h, 0),
                .lease => |at| try c.leased(at + (lo - s.img), h, out),
            }
        }
    }

    /// The device address of image bytes [off, off + n) (n > 0): one lease segment inside one snapshot page.
    pub fn device(c: Copy, off: u64, n: u64) !u64 {
        const arena = c.pool.arena orelse return error.NoSnapshotRoom;
        for (c.segs) |s| {
            if (off < s.img or off + n > s.img + s.len) continue;
            const at = switch (s.from) {
                .lease => |l| l + (off - s.img),
                else => return error.SnapshotShape,
            };
            if (at % pages.page + n > pages.page or at + n > c.lease.ids.items.len * pages.page) return error.SnapshotSize;
            return arena.ptr + @as(u64, c.lease.ids.items[@intCast(at / pages.page)]) * pages.page + at % pages.page;
        }
        return error.SnapshotShape;
    }

    /// Lease bytes from `at`, a page at a time (each page is a run of the snapshot arena).
    fn leased(c: Copy, at: u64, host: []u8, out: bool) !void {
        const arena = c.pool.arena orelse return error.NoSnapshotRoom;
        if (at + host.len > c.lease.ids.items.len * pages.page) return error.SnapshotSize;
        var done: usize = 0;
        while (done < host.len) {
            const x = at + done;
            const n = @min(host.len - done, pages.page - x % pages.page);
            const dev = arena.ptr + @as(u64, c.lease.ids.items[@intCast(x / pages.page)]) * pages.page + x % pages.page;
            const p = host[done..][0..n];
            if (out) {
                try c.d.check(c.d.api.cuMemcpyDtoH_v2(p.ptr, dev, n), "cuMemcpyDtoH");
            } else try c.d.check(c.d.api.cuMemcpyHtoD_v2(dev, p.ptr, n), "cuMemcpyHtoD");
            done += n;
        }
    }
};

test {
    _ = @import("cuda_disk_image_test.zig");
    _ = @import("cuda_disk_canon_test.zig");
}
