//! TF_GLM_OVN_LINEAGE on the CPU (test-glm-snapshots-host): turns, forks and page moves on a window and on pages, run
//! with the knob and without through the recording driver; every restored read set and kept state equal byte for byte.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const t = std.testing;
const lin = glm.lineage;
const State = glm.forward.State;
const Engine = glm.engine.Engine;
const Caches = glm.dsa_block.Caches;

/// Byte `k` of row `r` as a write salted `salt` leaves it.
fn byte(salt: usize, r: usize, k: usize) u8 {
    return @truncate(salt * 131 + r * 7 + r / 251 + k * 13);
}

/// Row `r`'s first byte in a cache of `width`-byte rows, `per` rows a page when the cache is on pages.
fn rowAt(st: *const State, x: Caches, base: u64, r: usize, width: usize, per: usize) u64 {
    return if (x.pt == 0) base + r * width else base + @as(u64, st.pages[r / per]) * x.ps + (r % per) * width;
}

/// Rows [lo, hi) of cache `x` (ring pair `ring`, based at `base`) as a pass writes them, and the pools it completes.
fn rows(st: *const State, c: glm.config.Config, x: Caches, ring: usize, base: usize, lo: usize, hi: usize, salt: usize) void {
    const lw: usize = c.kv_lora * @as(usize, if (x.ls == 0) 2 else 1);
    for (lo..hi) |r| {
        for (rec.bytes(rowAt(st, x, x.lc, r, lw, 512), lw), 0..) |*b, k| {
            b.* = byte(salt, r, k);
        }
        if (x.ls != 0) for (rec.bytes(rowAt(st, x, x.ls, r, 4, 512), 4), 0..) |*b, k| {
            b.* = byte(salt + 1, r, k);
        };
        for (0..2) |q| for (rec.bytes(st.rings[2 * ring + q] + (r - base) * st.ring_row, st.ring_row), 0..) |*b, k| {
            b.* = byte(salt + 2 + q, r, k);
        };
    }
    const pw = glm.sparse.pooledRow(c.i_dim);
    var p = lo / 4;
    while (4 * p + 3 < hi and p < x.pk_rows) : (p += 1) { // completed pools only (tf_glm_pool_keys), from their rows
        for (rec.bytes(rowAt(st, x, x.pk, p, pw, 128), pw), 0..) |*b, k| {
            var s: u8 = @truncate(p);
            for (0..4) |q| s +%= rec.bytes(rowAt(st, x, x.lc, 4 * p + q, lw, 512) + k % lw, 1)[0];
            b.* = s;
        }
    }
}

/// A pass of `n` rows from the position (State.ready first, as the forward), `commit` of them committed.
fn write(e: *Engine, salt: usize, n: usize, commit: usize) !void {
    const st = &e.cur.st;
    try st.ready(e.stream, n);
    const pos = st.pos;
    for (st.caches, 0..) |x, j| rows(st, e.c, x, j, st.base, pos, pos + n, salt);
    for (rec.bytes(st.rec, 2 * st.kda_layers * st.rec_layer), 0..) |*b, k| b.* = byte(salt, pos, k);
    for (rec.bytes(st.conv, st.kda_layers * 3 * st.conv_c * 2), 0..) |*b, k| b.* = byte(salt + 5, pos, k);
    st.cur[0] ^= 1;
    if (e.mtp) {
        try st.readyMtp(e.stream, n);
        rows(st, e.c, st.mtp_caches, st.caches.len, st.mtp_base, st.mtp_len, st.mtp_len + n, salt + 9);
        try st.setMtpLen(e.stream, st.mtp_len + commit);
    }
    try st.setPos(e.stream, pos + commit);
}

/// Engine.begin's sequence part (State.reset, through calls the recording driver has).
fn begin(e: *Engine) !void {
    const st = &e.cur.st;
    if (lin.on) st.lin.reset(st.pos);
    @memset(rec.bytes(st.conv, st.kda_layers * 3 * st.conv_c * 2), 0);
    @memset(rec.bytes(st.rec, 2 * st.kda_layers * st.rec_layer), 0);
    const zero: [64]u1 = @splat(0);
    try st.restoreHost(e.stream, 0, 0, zero[0..st.cur.len], .{ 0, 0 });
}

/// The host words a restore sets, after `bytes`.
fn words(a: std.mem.Allocator, out: *std.ArrayList(u8), pos: usize, mtp_len: usize, bases: [2]usize, cur: []const u1) !void {
    for ([_]usize{ pos, mtp_len, bases[0], bases[1] }) |v| try out.appendSlice(a, std.mem.asBytes(&v));
    for (cur) |b| try out.append(a, b);
}

/// The bound sequence's read set at its position, through the knob-off plan.
fn readSet(a: std.mem.Allocator, e: *Engine) ![]u8 {
    const st = &e.cur.st;
    const plan = try glm.snapshots.layout.Plan.init(st, &e.c, st.pos, if (e.mtp) st.mtp_len else null);
    var out: std.ArrayList(u8) = .empty;
    for (plan.parts[0..plan.count]) |p| {
        var done: usize = 0;
        while (done < p.len) {
            const x = plan.piece(p, done);
            try out.appendSlice(a, rec.bytes(x.live, x.len));
            done += x.len;
        }
    }
    try words(a, &out, st.pos, st.mtp_len, .{ st.base, st.mtp_base }, st.cur);
    return out.toOwnedSlice(a);
}

/// A kept state's bytes in the knob-off plan's order, wherever its lease holds them.
fn image(a: std.mem.Allocator, e: *Engine, id: u64) ![]u8 {
    const s = e.snaps.get(id).?;
    const st = &e.cur.st;
    const plan = try glm.snapshots.layout.Plan.sized(st, &e.c, e.capacity, s.pos, if (e.mtp) s.mtp_len else null);
    const l = glm.lineage_copy.lay(&plan, st, if (s.lin) |k| k.padded else false);
    const arena = e.snap_pages.?.arena.?.ptr;
    const P = glm.snapshots.pages.page;
    var out: std.ArrayList(u8) = .empty;
    for (plan.parts[0..plan.count], 0..) |p, i| {
        var done: usize = 0;
        while (done < p.len) {
            const x = l.at[i] + done;
            const n = @min(p.len - done, P - x % P);
            try out.appendSlice(a, rec.bytes(arena + @as(u64, s.lease.ids.items[x / P]) * P + x % P, n));
            done += n;
        }
    }
    try words(a, &out, s.pos, s.mtp_len, s.bases, s.cur);
    return out.toOwnedSlice(a);
}

fn keep(a: std.mem.Allocator, e: *Engine, out: *std.ArrayList([]u8)) !u64 {
    const id = try e.keepState();
    try out.append(a, try image(a, e, id));
    return id;
}

fn restore(a: std.mem.Allocator, e: *Engine, id: u64, out: *std.ArrayList([]u8)) !void {
    try e.restoreState(id);
    try out.append(a, try readSet(a, e));
}

/// A sequence of the fixture's shapes and `cap` rows: a window, or (`kv`) on pages `ids`.
fn sequence(d: *const cuda.Driver, f: *const Fixture, cap: usize, kv: ?u64, page: u64, ids: []const u32) !State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    const at = kv orelse return State.initFormat(t.allocator, d, &w, cap, true);
    var st = try State.initPaged(t.allocator, d, &w, cap, true, .{ .kv = at, .pages = glm.pages.tableLen(cap), .page_bytes = page });
    st.pages = ids;
    st.capacity = ids.len * 512;
    return st;
}

/// A window at the model's index width (bf16 pooled keys of 256 bytes, 32,768 rows' pools a page): keeps past that
/// share a page of pooled keys as well as pages of latent rows.
fn wideArm(a: std.mem.Allocator, on: bool, out: *std.ArrayList([]u8)) !glm.lineage_copy.Stats {
    lin.on = on;
    defer lin.on = false;
    glm.lineage_copy.stats = .{};
    var d = rec.init();
    var f: Fixture = undefined;
    try f.initFormat(&d, 0, 2, false, true);
    defer f.deinit();
    const cap = 34816;
    f.e.c.i_dim = 128;
    f.seq.st.deinit();
    f.seq.st = try sequence(&d, &f, cap, null, 0, &.{});
    f.e.capacity = cap;
    const e = &f.e;
    try e.prepareSnapshots(5 * e.stateBytes(34000));
    try begin(e);
    for (0..8) |i| try write(e, 30 + i, 4096, 4096); // a prompt in chunks: each stays inside the index rings
    try write(e, 40, 232, 232);
    const k1 = try keep(a, e, out); // 33,000 rows
    try write(e, 41, 200, 200);
    const k2 = try keep(a, e, out); // shares k1's pooled page and 8 latent pages
    try begin(e);
    try restore(a, e, k2, out);
    try write(e, 42, 300, 300);
    _ = try keep(a, e, out); // shares k2's
    try restore(a, e, k1, out);
    return glm.lineage_copy.stats;
}

test "GLM snapshots: TF_GLM_OVN_LINEAGE at the model's index width shares pooled-key pages, byte for byte" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var off: std.ArrayList([]u8) = .empty;
    var on: std.ArrayList([]u8) = .empty;
    _ = try wideArm(a, false, &off);
    const s = try wideArm(a, true, &on);
    try t.expectEqual(off.items.len, on.items.len);
    for (off.items, on.items, 0..) |x, y, i| {
        if (!std.mem.eql(u8, x, y)) {
            std.debug.print("lineage wide: record {d} differs at byte {?d} of {d}\n", .{ i, std.mem.indexOfDiff(u8, x, y), x.len });
            return error.TestExpectedEqual;
        }
    }
    try t.expectEqual(@as(u64, 18 * glm.snapshots.pages.page), s.shared); // two keeps: 8 latent and 1 pooled page each
}

/// The scenario on one arm: what every restore left and every keep holds, in order.
fn arm(a: std.mem.Allocator, on: bool, out: *std.ArrayList([]u8)) !glm.lineage_copy.Stats {
    lin.on = on;
    defer lin.on = false;
    glm.lineage_copy.stats = .{};
    var d = rec.init();
    var f: Fixture = undefined;
    try f.initFormat(&d, 0, 2, true, true);
    defer f.deinit();
    const e = &f.e;
    try e.prepareSnapshots(12 * e.stateBytes(12000));
    // a pass before any begin: its rows' writer has no tag, so no other state shares them (the paged one's below)
    try write(e, 20, 4500, 4500);
    _ = try keep(a, e, out);
    // a conversation on a window: a turn kept twice and past a decode window, the next turn resumed, a fork
    try begin(e);
    try write(e, 1, 4000, 4000);
    try write(e, 2, 4000, 4000);
    const k1 = try keep(a, e, out);
    try write(e, 3, 1001, 1001); // a partial pool at the mark
    const k2 = try keep(a, e, out);
    try write(e, 4, 8, 3); // a verify window: rows past its commit are not the turn's
    try write(e, 5, 8, 8);
    const k3 = try keep(a, e, out);
    try begin(e);
    try restore(a, e, k2, out);
    try restore(a, e, k3, out); // no pass between: k2's partial pool and spare row are not k3's
    try restore(a, e, k1, out);
    try write(e, 10, 16, 2); // a verify window: its uncommitted rows are not k3's
    try restore(a, e, k3, out);
    try begin(e);
    try restore(a, e, k2, out); // the warm turn: every row is in place
    try write(e, 6, 2000, 2000);
    const k4 = try keep(a, e, out); // shares the first two pages of latent rows
    try begin(e);
    try restore(a, e, k1, out); // a fork of the first state
    try write(e, 7, 3000, 3000);
    _ = try keep(a, e, out);
    try restore(a, e, k4, out); // back to the other branch: the rows past the fork are copied
    try restore(a, e, k2, out); // no pass between, k2 inside k4's rows
    try restore(a, e, k3, out);
    const k6 = try keep(a, e, out); // a keep right after a restore: all of it shared or copied
    try restore(a, e, k6, out);
    // a sequence on pages: kept, given back, mapped again with two pages swapped, resumed
    const c = e.c;
    const page = 2 * (512 * (c.kv_lora + 4) + 128 * glm.sparse.pooledRow(c.i_dim));
    var kv = try cuda.DeviceBuffer.alloc(&d, 12 * page);
    defer kv.free();
    const first = [_]u32{ 3, 1, 0, 2, 4, 5, 6, 7, 8, 9 };
    const again = [_]u32{ 3, 2, 0, 1, 4, 5, 6, 7, 8, 9 };
    var paged = f.seq;
    paged.st = try sequence(&d, &f, kv.ptr, page, &first);
    defer paged.st.deinit();
    const window = e.cur;
    e.cur = &paged;
    defer e.cur = window;
    try write(e, 21, 4500, 4500); // the other untagged pass: the window's untagged state holds other bytes
    _ = try keep(a, e, out);
    try begin(e);
    try write(e, 8, 1500, 1500);
    const p1 = try keep(a, e, out);
    try write(e, 9, 300, 300);
    const p2 = try keep(a, e, out);
    var tags: [12]lin.PageTag = @splat(.{});
    if (on) lin.closing(&tags, &paged.st.lin, paged.st.pos, paged.st.pages);
    paged.st.pages = &again;
    if (on) lin.mapping(&tags, &paged.st.lin, paged.st.pos, &again, false);
    try begin(e);
    try restore(a, e, p2, out); // pages 3 and 0 hold their rows where they were; 1 and 2 swapped places (two ranges)
    try restore(a, e, p1, out);
    e.cur = window;
    try restore(a, e, p2, out); // onto the window, which holds none of them
    if (on) { // the store sees every lease's pages, as without the knob
        const pool = &e.snap_pages.?;
        var held: usize = 0;
        var it = e.snaps.valueIterator();
        while (it.next()) |s| held += s.lease.ids.items.len;
        try t.expectEqual(pool.arena.?.len, held * glm.snapshots.pages.page + pool.spare());
        try t.expect(pool.pages.available() >= pool.spare() / glm.snapshots.pages.page);
    }
    return glm.lineage_copy.stats;
}

test "GLM snapshots: TF_GLM_OVN_LINEAGE restores and keeps every state byte for byte as without it, copying less" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var off: std.ArrayList([]u8) = .empty;
    var on: std.ArrayList([]u8) = .empty;
    _ = try arm(a, false, &off);
    const s = try arm(a, true, &on);
    try t.expectEqual(off.items.len, on.items.len);
    for (off.items, on.items, 0..) |x, y, i| {
        if (!std.mem.eql(u8, x, y)) {
            std.debug.print("lineage: record {d} differs at byte {?d} of {d}\n", .{ i, std.mem.indexOfDiff(u8, x, y), x.len });
            return error.TestExpectedEqual;
        }
    }
    // the warm turn alone left k2's 9,001 latent rows and their scales in place (one DSA layer here)
    try t.expect(s.skipped >= 9001 * (512 + 4));
    try t.expect(s.shared >= 2 * glm.snapshots.pages.page); // and the keeps shared whole pages of latent rows
}

test "GLM snapshots: lineage keeps share pages of whole pools only, and rows neither lineage knows are copied" {
    const lc = glm.lineage_copy;
    // bf16 pooled keys, 256 bytes a pool: a page holds the pools of 32,768 rows
    try t.expectEqual(@as(usize, 3), lc.sharedPages(true, 4 * 32768 - 1, 256));
    try t.expectEqual(@as(usize, 4), lc.sharedPages(true, 4 * 32768, 256));
    try t.expectEqual(@as(usize, 1), lc.sharedPages(false, 4096, 512)); // fp8 latent rows: 4,096 a page
    try t.expectEqual(@as(usize, 0), lc.sharedPages(false, 4095, 512));
    var out: [4]lin.Range = undefined;
    try t.expectEqualSlices(lin.Range, &.{.{ .lo = 0, .hi = 10 }}, lin.missing(&.{}, &.{}, 10, &out));
    const zero = [_]lin.Seg{.{ .end = 10, .tag = 0 }};
    try t.expectEqualSlices(lin.Range, &.{.{ .lo = 0, .hi = 10 }}, lin.missing(&zero, &zero, 10, &out));
}

test "GLM snapshots: lineage pages for a restore come from the free pages that hold its rows, in row order" {
    const kv = @import("core").kv_pool;
    var pool = try kv.Pool.init(t.allocator, 8, 1);
    defer pool.deinit();
    var tags: [8]lin.PageTag = @splat(.{});
    var l: lin.Live = .{};
    l.reset(0);
    l.wrote(0, 1500);
    l.settle(1500);
    const k: [1]lin.Seg = .{.{ .end = 1500, .tag = l.content()[0].tag }};
    lin.closing(&tags, &l, 1500, &.{ 5, 2, 7 });
    var lease: kv.Lease = .{};
    defer pool.finish(&lease);
    try pool.promise(&lease, 4);
    try lin.takeFor(&pool, &lease, 4, &tags, &k, 1500);
    try t.expectEqualSlices(u32, &.{ 5, 2, 7 }, lease.ids.items[0..3]);
    try t.expect(lease.ids.items[3] != 5 and lease.ids.items[3] != 2 and lease.ids.items[3] != 7);
}
