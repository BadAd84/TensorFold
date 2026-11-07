//! GLM snapshots of paged sequences: a prefix kept and resumed through its pages, pieces inside a page's span, and a
//! prefix's size while the stream or the engine's own sequence is on pages.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;

/// A paged sequence of the fixture's shape (a DSA and a KDA layer), its caches on the pages at `kv`, `ids` mapped.
fn pagedSeq(d: *const cuda.Driver, f: *const Fixture, kv: u64, page: u64, ids: []const u32) !glm.forward.State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    var st = try glm.forward.State.initPaged(t.allocator, d, &w, 16448, false, .{ .kv = kv, .pages = glm.pages.tableLen(16448), .page_bytes = page });
    st.pages = ids;
    st.capacity = ids.len * 512;
    return st;
}

/// Row `r`'s first byte in a cache of `width`-byte rows, `per` rows a page when the cache is on pages.
fn rowAt(st: *const glm.forward.State, x: glm.dsa_block.Caches, base: u64, r: usize, width: usize, per: usize) u64 {
    return if (x.pt == 0) base + r * width else base + @as(u64, st.pages[r / per]) * x.ps + (r % per) * width;
}

fn same(a: u64, b: u64, len: usize) !void {
    try t.expectEqualSlices(u8, rec.bytes(a, len), rec.bytes(b, len));
}

/// Every field a prefix at `at` reads (the MTP layer's `head` rows), in `a` and in `b`, from either layout.
fn expectSamePrefix(c: glm.config.Config, a: *const glm.forward.State, b: *const glm.forward.State, at: usize, head: ?usize) !void {
    const kv = a.kda_layers * 8 * a.heads * 128;
    try same(a.pos_dev, b.pos_dev, 4);
    try same(a.conv, b.conv, a.kda_layers * 3 * a.conv_c * 2);
    try same(a.rec, b.rec, 2 * a.kda_layers * a.rec_layer);
    try same(a.proj, b.proj, a.kda_layers * a.proj_layer * 2);
    try same(a.save_k, b.save_k, kv * 4);
    try same(a.save_v, b.save_v, kv * 2);
    try same(a.save_g, b.save_g, kv * 4);
    try same(a.save_b, b.save_b, a.kda_layers * 8 * a.heads * 4);
    for (0..a.caches.len + 1) |j| {
        const n = if (j < a.caches.len) at else head orelse break;
        const x = if (j < a.caches.len) a.caches[j] else a.mtp_caches;
        const y = if (j < b.caches.len) b.caches[j] else b.mtp_caches;
        const lat = c.kv_lora * 2;
        for (0..n) |r| try same(rowAt(a, x, x.lc, r, lat, 512), rowAt(b, y, y.lc, r, lat, 512), lat);
        for (0..2) |k| try same(a.rings[2 * j + k], b.rings[2 * j + k], @min(n, a.ring_rows) * a.ring_row);
        for (0..@min(n / 4 + 2, x.pk_rows)) |p| try same(rowAt(a, x, x.pk, p, c.i_dim * 2, 128), rowAt(b, y, y.pk, p, c.i_dim * 2, 128), c.i_dim * 2);
    }
    try same(a.mtp_pos_dev, b.mtp_pos_dev, 4);
    try same(a.prompt_hidden, b.prompt_hidden, c.hidden * 2);
}

test "GLM snapshots: a paged sequence's prefix keeps and resumes through its pages in row order, to and from a window" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| {
        var f: Fixture = undefined;
        try f.init(&d, 0, 2, mtp);
        defer f.deinit();
        const c = f.e.c;
        const page = 2 * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2); // the DSA layer's part and the MTP layer's
        const ids = [_]u32{ 5, 2, 9, 0, 7, 1, 3, 8, 4, 6 };
        var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
        defer kv.free();
        var paged = f.seq;
        paged.st = try pagedSeq(&d, &f, kv.ptr, page, &ids);
        defer paged.st.deinit();
        const window = &f.seq;
        try f.e.prepareSnapshots(3 * f.e.stateBytes(4096));
        for ([_]usize{ 511, 1537, 4096 }) |at| {
            const head: ?usize = if (mtp) at + 7 else null;
            // a window's prefix onto the pages: the paged arena's table stays its own
            f.e.cur = window;
            try f.mark(at, head orelse at, @intCast(at % 251));
            const id = try f.e.keepState();
            @memset(rec.bytes(paged.st.arena.buf.ptr, paged.st.arena.buf.len), 0xE9);
            @memset(rec.bytes(kv.ptr, kv.len), 0xE9);
            f.e.cur = &paged;
            try f.e.restoreState(id);
            try expectSamePrefix(c, &window.st, &paged.st, at, head);
            try t.expectEqual(at, paged.st.pos);
            for (rec.bytes(paged.st.table, ids.len * 4)) |b| try t.expectEqual(@as(u8, 0xE9), b);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
            // the pages' prefix back onto a window
            const back = try f.e.keepState();
            @memset(rec.bytes(window.st.arena.buf.ptr, window.st.arena.buf.len), 0x5C);
            f.e.cur = window;
            try f.e.restoreState(back);
            try expectSamePrefix(c, &paged.st, &window.st, at, head);
            try t.expectEqual(at, window.st.pos);
            f.e.dropState(back);
            f.e.dropped.clearRetainingCapacity();
        }
        f.e.cur = window;
    }
}

test "GLM snapshots: a paged part's piece from inside a page's span keeps its offset, on the page the order names" {
    const ids = [_]u32{ 3, 0, 2 };
    const plan: glm.snapshots.layout.Plan = .{ .ids = &ids };
    const part: glm.snapshots.layout.Part = .{ .live = 1 << 20, .len = 250, .span = 100, .page = 4096 };
    // a snapshot page that ends inside a span leaves the rest of that span to the next piece
    const a = plan.piece(part, 130);
    try t.expectEqual(@as(u64, (1 << 20) + 30), a.live);
    try t.expectEqual(@as(usize, 70), a.len);
    const b = plan.piece(part, 230);
    try t.expectEqual(@as(u64, (1 << 20) + 2 * 4096 + 30), b.live);
    try t.expectEqual(@as(usize, 20), b.len);
}

test "GLM snapshots: while a paged stream is current, a prefix's size is the engine's own sequence's, past its pages" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    f.e.seqs = .empty;
    defer f.e.seqs.deinit(t.allocator);
    try f.e.seqs.append(t.allocator, &f.seq);
    const c = f.e.c;
    const page = 2 * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2);
    const ids = [_]u32{ 1, 0 };
    var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
    defer kv.free();
    var paged = f.seq;
    paged.st = try pagedSeq(&d, &f, kv.ptr, page, &ids);
    defer paged.st.deinit();
    const own = f.e.stateBytes(4096);
    try t.expect(own < std.math.maxInt(u64));
    f.e.cur = &paged;
    defer f.e.cur = &f.seq;
    try t.expectEqual(own, f.e.stateBytes(4096));
}

test "GLM snapshots: with the engine's own sequence on pages too, a prefix's size past a stream's pages is a window's" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    const ats = [_]usize{ 511, 4096, 16000 };
    var sizes: [ats.len]u64 = undefined;
    for (ats, &sizes) |at, *s| s.* = f.e.stateBytes(at); // the fixture's sequence holds a whole window
    const c = f.e.c;
    const page = 2 * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2);
    const ids = [_]u32{ 1, 0 };
    var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
    defer kv.free();
    var paged = f.seq;
    paged.st = try pagedSeq(&d, &f, kv.ptr, page, &ids);
    defer paged.st.deinit();
    var own = f.seq; // the engine's own, on pages and holding none: no whole window anywhere
    own.st = try pagedSeq(&d, &f, kv.ptr, page, &.{});
    defer own.st.deinit();
    f.e.seqs = .empty;
    defer f.e.seqs.deinit(t.allocator);
    try f.e.seqs.append(t.allocator, &own);
    defer f.e.cur = &f.seq;
    for ([_]*@TypeOf(f.seq){ &paged, &own }) |cur| {
        f.e.cur = cur;
        for (ats, sizes) |at, s| try t.expectEqual(s, f.e.stateBytes(at));
    }
}
