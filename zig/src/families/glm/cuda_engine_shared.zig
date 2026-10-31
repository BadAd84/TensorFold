//! Engine.verifyShared and what the shared rows size (e.bs, DRAW1, SEG_DSA), made last: at the first verify.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const draw1 = @import("cuda_draw1.zig");
const ranks = @import("cuda_ranks.zig");
const Engine = engine.Engine;
const Part = engine.Part;

/// Device bytes `ensure` will allocate on this rank (SEG_DSA's tables aside): the rooms reserve them at load.
pub fn laterBytes(e: *const Engine) !u64 {
    const rows = try ranks.bufferRows(e.settings);
    const bs: u64 = if (e.mtp) try fw.Buffers.sharedBytes(&e.w, rows, e.capacity) else 0;
    return bs + if (cuda.ovn.on("DRAW1")) draw1.deviceBytes(e.share.world, rows, e.c.vocab) else 0;
}

/// The shared buffers, DRAW1's and SEG_DSA's, once, at each rank's first verify: nothing a prompt reads moves.
pub fn ensure(e: *Engine) !void {
    if (e.shared_made) return;
    errdefer release(e);
    const rows = try ranks.bufferRows(e.settings);
    const held = cuda.usage(false).device;
    if (e.mtp) {
        e.bs = try fw.Buffers.initShared(e.d, &e.w, rows, e.capacity);
        const b = &e.bs.?;
        if (e.seg_kda) |*k| b.seg.kda = k;
        if (cuda.ovn.on("SEG_DSA")) { // glm_glue is module 6, glm_ops module 1
            e.seg_dsa = try @import("cuda_dsa_seg.zig").SegDsa.init(e.gpa, e.d, e.m.mods[6], e.m.mods[1], &e.w, b.rows, b.nch, e.capacity);
            b.seg.dsa = &e.seg_dsa.?;
        }
    }
    if (cuda.ovn.on("DRAW1")) e.draw1 = try draw1.Draw1.init(e);
    e.shared_made = true;
    const made = cuda.usage(false).device - held;
    if (e.share.rank == 0) std.log.info("glm: shared windows of {d} rows (cap {d}) made after the load: {d} MiB, {d} MiB held before them", .{ rows, e.settings.shared_rows, made >> 20, held >> 20 });
}

/// Frees what `ensure` made; the next verify makes it again (TF_GLM_OVN_COST_TIME's load-time passes end here).
pub fn release(e: *Engine) void {
    if (e.seg_dsa) |*x| x.deinit();
    e.seg_dsa = null;
    if (e.draw1) |*x| x.deinit();
    e.draw1 = null;
    if (e.bs) |*b| b.deinit();
    e.bs = null;
    e.shared_made = false;
}

/// The parts' total rows; refused past `cap` (TF_GLM_OVN_SHARED_ROWS), max_shared streams, a window, or a slot twice.
pub fn rowsOf(parts: []const Part, cap: usize) !usize {
    if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > cap) return error.WindowTooWide;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
        rows += p.ids.len;
    }
    return rows;
}

/// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
pub fn verifyShared(e: *Engine, parts: []const Part) !void {
    _ = try rowsOf(parts, e.settings.shared_rows);
    try ensure(e);
    const b = if (e.bs) |*x| x else return error.NoSharedWindows;
    std.debug.assert(b.rows >= e.settings.shared_rows); // ensure sized e.bs by bufferRows
    var segs: [engine.max_shared]fw.Seg = undefined;
    var ids: [engine.max_shared_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        try e.bind(p.slot);
        segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
        @memcpy(ids[rows..][0..p.ids.len], p.ids);
        rows += p.ids.len;
    }
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
    const prof = if (e.shared_prof) |sp| try sp.round(e.stream, parts.len, rows) else null;
    const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .l2pf = if (e.l2pf) |*p| p else null, .taps = if (e.drafter) |dr| dr.taps(false) else null, .prof = prof, .side = @import("cuda_moe_side.zig").pick(e, b) };
    try f.run(@ptrCast(ids[0..rows]));
    if (e.drafter) |dr| for (parts, segs[0..parts.len], 0..) |p, g, i| try dr.noteVerify(p.slot, g.row0, g.rows, g.st.pos, i == 0);
    if (e.draw1 != null and !e.structured) try drawPacked(e, b, parts, segs[0..parts.len]) else for (parts, segs[0..parts.len]) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.bits, p.out);
    }
    if (prof) |p| {
        p.mark(e.stream, .draw); // the parts' draws, their exchanges and host syncs
        p.end();
    }
}

/// TF_GLM_OVN_DRAW1: every part whose rule draw1 takes in one draw; the others as before, each on its own.
fn drawPacked(e: *Engine, b: *const fw.Buffers, parts: []const Part, segs: []const fw.Seg) !void {
    var packed_parts: [engine.max_shared]draw1.Part = undefined;
    var n: usize = 0;
    for (parts, segs) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        if (!draw1.takes(e.cur.sampling)) {
            try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.bits, p.out);
            continue;
        }
        packed_parts[n] = .{ .row0 = g.row0, .rows = g.rows, .first = g.st.pos + 1, .sampling = e.cur.sampling, .out = p.out };
        n += 1;
    }
    if (n > 0) try draw1.draw(e, b.logits, packed_parts[0..n]);
}

test "a shared verify's parts: up to the load's rows, each a window, each slot once, at most max_shared streams" {
    var ids: [engine.max_shared_rows]u32 = @splat(7);
    var out: [engine.max_shared_rows]u32 = undefined;
    var parts: [engine.max_shared]Part = undefined;
    for (&parts, 0..) |*p, i| p.* = .{ .slot = @intCast(i), .ids = ids[0..fw.window_rows], .out = out[0..fw.window_rows] };
    for ([_]usize{ 32, 40, 48, 56, 64 }) |cap| {
        const full = cap / fw.window_rows; // windows of 8 rows that fill the setting exactly
        try std.testing.expectEqual(cap, try rowsOf(parts[0..full], cap));
        try std.testing.expectError(error.WindowTooWide, rowsOf(parts[0 .. full + 1], cap)); // one window past it
    }
    try std.testing.expectError(error.WindowTooWide, rowsOf(parts[0..5], 32)); // 40 rows at the default
    parts[1].slot = 0;
    try std.testing.expectError(error.SlotTwice, rowsOf(parts[0..2], 64));
    parts[1] = .{ .slot = 1, .ids = ids[0 .. fw.window_rows + 1], .out = out[0 .. fw.window_rows + 1] };
    try std.testing.expectError(error.WindowTooWide, rowsOf(parts[0..2], 64));
    try std.testing.expectError(error.WindowTooWide, rowsOf(parts[0..0], 64));
}

test "a shared verify past the load's TF_GLM_OVN_SHARED_ROWS is refused before any rank hears it or a slot binds" {
    var e: Engine = undefined; // no slot, link or device: a refusal touches none of them
    e.settings = .{}; // the default 32 rows
    var b: fw.Buffers = undefined;
    b.rows = e.settings.shared_rows;
    e.bs = b;
    var ids: [fw.window_rows]u32 = @splat(7);
    var out: [5][fw.window_rows]u32 = undefined;
    var parts: [5]Part = undefined;
    for (&parts, &out, 0..) |*p, *o, i| p.* = .{ .slot = @intCast(i), .ids = &ids, .out = o };
    try std.testing.expectError(error.WindowTooWide, verifyShared(&e, &parts)); // 40 rows
    const x: @import("cuda_follow.zig").Leader = .{ .e = &e, .link = undefined };
    try std.testing.expectError(error.WindowTooWide, x.verifyShared(&parts));
}

test "an engine refuses shared rows TF_GLM_OVN_SHARED_ROWS does not allow (auto unresolved) before anything else" {
    for ([_]u32{ 0, 36, 72 }) |rows| {
        const o: engine.Options = .{ .capacity = 4096, .mtp = true, .settings = .{ .shared_rows = rows } };
        try std.testing.expectError(error.BadSharedRows, Engine.init(std.testing.allocator, std.testing.io, undefined, undefined, "/no-such-glm", undefined, null, 0, 1, o));
    }
}

test "the shared rows size nothing at load: Engine.init and the knobs' open make no shared buffer, DRAW1 or SEG_DSA" {
    const src = @embedFile("cuda_engine.zig");
    const at = std.mem.indexOf(u8, src, "pub fn init(").?;
    const init = src[at..std.mem.indexOfPos(u8, src, at, "pub fn deinit(").?];
    const ovn = @embedFile("cuda_engine_ovn.zig");
    const open = ovn[0..std.mem.indexOf(u8, ovn, "pub fn close(").?];
    for ([_][]const u8{ "initShared", "Draw1.init", "SegDsa", "ensure(" }) |made| {
        try std.testing.expect(std.mem.indexOf(u8, init, made) == null); // before the page pool: would move what follows
        try std.testing.expect(std.mem.indexOf(u8, open, made) == null);
    }
    try std.testing.expect(std.mem.indexOf(u8, src, "@import(\"cuda_engine_shared.zig\").ensure(e)") != null); // verify makes them
}

test "release frees what ensure made and lets the next verify make it again" {
    var e: Engine = undefined;
    e.bs = null;
    e.draw1 = null;
    e.seg_dsa = null;
    e.shared_made = true;
    release(&e);
    try std.testing.expect(!e.shared_made and e.bs == null and e.draw1 == null and e.seg_dsa == null);
}
