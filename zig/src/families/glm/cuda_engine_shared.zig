//! Engine.verifyShared: several streams' verify windows in one forward, then each part's draws.
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const draw1 = @import("cuda_draw1.zig");
const Engine = engine.Engine;
const Part = engine.Part;

/// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
pub fn verifyShared(e: *Engine, parts: []const Part) !void {
    const b = if (e.bs) |*x| x else return error.NoSharedWindows;
    if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
    var segs: [engine.max_shared]fw.Seg = undefined;
    var ids: [engine.shared_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > engine.shared_rows) return error.WindowTooWide;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
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
