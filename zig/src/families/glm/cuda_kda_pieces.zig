//! TF_GLM_OVN_KDA_PIECES: a multi-prompt chunk's KDA chains in one launch (PrefillLeftovers-2-kda-pieces-grid.patch).
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const kda = @import("cuda_kda.zig");
const kda_block = @import("cuda_kda_block.zig");
const kda_chunk = @import("cuda_kda_chunk.zig");
const kda_seg = @import("cuda_kda_seg.zig");
const state = @import("cuda_state.zig");
const prefill_rows = state.prefill_rows;

/// The most pieces one launch holds (Engine.max_together, the prompts prefillMany takes).
pub const max_pieces = 16;

/// The routing a chunk's pieces take: batch them (segs > 0, one chains launch) or not (segs 0, the per-piece loop).
pub const Plan = struct {
    table: [max_pieces]kda_seg.Seg = undefined,
    segs: usize = 0,
    wide: bool = false, // every piece takes the three-kernel chain (all >= kda.wide_rows rows)
    max_rows: usize = 0, // the widest piece's rows: the wide chain's prep grid x
    conv_e: usize = 0, // a layer's stride, bf16 elements: the conv window
    rec_e: usize = 0, // a layer's stride, fp32 elements: the recurrent state
};

/// The pieces' one-launch plan (port of forward.kda_pieces / kda_chunked.chain_pieces), or segs 0 so the loop runs.
pub fn plan(segs: []const fw.Seg, saves: kda.Saves, chunked: bool) Plan {
    var out: Plan = .{};
    if (segs.len < 2 or segs.len > max_pieces) return out;
    if (chunked) return out; // TF_GLM_OVN_KDA_CHUNKED on: the WY chain's bits, not the *_seg kernels'
    const st0 = segs[0].st;
    if (st0.kda_layers == 0) return out;
    const n = st0.heads * kda.dk;
    const wide = segs[0].rows >= kda.wide_rows;
    var total: usize = 0;
    var max_rows: usize = 0;
    for (segs, out.table[0..segs.len]) |g, *o| {
        const st = g.st;
        if (st.heads != st0.heads or st.kda_layers != st0.kda_layers) return .{}; // one shape for every piece
        if ((g.rows >= kda.wide_rows) != wide) return .{}; // one chain form for the whole launch
        const cur = st.cur[0];
        for (st.cur) |c| if (c != cur) return .{}; // every layer at one parity: recAt(cur, 0) + li * rec_e reaches li
        const krow = g.row0 * n; // the piece's first row in the k / v / g scratch (heads * 128 a row)
        o.* = .{
            .row0 = @intCast(g.row0),
            .rows = @intCast(g.rows),
            .cs = st.conv,
            .state_in = st.recAt(cur, 0),
            .state_out = st.recAt(1 - cur, 0),
            .k = saves.k + @as(u64, krow) * 4,
            .v = saves.v + @as(u64, krow) * 2,
            .g = saves.g + @as(u64, krow) * 4,
            .b = saves.b + @as(u64, g.row0) * st0.heads * 4,
        };
        total += g.rows;
        max_rows = @max(max_rows, g.rows);
    }
    if (total > prefill_rows) return .{};
    out.conv_e = 3 * st0.conv_c;
    out.rec_e = st0.rec_layer / 4;
    out.wide = wide;
    out.max_rows = max_rows;
    out.segs = segs.len;
    return out;
}

pub const PiecesKda = struct {
    chain: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    table: cuda.DeviceBuffer, // [max_pieces]kda_seg.Seg
    segs: usize = 0, // this chunk's pieces (0: not batched, cuda_shared.mixer's per-piece loop runs)
    wide: bool = false,
    max_rows: usize = 0,
    conv_e: usize = 0,
    rec_e: usize = 0,

    /// Made at load with the knob on, from glm_kda (the forward's module 0), the same *_seg kernels SEG_KDA loads.
    pub fn init(d: *const cuda.Driver, m: cuda.Module) !PiecesKda {
        return .{
            .chain = try m.function("tf_glm_kda_chain_seg"),
            .prep = try m.function("tf_glm_kda_prep_seg"),
            .step = try m.function("tf_glm_kda_step_seg"),
            .table = try cuda.DeviceBuffer.alloc(d, max_pieces * @sizeOf(kda_seg.Seg)),
        };
    }

    pub fn deinit(x: *PiecesKda) void {
        x.table.free();
    }

    /// shared.begin for a prompt buffer: plan the chunk's pieces, upload the table; segs 0 leaves the per-piece loop.
    pub fn fillPrompt(x: *PiecesKda, segs: []const fw.Seg, saves: kda.Saves, stream: cuda.Stream) !void {
        x.segs = 0;
        const p = plan(segs, saves, kda_chunk.enabled());
        if (p.segs == 0) return;
        try x.table.uploadAsync(0, std.mem.sliceAsBytes(p.table[0..p.segs]), stream.handle);
        x.conv_e = p.conv_e;
        x.rec_e = p.rec_e;
        x.wide = p.wide;
        x.max_rows = p.max_rows;
        x.segs = p.segs;
    }

    /// Layer li's chains for all pieces in one launch set (chain_seg, or prep_seg / step_seg / out); out at saves.out.
    pub fn chains(x: *const PiecesKda, ks: *const kda.Kernels, stream: cuda.Stream, l: kda_block.Layer, s: kda_block.Scratch, li: usize, rows: usize) !void {
        const h: c_int = @intCast(l.heads);
        const n = l.heads * kda.dk;
        const p_stride: c_int = @intCast(l.matWidth());
        const b_off: c_int = @intCast(3 * n + 2 * kda.dk);
        const segs: u32 = @intCast(x.segs);
        if (!x.wide) {
            var a: cuda.Args = .{};
            a.add(x.table.ptr);
            for ([_]i64{ @intCast(li), @intCast(x.conv_e), @intCast(x.rec_e), 0, 0 }) |v| a.add(v); // li, conv_e, rec_e, kv_e, b_e
            a.add(h);
            a.add(s.p);
            a.add(p_stride);
            a.add(b_off);
            a.add(s.ka);
            a.add(@as(c_int, @intCast(n)));
            a.add(s.kg);
            a.add(@as(c_int, @intCast(n)));
            for ([_]u64{ l.conv_w, l.a_log, l.dt_bias, l.norm_w }) |v| a.add(v);
            a.add(l.eps);
            a.add(l.lower);
            a.add(s.saves.out);
            return cuda.launch.launch(x.chain, .{ .grid = .{ .x = @intCast(l.heads), .y = segs }, .block = .{ .x = 1024 } }, stream, &a);
        }
        var a: cuda.Args = .{};
        a.add(x.table.ptr);
        for ([_]i64{ @intCast(li), @intCast(x.conv_e), 0, 0 }) |v| a.add(v); // li, conv_e, kv_e, b_e
        a.add(h);
        a.add(s.p);
        a.add(p_stride);
        a.add(b_off);
        a.add(s.ka);
        a.add(@as(c_int, @intCast(n)));
        for ([_]u64{ l.conv_w, l.a_log, l.dt_bias }) |v| a.add(v);
        a.add(l.lower);
        a.add(s.wide.q);
        try cuda.launch.launch(x.prep, .{ .grid = .{ .x = @intCast(x.max_rows), .y = @intCast(l.heads), .z = segs }, .block = .{ .x = 512 } }, stream, &a);
        var b: cuda.Args = .{};
        b.add(x.table.ptr);
        for ([_]i64{ @intCast(li), @intCast(x.rec_e), 0, 0 }) |v| b.add(v); // li, rec_e, kv_e, b_e
        b.add(h);
        b.add(s.wide.q);
        b.add(s.wide.y);
        try cuda.launch.launch(x.step, .{ .grid = .{ .x = @intCast(l.heads), .y = kda.dv / 4 / 8, .z = segs }, .block = .{ .x = 8 * 32 } }, stream, &b);
        var o: cuda.Args = .{}; // tf_glm_kda_out over every piece's rows at once (they are contiguous): a row reads only its row
        o.add(h);
        o.add(s.wide.y);
        o.add(s.kg);
        o.add(@as(c_int, @intCast(n)));
        o.add(l.norm_w);
        o.add(l.eps);
        o.add(s.saves.out);
        try cuda.launch.launch(ks.out, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(l.heads) }, .block = .{ .x = kda.dv } }, stream, &o);
    }
};

// -- host tests on plan (the decision), from cuda_prompts.pass's shapes: contiguous fresh prompts -----------------

const DK = kda.dk;

/// A fresh prompt stream as cuda_prompts.pass binds it (pos 0, cur all zero); only the fields plan reads are set.
fn freshState(cur_buf: []u1, heads: usize, layers: usize, base: u64) state.State {
    @memset(cur_buf, 0);
    var st: state.State = undefined;
    st.heads = heads;
    st.kda_layers = layers;
    st.cur = cur_buf;
    st.conv = base + 0x1000;
    st.rec = base + 0x2000;
    st.conv_c = 3 * heads * DK;
    st.rec_layer = heads * DK * DK * 4;
    return st;
}

/// cuda_prompts.pass's segs: contiguous row0, each a prompt's rows. `rows_each`'s length is the piece count.
fn piecesOf(sts: []state.State, segs: []fw.Seg, rows_each: []const usize) []fw.Seg {
    var at: usize = 0;
    for (rows_each, 0..) |r, k| {
        segs[k] = .{ .st = &sts[k], .row0 = at, .rows = r };
        at += r;
    }
    return segs[0..rows_each.len];
}

const test_heads = 22;
const test_layers = 4;
const test_saves: kda.Saves = .{ .out = 0x10_0000, .state_out = 0x20_0000, .k = 0x30_0000, .v = 0x40_0000, .g = 0x50_0000, .b = 0x60_0000 };

const Fixture = struct {
    sts: [max_pieces]state.State = undefined,
    curs: [max_pieces][test_layers]u1 = undefined,
    seg_buf: [max_pieces]fw.Seg = undefined,

    fn make(self: *Fixture, rows_each: []const usize) []fw.Seg {
        for (0..rows_each.len) |k| self.sts[k] = freshState(&self.curs[k], test_heads, test_layers, @as(u64, k) * 0x100_0000);
        return piecesOf(self.sts[0..rows_each.len], &self.seg_buf, rows_each);
    }
};

test "TF_GLM_OVN_KDA_PIECES: small fresh prompts batch as one non-wide launch, the table each piece's own rows" {
    var fx: Fixture = .{};
    const rows_each = [_]usize{ 39, 39, 39, 39 }; // production's four 39-row fork pieces (patch test, H22)
    const segs = fx.make(&rows_each);
    const p = plan(segs, test_saves, false);
    try std.testing.expectEqual(@as(usize, 4), p.segs);
    try std.testing.expect(!p.wide); // 39 < kda.wide_rows (64): the serial chain_seg
    try std.testing.expectEqual(@as(usize, 39), p.max_rows);
    const n = test_heads * DK;
    var at: usize = 0;
    for (0..4) |k| {
        const o = p.table[k];
        try std.testing.expectEqual(@as(i64, @intCast(at)), o.row0);
        try std.testing.expectEqual(@as(i64, 39), o.rows);
        try std.testing.expectEqual(fx.sts[k].conv, o.cs);
        try std.testing.expectEqual(fx.sts[k].recAt(0, 0), o.state_in); // fresh: cur 0
        try std.testing.expectEqual(fx.sts[k].recAt(1, 0), o.state_out);
        try std.testing.expectEqual(test_saves.k + @as(u64, at * n) * 4, o.k); // row0-baked, kv_e 0 in the kernel
        try std.testing.expectEqual(test_saves.v + @as(u64, at * n) * 2, o.v);
        try std.testing.expectEqual(test_saves.g + @as(u64, at * n) * 4, o.g);
        try std.testing.expectEqual(test_saves.b + @as(u64, at) * test_heads * 4, o.b);
        at += 39;
    }
    try std.testing.expectEqual(3 * fx.sts[0].conv_c, p.conv_e); // the conv window's layer stride: 3 rows of 3 * heads * 128
    try std.testing.expectEqual(@as(usize, test_heads * DK * DK), p.rec_e);
}

test "TF_GLM_OVN_KDA_PIECES: fresh prompts all at or past the wide floor batch as one wide launch" {
    var fx: Fixture = .{};
    const rows_each = [_]usize{ 64, 300, 128 }; // all >= kda.wide_rows: the prep / step / out chain
    const segs = fx.make(&rows_each);
    const p = plan(segs, test_saves, false);
    try std.testing.expectEqual(@as(usize, 3), p.segs);
    try std.testing.expect(p.wide);
    try std.testing.expectEqual(@as(usize, 300), p.max_rows); // the widest piece sizes the prep grid
    try std.testing.expectEqual(@as(i64, 64), p.table[1].row0); // contiguous after the 64-row piece
}

test "TF_GLM_OVN_KDA_PIECES: pieces that disagree on the chain form do not batch (one kernel cannot be two forms)" {
    var fx: Fixture = .{};
    const mixed = [_]usize{ 39, 300 }; // one below the wide floor, one past it
    try std.testing.expectEqual(@as(usize, 0), plan(fx.make(&mixed), test_saves, false).segs);
    const just_under = [_]usize{ 63, 100 }; // 63 serial, 100 wide
    try std.testing.expectEqual(@as(usize, 0), plan(fx.make(&just_under), test_saves, false).segs);
}

test "TF_GLM_OVN_KDA_PIECES: not batched for one piece, over max, or with the chunked WY kernel on" {
    var fx: Fixture = .{};
    const one = [_]usize{39};
    try std.testing.expectEqual(@as(usize, 0), plan(fx.make(&one), test_saves, false).segs);
    const four = [_]usize{ 39, 39, 39, 39 };
    try std.testing.expectEqual(@as(usize, 0), plan(fx.make(&four), test_saves, true).segs); // chunked on
    try std.testing.expectEqual(@as(usize, 4), plan(fx.make(&four), test_saves, false).segs); // the same shape, chunked off
}

test "TF_GLM_OVN_KDA_PIECES: a stream out of step across layers does not batch (the layer-0 table needs one parity)" {
    var fx: Fixture = .{};
    const rows_each = [_]usize{ 100, 100 }; // both wide, so the parity guard fires on its own, not the form
    const segs = fx.make(&rows_each);
    try std.testing.expectEqual(@as(usize, 2), plan(segs, test_saves, false).segs); // the aligned base case
    fx.curs[1][2] = 1; // piece 1's layer 2 at the other parity: recAt(cur, 0) + li * rec_e no longer reaches it
    try std.testing.expectEqual(@as(usize, 0), plan(segs, test_saves, false).segs);
}

test "TF_GLM_OVN_KDA_PIECES: pieces whose rows sum past the buffer do not batch" {
    var fx: Fixture = .{};
    const rows_each = [_]usize{ 2048, 2049 }; // 4097 > prefill_rows, both wide so the form agrees
    try std.testing.expectEqual(@as(usize, 0), plan(fx.make(&rows_each), test_saves, false).segs);
}
