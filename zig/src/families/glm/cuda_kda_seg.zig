//! TF_GLM_OVN_SEG_KDA: a shared window's KDA chains for every segment in one launch set (glm_kda.cu's *_seg kernels:
//! a segment a grid z or y, tf_glm_kda_out over all rows), from a table of each segment's rows and layer-0 places.
//! Each row's arithmetic is the per-segment chain's; only which pointers a block reads moves into the table.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const kda = @import("cuda_kda.zig");
const kda_block = @import("cuda_kda_block.zig");
const window_rows = @import("cuda_state.zig").window_rows;

/// The most segments a table holds (Engine.max_shared).
const max_segs = 16;

/// glm_kda.cu's KdaSeg: a segment's rows, its conv window, states and saved rows at layer 0.
const Seg = extern struct { row0: i64, rows: i64, cs: u64, state_in: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };

pub const SegKda = struct {
    prep: cuda.Function,
    step: cuda.Function,
    chain: cuda.Function,
    table: cuda.DeviceBuffer, // [max_segs]Seg
    segs: usize = 0, // the table's segments this forward (0: it holds none, the per-segment loop runs)
    conv_e: usize = 0, // a layer's stride in elements: conv window (bf16), state (fp32), saved k/v/g rows, beta
    rec_e: usize = 0,
    kv_e: usize = 0,
    b_e: usize = 0,

    /// Made at load with the knob on, from glm_kda (the forward's module 0, fw.Modules.load's order).
    pub fn init(d: *const cuda.Driver, m: cuda.Module) !SegKda {
        return .{ .prep = try m.function("tf_glm_kda_prep_seg"), .step = try m.function("tf_glm_kda_step_seg"), .chain = try m.function("tf_glm_kda_chain_seg"), .table = try cuda.DeviceBuffer.alloc(d, max_segs * @sizeOf(Seg)) };
    }

    pub fn deinit(x: *SegKda) void {
        x.table.free();
    }

    /// shared.begin: the window's segments into the table, unless one's KDA layers are out of step (it keeps the loop).
    pub fn fill(x: *SegKda, segs: []const fw.Seg, stream: cuda.Stream) !void {
        x.segs = 0;
        if (segs.len < 2 or segs.len > max_segs) return;
        var t: [max_segs]Seg = undefined;
        for (segs, t[0..segs.len]) |g, *o| {
            const st = g.st;
            if (st.kda_layers == 0 or g.rows > window_rows) return;
            const cur = st.cur[0];
            for (st.cur) |c| if (c != cur) return;
            o.* = .{ .row0 = @intCast(g.row0), .rows = @intCast(g.rows), .cs = st.conv, .state_in = st.recAt(cur, 0), .state_out = st.recAt(1 - cur, 0), .k = st.save_k, .v = st.save_v, .g = st.save_g, .b = st.save_b };
        }
        try x.table.uploadAsync(0, std.mem.sliceAsBytes(t[0..segs.len]), stream.handle);
        const st = segs[0].st; // every sequence has the same shapes
        x.conv_e = 3 * st.conv_c;
        x.rec_e = st.rec_layer / 4;
        x.kv_e = window_rows * st.heads * kda.dk;
        x.b_e = window_rows * st.heads;
        x.segs = segs.len;
    }

    /// Layer `li`'s chains for the table's segments: `rows` projected rows from s.p, s.ka and s.kg; out at s.saves.out.
    pub fn chains(x: *const SegKda, ks: *const kda.Kernels, stream: cuda.Stream, l: kda_block.Layer, s: kda_block.Scratch, li: usize, rows: usize, wide: bool) !void {
        const h: c_int = @intCast(l.heads);
        const n = l.heads * kda.dk;
        const p_stride: c_int = @intCast(l.matWidth());
        const b_off: c_int = @intCast(3 * n + 2 * kda.dk);
        const segs: u32 = @intCast(x.segs);
        if (!wide) {
            var a: cuda.Args = .{};
            a.add(x.table.ptr);
            for ([_]usize{ li, x.conv_e, x.rec_e, x.kv_e, x.b_e }) |v| a.add(@as(i64, @intCast(v)));
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
        for ([_]usize{ li, x.conv_e, x.kv_e, x.b_e }) |v| a.add(@as(i64, @intCast(v)));
        a.add(h);
        a.add(s.p);
        a.add(p_stride);
        a.add(b_off);
        a.add(s.ka);
        a.add(@as(c_int, @intCast(n)));
        for ([_]u64{ l.conv_w, l.a_log, l.dt_bias }) |v| a.add(v);
        a.add(l.lower);
        a.add(s.wide.q);
        try cuda.launch.launch(x.prep, .{ .grid = .{ .x = window_rows, .y = @intCast(l.heads), .z = segs }, .block = .{ .x = 512 } }, stream, &a);
        var b: cuda.Args = .{};
        b.add(x.table.ptr);
        for ([_]usize{ li, x.rec_e, x.kv_e, x.b_e }) |v| b.add(@as(i64, @intCast(v)));
        b.add(h);
        b.add(s.wide.q);
        b.add(s.wide.y);
        try cuda.launch.launch(x.step, .{ .grid = .{ .x = @intCast(l.heads), .y = kda.dv / 4 / 8, .z = segs }, .block = .{ .x = 8 * 32 } }, stream, &b);
        var o: cuda.Args = .{}; // tf_glm_kda_out over every segment's rows at once: a row's norm reads only its row
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

comptime {
    std.debug.assert(@sizeOf(Seg) == 72);
}
