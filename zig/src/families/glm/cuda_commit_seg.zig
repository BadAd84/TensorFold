//! TF_GLM_OVN_SEG_COMMIT: a shared round's commits as one step and one launch set: every stream's KDA replay (those
//! keeping fewer rows than they ran: tf_glm_kda_replay_layers_seg) and conv shift (tf_glm_conv_shift_seg) from a
//! table, then each stream's position. Forward.commit's checks and kernels, the streams in a grid dimension.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const kda = @import("cuda_kda.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

const max_segs = engine.max_shared;

/// glm_kda.cu's KdaKeep and glm_glue's tf_glm_conv_seg (u64 words).
const Keep = extern struct { keep: i64, pending: i64, state_in: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };
const Conv = extern struct { conv: u64, proj: u64, keep: i64 };

/// One stream's commit: its slot and the rows of its last window it keeps.
pub const Kept = struct { slot: u32, n: u32 };

pub const SegCommit = struct {
    replay: cuda.Function,
    shift: cuda.Function,
    table: cuda.DeviceBuffer, // [max_segs]Keep, then [max_segs]Conv

    /// Made at load with the knob on, from glm_kda and glm_glue (the forward's modules 0 and 6).
    pub fn init(d: *const cuda.Driver, kda_module: cuda.Module, glue_module: cuda.Module) !SegCommit {
        return .{ .replay = try kda_module.function("tf_glm_kda_replay_layers_seg"), .shift = try glue_module.function("tf_glm_conv_shift_seg"), .table = try cuda.DeviceBuffer.alloc(d, max_segs * (@sizeOf(Keep) + @sizeOf(Conv))) };
    }

    pub fn deinit(x: *SegCommit) void {
        x.table.free();
    }
};

/// Every stream's commit of its last window's first `n` rows, on every rank; refused whole before any work.
pub fn keepShared(e: *Engine, kept: []const Kept) !void {
    const x = if (e.seg_commit) |*v| v else return error.NoSegCommit;
    if (kept.len == 0 or kept.len > max_segs) return error.BadCommit;
    var keeps: [max_segs]Keep = undefined;
    var convs: [max_segs]Conv = undefined;
    var sts: [max_segs]*fw.State = undefined;
    for (kept, 0..) |c, i| { // Forward.commit's checks, for every stream before the first launch
        for (kept[0..i]) |q| if (q.slot == c.slot) return error.SlotTwice;
        try e.bind(c.slot);
        const st = &e.cur.st;
        if (c.n == 0 or c.n > st.pending or st.prompt or st.kda_layers == 0) return error.BadCommit;
        const cur = st.cur[0];
        for (st.cur) |y| if (y != cur) return error.KdaLayersOutOfStep;
        keeps[i] = .{ .keep = c.n, .pending = @intCast(st.pending), .state_in = st.recAt(cur, 0), .state_out = st.recAt(1 - cur, 0), .k = st.save_k, .v = st.save_v, .g = st.save_g, .b = st.save_b };
        convs[i] = .{ .conv = st.conv, .proj = st.proj, .keep = c.n };
        sts[i] = st;
    }
    const st = sts[0]; // every sequence has the same shapes
    const h = st.heads;
    const n: u32 = @intCast(kept.len);
    try x.table.uploadAsync(0, std.mem.sliceAsBytes(keeps[0..kept.len]), e.stream.handle);
    try x.table.uploadAsync(max_segs * @sizeOf(Keep), std.mem.sliceAsBytes(convs[0..kept.len]), e.stream.handle);
    var a: cuda.Args = .{};
    a.add(x.table.ptr);
    a.add(@as(c_int, @intCast(h)));
    a.add(@as(usize, st.rec_layer / 4));
    a.add(@as(usize, fw.window_rows * h * kda.dk));
    a.add(@as(usize, fw.window_rows * h));
    try cuda.launch.launch(x.replay, .{ .grid = .{ .x = @intCast(st.kda_layers * h), .y = n }, .block = .{ .x = 1024 } }, e.stream, &a);
    var b: cuda.Args = .{};
    b.add(x.table.ptr + max_segs * @sizeOf(Keep));
    inline for (.{ 3 * st.conv_c, st.proj_layer, st.proj_row, st.conv_c }) |v| b.add(@as(i64, @intCast(v)));
    try cuda.launch.launch(x.shift, .{ .grid = .{ .x = @intCast(st.kda_layers), .y = @intCast((st.conv_c + 255) / 256), .z = n }, .block = .{ .x = 256 } }, e.stream, &b);
    for (sts[0..kept.len], kept) |s, c| {
        for (s.cur) |*y| y.* = 1 - y.*;
        s.pending = 0;
        try s.setPos(e.stream, s.pos + c.n);
    }
}

comptime {
    std.debug.assert(@sizeOf(Keep) == 64 and @sizeOf(Conv) == 24);
}
