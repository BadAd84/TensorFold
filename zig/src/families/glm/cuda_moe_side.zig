//! TF_GLM_OVN_SHARED_SIDE=N: a decode window's shared expert on a high-priority side stream beside the routed ones.
const cuda = @import("cuda");
const moe = @import("cuda_moe.zig");
const fw = @import("cuda_forward.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// The side stream and its fork and join events; `most`: forks while at most this many streams are live.
pub const Side = struct {
    stream: cuda.Stream,
    fork: cuda.Event,
    join: cuda.Event,
    most: usize,

    /// The knob's side (unset or 0: none, and nothing is made).
    pub fn init(d: *const cuda.Driver) !?Side {
        const most = try cuda.ovn.int("SHARED_SIDE", 0);
        if (most == 0) return null;
        var stream = try cuda.Stream.initPriority(d, .highest);
        errdefer stream.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        const join = try cuda.Event.init(d, false);
        return .{ .stream = stream, .fork = fork, .join = join, .most = @intCast(most) };
    }

    pub fn deinit(s: *Side) void {
        s.join.deinit();
        s.fork.deinit();
        s.stream.deinit();
        s.* = undefined;
    }

    /// `main` waits for every item queued on the side stream so far.
    fn rejoin(s: *const Side, main: cuda.Stream) !void {
        try s.join.record(s.stream);
        try main.wait(s.join);
    }
};

/// Streams with a sequence on this engine (every rank keeps the same): open pages' slots, else every made slot.
pub fn live(e: *const Engine) usize {
    var n: usize = 0;
    if (e.pages) |p| {
        for (p.used) |u| n += @intFromBool(u);
        return n;
    }
    for (e.seqs.items) |q| n += @intFromBool(q != null);
    return n;
}

/// The side a forward on buffers `b` forks its shared experts on: decode windows while at most `most` streams live.
pub fn pick(e: *const Engine, b: *const fw.Buffers) ?*const Side {
    const s = if (e.m.side) |*x| x else return null;
    if (b.prefill or live(e) > s.most) return null;
    return s;
}

/// moe.block of a decode window: the shared expert forked after the site's hc_pre, joined before the fill's tail.
pub fn block(k: moe.Kernels, main: cuda.Stream, side: *const Side, l: moe.Layer, s: moe.Scratch, x: u64, x_stride: usize, rows: usize, out: u64) !void {
    try side.fork.record(main);
    try side.stream.wait(side.fork);
    errdefer side.rejoin(main) catch {}; // a failed window leaves no side work running past it
    try moe.sharedFront(k, side.stream, l, s, x, x_stride, 0, rows);
    try moe.body(k, main, l, s, x, x_stride, rows, false);
    try side.rejoin(main);
    try moe.fillTail(k, main, l, s, 0, rows, false, out);
}
