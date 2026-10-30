//! GLM profiler lifetime operation used by unwind and close.
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;
const Prof = @import("cuda_prof.zig").Prof;
const max_shared = @import("cuda_engine.zig").max_shared;
const shared_rows = @import("cuda_engine.zig").shared_rows;

pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const p = std.c.getenv("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    const s = std.mem.span(p);
    return if (s.len == 0) @import("heat").default_root else s;
}

pub fn verifyLabel(rows: usize) []const u8 {
    const l = [_][]const u8{ "verify 1 row", "verify 2 rows", "verify 3 rows", "verify 4 rows", "verify 5 rows", "verify 6 rows", "verify 7 rows", "verify 8 rows" };
    return l[@min(rows, l.len) - 1];
}

/// Shared rounds' row buckets: to 8, 16, 24 and 32 rows.
const buckets = shared_rows / 8;

/// TF_GLM_OVN_SHARED_PROF: shared rounds' sections by segments and rows, and their commits by segments (made on use).
pub const Shared = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    io: std.Io,
    rounds: [max_shared + 1][buckets]?*Prof = @splat(@splat(null)),
    commits: [max_shared + 1]?*Prof = @splat(null),

    const round_labels = labels: {
        var l: [max_shared + 1][buckets][]const u8 = undefined;
        for (&l, 0..) |*by_rows, segs| {
            for (by_rows, 0..) |*x, b| x.* = std.fmt.comptimePrint("shared {d} segs to {d} rows", .{ segs, 8 * (b + 1) });
        }
        break :labels l;
    };
    const commit_labels = labels: {
        var l: [max_shared + 1][]const u8 = undefined;
        for (&l, 0..) |*x, segs| x.* = std.fmt.comptimePrint("shared commit {d} segs", .{segs});
        break :labels l;
    };

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, io: std.Io) !*Shared {
        const s = try gpa.create(Shared);
        s.* = .{ .gpa = gpa, .d = d, .io = io };
        return s;
    }

    pub fn destroy(s: *Shared) void {
        for (&s.rounds) |*by_rows| for (by_rows) |*p| if (p.*) |v| v.deinit();
        for (&s.commits) |*p| if (p.*) |v| v.deinit();
        s.gpa.destroy(s);
    }

    fn made(s: *Shared, slot: *?*Prof) !*Prof {
        if (slot.* == null) slot.* = try Prof.init(s.d, s.io);
        return slot.*.?;
    }

    /// A shared forward of `segs` streams and `rows` rows begins on `stream`: its profiler, labelled by both.
    pub fn round(s: *Shared, stream: cuda.Stream, segs: usize, rows: usize) !*Prof {
        const b = (@min(rows, shared_rows) - 1) / 8;
        const p = try s.made(&s.rounds[segs][b]);
        try p.begin(stream, round_labels[segs][b]);
        return p;
    }

    /// The commits of a shared round's `segs` streams begin on `stream`.
    pub fn commit(s: *Shared, stream: cuda.Stream, segs: usize) !*Prof {
        const p = try s.made(&s.commits[segs]);
        try p.begin(stream, commit_labels[segs]);
        return p;
    }
};
