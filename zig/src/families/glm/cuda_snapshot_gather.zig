//! TF_GLM_OVN_SNAP_GATHER: a keep's or restore's pieces copied by a tf_glm_snap_copy launch a batch, not a memcpy each.
const std = @import("std");
const cuda = @import("cuda");
const Plan = @import("cuda_snapshot_layout.zig").Plan;
const kv = @import("core").kv_pool;

/// One copy as the kernel reads it (glm_glue_rows.cuh's tf_glm_snap_piece).
pub const Piece = extern struct { src: u64, dst: u64, len: u64 };

/// Pieces a launch copies: its list's device bytes are made once, with the knob, beside the snapshot pages.
pub const batch = 8192;

/// The kernel's block: 256 threads, 16-byte copies where a piece's ends allow.
const threads = 256;

pub const Gather = struct {
    f: cuda.Function,
    dev: cuda.DeviceBuffer,
    host: []Piece,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, f: cuda.Function) !Gather {
        const host = try gpa.alloc(Piece, batch);
        errdefer gpa.free(host);
        return .{ .f = f, .dev = try cuda.DeviceBuffer.alloc(d, batch * @sizeOf(Piece)), .host = host, .gpa = gpa };
    }

    pub fn deinit(g: *Gather) void {
        g.dev.free();
        g.gpa.free(g.host);
    }

    /// cuda_snapshot_pages' Pool.move by batches: the same pieces in the same order, each copied whole by one block.
    pub fn move(g: *const Gather, arena: u64, page: usize, stream: cuda.Stream, lease: *const kv.Lease, plan: *const Plan, into: bool) !void {
        var n: usize = 0;
        var off: usize = 0;
        for (plan.parts[0..plan.count]) |part| {
            var done: usize = 0;
            while (done < part.len) {
                const at = off + done;
                const target = arena + @as(u64, lease.ids.items[at / page]) * page + at % page;
                const live = plan.piece(part, done);
                const len = @min(live.len, page - at % page);
                g.host[n] = if (into) .{ .src = live.live, .dst = target, .len = len } else .{ .src = target, .dst = live.live, .len = len };
                n += 1;
                if (n == batch) {
                    try g.flush(stream, n);
                    n = 0;
                }
                done += len;
            }
            off += part.len;
        }
        if (n > 0) try g.flush(stream, n);
    }

    /// The list uploaded from pageable memory (staged before the call returns, so the next batch may refill it).
    pub fn flush(g: *const Gather, stream: cuda.Stream, n: usize) !void {
        try g.dev.uploadAsync(0, std.mem.sliceAsBytes(g.host[0..n]), stream.handle);
        var a: cuda.Args = .{};
        a.add(g.dev.ptr);
        a.add(@as(u32, @intCast(n)));
        try cuda.launch.launch(g.f, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = threads } }, stream, &a);
    }
};

test "a piece is three 8-byte words, as the kernel's struct" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Piece));
}
