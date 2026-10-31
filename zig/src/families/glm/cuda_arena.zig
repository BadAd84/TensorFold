//! One device allocation carved into 256-byte aligned fields (cuda_state's Buffers and State).
const std = @import("std");
const cuda = @import("cuda");

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    pub fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

/// The bytes `arena` allocates for `sizes`: each 256-byte aligned, at least 256 in all.
pub fn total(sizes: []const usize) usize {
    var n: usize = 0;
    for (sizes) |s| n += std.mem.alignForward(usize, s, 256);
    return @max(n, 256);
}

/// Allocates `sizes` (bytes) in one zeroed device buffer and hands out each field's address in order.
pub fn arena(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
    var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total(sizes)) };
    errdefer a.buf.free();
    try a.buf.fill8(0, null);
    for (sizes, out) |n, *p| p.* = a.take(n);
    return a;
}

test "an arena's bytes: each field 256-byte aligned, 256 at least" {
    try std.testing.expectEqual(@as(usize, 256), total(&.{}));
    try std.testing.expectEqual(@as(usize, 1024), total(&.{ 1, 256, 257 })); // 256 + 256 + 512
}
