//! A DSA layer's kv_b halves as the latent path reads them: bf16 as stored, or E4M3 and fp32 scales (TF_GLM_OVN_KVB8).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const formats = @import("cuda_formats.zig");
const Tensor = core.checkpoint.Tensor;

/// Latent columns one scale covers (the Python family's qmm.F8_BLOCK).
pub const block = 128;

/// Each head's kv_b rows off .. off + len uploaded as `w`; TF_GLM_OVN_KVB8 (bf16 kv_b): E4M3 `w`, fp32 scales `s`.
pub fn half(L: anytype, mx: ?formats.mxfp8.Mxfp8, t: Tensor, name: []const u8, heads: [2]usize, per: usize, off: usize, len: usize, total: usize, lw: usize, comptime w: []const u8, comptime s: []const u8, i: u32) !struct { u64, u64 } {
    for (heads[0]..heads[1]) |head| {
        const lo = head * per + off;
        if (mx) |p| try formats.mxfp8.appendWide(L, p, name, lo, lo + len) else try L.rows(t, lo, lo + len, total, .bf16);
    }
    if (mx != null or !cuda.ovn.on("KVB8")) return .{ try L.put(w, .{i}), 0 };
    if (lw % block != 0 or L.host.items.len % (block * 2) != 0) return error.LatentShape;
    const n = L.host.items.len / 2;
    const codes = try L.gpa.alloc(u8, n);
    defer L.gpa.free(codes);
    const scales = try L.gpa.alloc(f32, n / block);
    defer L.gpa.free(scales);
    quantize(L.host.items, codes, scales) catch |err| {
        std.log.err("{s}: TF_GLM_OVN_KVB8 refuses a non-finite kv_b value", .{name});
        return err;
    };
    L.host.clearRetainingCapacity();
    try L.host.appendSlice(L.gpa, codes);
    const wp = try L.put(w, .{i});
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(scales));
    return .{ wp, try L.put(s, .{i}) };
}

fn bf(bits: []const u8, j: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bits[2 * j ..][0..2], .little)) << 16);
}

/// qmm.make_f8 on bf16 bytes: per 128 values a scale max |x| / 448 (at least 1e-30) and codes e4m3(x / scale).
pub fn quantize(bits: []const u8, codes: []u8, scales: []f32) error{NonFiniteKvb}!void {
    std.debug.assert(bits.len == 2 * codes.len and codes.len == scales.len * block);
    for (scales, 0..) |*sc, b| {
        var amax: f32 = 0;
        for (b * block..(b + 1) * block) |j| {
            const v = bf(bits, j);
            if (!std.math.isFinite(v)) return error.NonFiniteKvb;
            amax = @max(amax, @abs(v));
        }
        sc.* = @max(amax / 448.0, @as(f32, 1e-30));
        for (b * block..(b + 1) * block) |j| codes[j] = core.exlr8.codec.e4m3FromF32(bf(bits, j) / sc.*);
    }
}

test "KVB8: make_f8's scale and codes a 128-column block" {
    var bits: [2 * 2 * block]u8 = @splat(0);
    std.mem.writeInt(u16, bits[2 * 5 ..][0..2], 0x4040, .little); // 3.0, the first block's largest
    std.mem.writeInt(u16, bits[2 * 9 ..][0..2], 0xbfc0, .little); // -1.5
    var codes: [2 * block]u8 = undefined;
    var scales: [2]f32 = undefined;
    try quantize(&bits, &codes, &scales);
    try std.testing.expectEqual(@as(f32, 3.0) / 448.0, scales[0]);
    try std.testing.expectEqual(@as(u8, 0x7e), codes[5]); // 448
    try std.testing.expectEqual(@as(u8, 0xf6), codes[9]); // -224
    try std.testing.expectEqual(@as(u8, 0), codes[0]);
    try std.testing.expectEqual(@as(f32, 1e-30), scales[1]); // an all-zero block
    for (codes[block..]) |c| try std.testing.expectEqual(@as(u8, 0), c);
    std.mem.writeInt(u16, bits[2 * 200 ..][0..2], 0x7fc0, .little);
    try std.testing.expectError(error.NonFiniteKvb, quantize(&bits, &codes, &scales));
}
