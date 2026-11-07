//! The OVN knobs that are bits of `ovn_comms`, so ranks that set them differently are refused before any collective.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");

test "TF_GLM_OVN_PRX: the knob is a bit of ovn_comms, so ranks that differ are refused before its collective setup" {
    cuda.ovn.fake("PRX", "1");
    defer cuda.ovn.fake("PRX", null);
    try std.testing.expectEqual(@as(u32, 1 << 3), try ranks.ovnComms() & 1 << 3);
    cuda.ovn.fake("PRX", null);
    try std.testing.expectEqual(@as(u32, 0), try ranks.ovnComms() & 1 << 3);
}

test "TF_GLM_OVN_PRX_ALONE: a bit of ovn_comms apart from PRX's, so ranks that pick passes differently are refused" {
    cuda.ovn.fake("PRX", null);
    cuda.ovn.fake("PRX_ALONE", "1");
    defer cuda.ovn.fake("PRX_ALONE", null);
    try std.testing.expectEqual(@as(u32, 1 << 4), try ranks.ovnComms() & 0x18); // bit 4, PRX's bit 3 untouched
    cuda.ovn.fake("PRX_ALONE", null);
    try std.testing.expectEqual(@as(u32, 0), try ranks.ovnComms() & 1 << 4);
}
