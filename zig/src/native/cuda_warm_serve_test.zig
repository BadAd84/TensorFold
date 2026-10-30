//! A family's warm-turn wait and warm_alone reach the real lane host without a device (Host.serve).
const std = @import("std");
const lanes = @import("lanes");

/// The host's warm_quiet and warm_alone after serving family F with warm turns on or off.
fn served(comptime Host: type, comptime serve: anytype, comptime F: type, warm: bool) !struct { quiet: u64, alone: bool } {
    const gpa = std.testing.allocator;
    const Family = struct {
        fn release(_: *anyopaque) void {}
    };
    var unused: u8 = 0;
    var backend: lanes.fake.Fake = .{ .gpa = gpa };
    defer backend.deinit();
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = null, .family = &unused, .release = Family.release, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    try serve(h, F, std.testing.io, .{ .exact_width = 8 }, 8, .{ .lanes = 2, .warm_turns = warm }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    return .{ .quiet = h.host.warm_quiet, .alone = h.host.warm_alone };
}

pub fn check(comptime Host: type, comptime serve: anytype) !void {
    const Glm = struct { // GLM's shape: a knob's wait at load, and its pass waits out every admitted prompt
        pub fn warmQuietNow() ?u64 {
            return 7;
        }
        pub const warm_alone = true;
    };
    const Fixed = struct { // Nemotron's: a constant wait
        pub const warm_quiet: u64 = 150;
    };
    const on = try served(Host, serve, Glm, true);
    try std.testing.expect(on.quiet == 7 and on.alone);
    const fixed = try served(Host, serve, Fixed, true);
    try std.testing.expect(fixed.quiet == 150 and !fixed.alone);
    const off = try served(Host, serve, Glm, false);
    try std.testing.expect(off.quiet == 0 and !off.alone); // no warm turns: no wait, nothing waited out
}
