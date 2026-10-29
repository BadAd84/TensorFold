//! Speed-hunt knobs (`TF_GLM_OVN_<NAME>`): read from the environment once per process, default off.
//! Every rank runs with the same environment, so every rank sees the same value.
const std = @import("std");

fn raw(comptime name: []const u8) ?[]const u8 {
    const v = std.c.getenv("TF_GLM_OVN_" ++ name) orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

/// `TF_GLM_OVN_<name>` as text (unset or empty: null), cached after the first read.
pub fn text(comptime name: []const u8) ?[]const u8 {
    const S = struct {
        var read = false;
        var value: ?[]const u8 = null;
    };
    if (!S.read) {
        S.value = raw(name);
        S.read = true;
    }
    return S.value;
}

/// `TF_GLM_OVN_<name>` as an unsigned integer; unset or empty is `default`, a value that does not parse is refused.
pub fn int(comptime name: []const u8, default: u64) error{BadOvnKnob}!u64 {
    const s = text(name) orelse return default;
    return std.fmt.parseInt(u64, s, 10) catch error.BadOvnKnob;
}

/// `TF_GLM_OVN_<name>` as a switch: unset, empty or "0" is off, anything else is on.
pub fn on(comptime name: []const u8) bool {
    const s = text(name) orelse return false;
    return !std.mem.eql(u8, s, "0");
}

test "ovn: an unset knob is off and takes its default" {
    try std.testing.expect(!on("NO_SUCH_KNOB_FOR_TESTS"));
    try std.testing.expectEqual(@as(u64, 7), try int("NO_SUCH_KNOB_FOR_TESTS", 7));
}
