//! Speed-hunt knobs (`TF_GLM_OVN_<NAME>`): read from the environment once per process, default off.
//! Every rank runs with the same environment, so every rank sees the same value.
const std = @import("std");

/// One knob's cache: `key` captures `name`, so every knob gets its own type and its own statics (a struct that
/// captured nothing would be one type for every name, and the first knob read would answer for all of them).
fn Cache(comptime name: []const u8) type {
    return struct {
        const key = "TF_GLM_OVN_" ++ name;
        var read = false;
        var value: ?[]const u8 = null;
    };
}

/// `TF_GLM_OVN_<name>` as text (unset or empty: null), cached after the first read.
pub fn text(comptime name: []const u8) ?[]const u8 {
    const S = Cache(name);
    if (!S.read) {
        const v = std.c.getenv(S.key);
        const s = if (v) |p| std.mem.span(p) else "";
        S.value = if (s.len == 0) null else s;
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

test "ovn: each knob keeps its own value (one knob read first never answers for another)" {
    Cache("TEST_A").read = true;
    Cache("TEST_A").value = "8";
    Cache("TEST_B").read = true;
    Cache("TEST_B").value = null;
    try std.testing.expectEqual(@as(u64, 8), try int("TEST_A", 0));
    try std.testing.expect(!on("TEST_B"));
    try std.testing.expect(on("TEST_A"));
}
