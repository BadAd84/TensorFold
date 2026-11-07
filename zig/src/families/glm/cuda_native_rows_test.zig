//! GLM's load-time settings that depend on the streams served: FP8 pages and the auto shared-row count.
const std = @import("std");
const native = @import("cuda_native.zig");
const ranks = @import("cuda_ranks.zig");
const paging = @import("cuda_start.zig").paging;

test "TF_KV_PAGES with --kv-cache fp8: the same choices as bf16 (FP8 rows lay out on pages, cuda_state.onPages)" {
    for ([_]usize{ 1, 2, 3, 16 }) |w| {
        try std.testing.expectEqual(w >= 2, try paging(null, w, true));
        try std.testing.expectEqual(w >= 2, try paging("auto", w, true));
        try std.testing.expect(!try paging("off", w, true));
        for ([_][]const u8{ "on", "1", "all" }) |v| try std.testing.expect(try paging(v, w, true));
    }
}

fn autoRowsEnv(name: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, name, "TF_GLM_OVN_SHARED_ROWS")) "auto" else null;
}

test "TF_GLM_OVN_SHARED_ROWS=auto: 64 rows above 4 --parallel streams, else 32; GLM's options hold no `lanes`" {
    var problem: []const u8 = "";
    for ([_]u32{ 1, 4, 5, 6, 8 }, [_]u32{ 32, 32, 64, 64, 64 }) |n, rows| {
        const s = try native.settingsOf(&autoRowsEnv, .{ .context = 4096, .drafts = true, .parallel = n, .kv8 = true }, &problem);
        try std.testing.expectEqual(rows, s.shared_rows);
        try std.testing.expect(s.kv8 and ranks.validRows(s.shared_rows));
    }
    try std.testing.expect(!@hasField(native.Options, "lanes")); // a `lanes` field would change the room the streams plan by
}
