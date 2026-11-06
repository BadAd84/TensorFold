//! A CUDA family's prompt cache in the native server: rules, budget and copies from the family's `snap_*` decls.
const std = @import("std");
const api = @import("engine_api");

/// Any row, the head one token on (Nemotron's); a family's snap_grid (or snapGridNow's), snap_lookahead, warmQuiet,
/// snap_fork and snap_tail if set.
pub fn cacheRules(comptime F: type) api.prompt_cache.Rules {
    var rules: api.prompt_cache.Rules = .{ .lookahead = 1, .min_prompt = 0 };
    if (@hasDecl(F, "snap_lookahead")) rules.lookahead = F.snap_lookahead;
    if (warmQuiet(F) != null) rules.warm = .waits;
    if (@hasDecl(F, "snap_fork")) rules.fork = F.snap_fork; // opt-in, a family at a time after its device check
    if (@hasDecl(F, "snap_tail")) rules.tail = if (@import("core").profile.lookup(F.snap_tail)) |v| std.mem.eql(u8, v, "1") else false;
    if (@hasDecl(F, "snap_grid")) {
        rules.planned = true;
        rules.grid = F.snap_grid;
    }
    if (@hasDecl(F, "snapGridNow")) if (F.snapGridNow()) |g| { // a grid only some settings need, read at load
        rules.planned = true;
        rules.grid = g;
    };
    return rules;
}

/// Native CUDA families own their grid, independently of the server's template chunk starts.
pub fn passRules(comptime F: type, prompt_rows: u32) api.prompt_cache.Rules {
    var rules = cacheRules(F);
    rules.chunk = prompt_rows;
    if (@hasDecl(F, "snap_repeat_min")) rules.repeat_min = if (std.c.getenv("TF_FLASHNEXT_REPEAT_CACHE")) |v| (if (std.mem.eql(u8, std.mem.span(v), "0")) 0 else F.snap_repeat_min) else F.snap_repeat_min;
    rules.chunk_starts = false;
    return rules;
}

/// A family's warm turns' wait for quiet: its warm_quiet, or warmQuietNow's (a knob read at load); null: no warm turns.
pub fn warmQuiet(comptime F: type) ?u64 {
    if (@hasDecl(F, "warm_quiet")) return F.warm_quiet;
    if (@hasDecl(F, "warmQuietNow")) return F.warmQuietNow();
    return null;
}

/// The cache's bytes less what the family sets aside for its states (`snap_ready`, as Nemotron's spare buffers).
pub fn cacheBudget(comptime F: type, target: anytype, cache: u64) !u64 {
    return if (@hasDecl(F, "snap_ready")) cache - try F.snap_ready(target, cache) else cache;
}

/// The family's `snap_*` copies and `image`, its disk rows (null: none); a container-level value, so the pointer holds.
pub fn Snaps(comptime F: type, comptime image: ?*const api.prompt_cache.rows.Image) type {
    return struct {
        pub const vtable: api.prompt_cache.Snapshots.VTable = .{
            .bytes = F.snap_bytes,
            .save = F.snap_save,
            .restore = F.snap_restore,
            .drop = F.snap_drop,
            .rows = image,
            .charged = if (@hasDecl(F, "snap_charged")) F.snap_charged else null,
            .spare = if (@hasDecl(F, "snap_spare")) F.snap_spare else null,
            .reuses = if (@hasDecl(F, "snap_reuses")) F.snap_reuses else null,
        };
    };
}

test "a family with snap_grid keeps on its chunk starts, any other at any row as Nemotron does" {
    const Glm = struct {
        pub const snap_grid: u32 = 2048;
    };
    const Other = struct {};
    const Warm = struct {
        pub const warm_quiet: u64 = 1;
    };
    const Headless = struct {
        pub const snap_lookahead: u32 = 0;
    };
    try std.testing.expectEqual(@as(u32, 0), cacheRules(Headless).lookahead);
    const any = cacheRules(Other);
    try std.testing.expect(!any.planned and any.min_prompt == 0 and any.lookahead == 1 and any.warm == .off);
    try std.testing.expect(cacheRules(Warm).warm == .waits and cacheRules(Glm).warm == .off);
    const Stub = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 0;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
    };
    var unused: u8 = 0;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const prompt = try arena.allocator().alloc(u32, 3547);
    for (prompt, 0..) |*t, i| t.* = @intCast(i);
    const vtable: api.prompt_cache.Snapshots.VTable = .{
        .bytes = Stub.bytes,
        .save = Stub.save,
        .restore = Stub.restore,
        .drop = Stub.drop,
    };
    const copies: api.prompt_cache.Snapshots = .{ .ptr = &unused, .vtable = &vtable };
    var rules = cacheRules(Glm);
    rules.tail = true; // the family's setting on
    var store = api.prompt_cache.Store.init(std.testing.allocator, copies, rules, 1 << 30);
    defer store.deinit();
    const marks = try store.marks(arena.allocator(), prompt, 0, 3540, &.{}, &.{}, &.{});
    try std.testing.expectEqualSlices(u32, &[_]u32{2048}, marks);
    // a raw prompt names no history: its tail keeps the same chunk start
    try std.testing.expectEqualSlices(u32, &[_]u32{2048}, try store.marks(arena.allocator(), prompt, 0, 0, &.{}, &.{}, &.{}));
    // snap_tail names the setting that turns the tail on at 1 (GLM refuses values but 0 and 1 at startup)
    const env = @import("sys").env;
    const Tail = struct {
        pub const snap_tail = "TF_SNAPS_TEST_TAIL";
    };
    defer _ = env.unsetenv(Tail.snap_tail);
    for ([_]?[:0]const u8{ null, "", "0", "2", "1" }) |v| {
        _ = if (v) |s| env.setenv(Tail.snap_tail, s, 1) else env.unsetenv(Tail.snap_tail);
        const on = v != null and std.mem.eql(u8, v.?, "1");
        try std.testing.expect(cacheRules(Tail).tail == on and passRules(Tail, 2048).tail == on);
    }
    try std.testing.expect(!cacheRules(Other).tail);
}

test "a family's snapGridNow plans its states on that grid only when it names one" {
    const On = struct {
        pub fn snapGridNow() ?u32 {
            return 32;
        }
    };
    const Off = struct {
        pub fn snapGridNow() ?u32 {
            return null;
        }
    };
    const on = cacheRules(On);
    try std.testing.expect(on.planned and on.grid == 32);
    const off = cacheRules(Off);
    try std.testing.expect(!off.planned and off.grid == 0);
}

test "a family's warmQuietNow turns warm turns on only when it names a wait" {
    const On = struct {
        pub fn warmQuietNow() ?u64 {
            return 7;
        }
    };
    const Off = struct {
        pub fn warmQuietNow() ?u64 {
            return null;
        }
    };
    const Fixed = struct {
        pub const warm_quiet: u64 = 5;
    };
    try std.testing.expectEqual(@as(?u64, 7), warmQuiet(On));
    try std.testing.expectEqual(@as(?u64, null), warmQuiet(Off));
    try std.testing.expectEqual(@as(?u64, 5), warmQuiet(Fixed));
    try std.testing.expectEqual(@as(?u64, null), warmQuiet(struct {}));
    try std.testing.expect(cacheRules(On).warm == .waits and cacheRules(Off).warm == .off and cacheRules(Fixed).warm == .waits);
}

test {
    _ = @import("cuda_snaps_test.zig");
}
