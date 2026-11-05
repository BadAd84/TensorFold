//! A request's `cache_salt` (vLLM's field): kept states are resumed only by requests carrying the same salt.
//! The salt never reaches the model. It becomes a prefix of the request's cache KEY, the token stream kept states
//! are matched against, encoded as values above every vocabulary id (the image key's range), so a salted request can
//! prefix-match neither an unsalted state nor one kept under another salt. A benchmark sends a fresh salt per cold
//! prompt and reads a true cold prefill each time while the states stay in the store for their own salt.
const std = @import("std");

/// The first key value a salt may take: above every vocabulary id, the range image keys use (image_key.zig).
pub const base: u32 = 0x9000_0000;
/// A salt's key length: one opener, then the hash's words.
pub const key_len: usize = 3;

/// `cache_salt` as a request body holds it: null when absent or JSON null, the string otherwise; an empty string or a
/// non-string is an error the caller refuses before any stream header goes out.
pub const Salt = union(enum) { none, some: []const u8, bad: []const u8 };

pub fn read(field: anytype) Salt {
    const v = field orelse return .none;
    return switch (v) {
        .null => .none,
        .string => |s| if (s.len == 0) .{ .bad = "cache_salt must be a non-empty string" } else .{ .some = s },
        else => .{ .bad = "cache_salt must be a non-empty string" },
    };
}

/// The key prefix a salt contributes: `key_len` values, all at or above `base`, distinct for distinct salts.
pub fn prefix(salt: []const u8) [key_len]u32 {
    const h = std.hash.Wyhash.hash(0x5a17, salt);
    return .{ base, base | (@as(u32, @truncate(h)) & 0x0fff_ffff), base | (@as(u32, @truncate(h >> 32)) & 0x0fff_ffff) };
}

/// The request's cache key with the salt in front: `key` is the key the renderer built (the prompt when empty).
pub fn salted(a: std.mem.Allocator, salt: []const u8, key: []const u32, prompt: []const u32) ![]u32 {
    const plain = if (key.len > 0) key else prompt;
    const out = try a.alloc(u32, key_len + plain.len);
    const p = prefix(salt);
    @memcpy(out[0..key_len], &p);
    @memcpy(out[key_len..], plain);
    return out;
}

test "a salt's prefix sits above every vocabulary id and differs between salts" {
    const a = prefix("alpha");
    const b = prefix("beta");
    for (a) |v| try std.testing.expect(v >= base);
    try std.testing.expect(!std.mem.eql(u32, &a, &b));
    try std.testing.expectEqualSlices(u32, &a, &prefix("alpha"));
}

test "a salted key is the salt's prefix then the plain key, and two salts never share a prefix with the unsalted key" {
    const prompt = [_]u32{ 10, 20, 30 };
    const s1 = try salted(std.testing.allocator, "one", &.{}, &prompt);
    defer std.testing.allocator.free(s1);
    const s2 = try salted(std.testing.allocator, "two", &.{}, &prompt);
    defer std.testing.allocator.free(s2);
    try std.testing.expectEqual(key_len + prompt.len, s1.len);
    try std.testing.expectEqualSlices(u32, &prompt, s1[key_len..]);
    try std.testing.expect(!std.mem.startsWith(u32, s1, &prompt));
    try std.testing.expect(!std.mem.startsWith(u32, s2, s1[0..key_len]));
}

test "read: absent and null are none, a string is some, empty or non-string is bad" {
    const json = @import("json");
    try std.testing.expect(read(@as(?json.Value, null)) == .none);
    try std.testing.expect(read(@as(?json.Value, .null)) == .none);
    try std.testing.expect(read(@as(?json.Value, .{ .string = "x" })) == .some);
    try std.testing.expect(read(@as(?json.Value, .{ .string = "" })) == .bad);
    try std.testing.expect(read(@as(?json.Value, .{ .bool = true })) == .bad);
}
