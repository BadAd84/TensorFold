//! TF_GLM_OVN_DISK_CARRY's replay on GLM images: one state placed two ways (KDA copies, ring bases) matches itself.
const std = @import("std");
const image = @import("cuda_disk_image.zig");
const sparse = @import("cuda_sparse.zig");
const Match = @import("lanes").rows_image.Match;
const Words = image.Words;
const a = std.testing.allocator;

/// A small shape: a prefix of a position word, conv states and three KDA layers' two copies, then a window's scratch.
const sh: image.Shape = .{
    .prefix = 320,
    .tail = 24,
    .layers = 2,
    .latent = 8,
    .scales = false,
    .ring_rows = 16,
    .ring_row = 4,
    .pk_rows = 64 / sparse.pool + 2,
    .pooled_row = 6,
    .capacity = 64,
    .head = true,
    .kda = .{ .pos = 0, .conv = 16, .conv_len = 24, .rec = 64, .rec_layer = 32, .layers = 3 },
    .hidden = 8,
    .hidden_len = 10,
};
const at: u32 = 30;
const mtp_len: u64 = 29;

/// The state's byte `i` of field `k`: what it holds, wherever it is placed.
fn held(k: u64, i: u64) u8 {
    return @truncate(std.hash.Wyhash.hash(k, std.mem.asBytes(&i)));
}

/// The state as words `w` place it, every byte it does not hold set to `noise` (stale copies, dead rows, scratch).
fn place(w: Words, noise: u8) ![]u8 {
    var parts: std.ArrayList(image.Part) = .empty;
    defer parts.deinit(a);
    try image.parts(sh, at, &parts, a);
    var offs: std.ArrayList(u64) = .empty;
    defer offs.deinit(a);
    var o: u64 = 0;
    for (parts.items) |p| {
        try offs.append(a, o);
        o += p.bytes;
    }
    const out = try a.alloc(u8, @intCast(o));
    @memset(out, noise);
    const words = w.encode();
    @memcpy(out[0..Words.bytes], std.mem.sliceAsBytes(&words));
    const pre = offs.items[1];
    std.mem.writeInt(u32, out[@intCast(pre)..][0..4], at, .little);
    for (0..sh.kda.conv_len) |i| out[@intCast(pre + sh.kda.conv + i)] = held(1, i);
    for (0..sh.kda.layers) |li| {
        const h = (w.cur >> @intCast(li)) & 1; // the copy its last pass wrote
        const live = pre + sh.kda.rec + (h * sh.kda.layers + li) * sh.kda.rec_layer;
        for (0..sh.kda.rec_layer) |j| out[@intCast(live + j)] = held(2 + li, j);
    }
    for (0..sh.layers) |d| try cache(out, offs.items[2 + d * 5 ..][0..4].*, at, w.base, 10 * (d + 1));
    try cache(out, offs.items[2 + sh.layers * 5 ..][0..4].*, mtp_len, w.mtp_base, 100);
    const tail = offs.items[offs.items.len - 1];
    std.mem.writeInt(u32, out[@intCast(tail)..][0..4], @intCast(mtp_len), .little);
    for (0..sh.hidden_len) |j| out[@intCast(tail + sh.hidden + j)] = held(200, j);
    return out;
}

/// One cache of `n` rows at its parts' offsets (latent, ring, ring, pooled): rows by position, a ring's from `base`.
fn cache(out: []u8, offs: [4]u64, n: u64, base: u64, k: u64) !void {
    for (0..n * sh.latent) |i| out[@intCast(offs[0] + i)] = held(k, i);
    for (1..3) |q| for (base..n) |p| for (0..sh.ring_row) |j| {
        out[@intCast(offs[q] + (p - base) * sh.ring_row + j)] = held(k + q, p * sh.ring_row + j);
    };
    for (0..n / sparse.pool * sh.pooled_row) |i| out[@intCast(offs[3] + i)] = held(k + 3, i);
}

/// The first stored byte whose match differs in `fresh` (null: every matched byte agrees).
fn differs(stored: []const u8, fresh: []const u8, s: Words, f: Words) !?u64 {
    var ms: std.ArrayList(Match) = .empty;
    defer ms.deinit(a);
    try image.canon(sh, at, s, f, &ms, a);
    var first: ?u64 = null;
    for (ms.items) |m| if (std.mem.indexOfDiff(u8, stored[@intCast(m.off)..][0..@intCast(m.len)], fresh[@intCast(m.fresh)..][0..@intCast(m.len)])) |i| {
        if (first == null or m.off + i < first.?) first = m.off + i;
    };
    return first;
}

const stored_w: Words = .{ .at = at, .mtp_len = mtp_len, .base = 16, .mtp_base = 16, .cur = 0b101 };
const fresh_w: Words = .{ .at = at, .mtp_len = mtp_len, .base = 20, .mtp_base = 24, .cur = 0b011 };

test "glm canon: one state placed with other KDA copies and ring bases matches itself; in place it would not" {
    const s = try place(stored_w, 0x11);
    defer a.free(s);
    const f = try place(fresh_w, 0x77);
    defer a.free(f);
    try std.testing.expectEqual(@as(?u64, null), try differs(s, f, stored_w, fresh_w));
    try std.testing.expect(std.mem.indexOfDiff(u8, s, f) != null); // byte for byte: the raw images part
    // the same placement both ways matches too, the stale bytes aside
    const again = try place(stored_w, 0x33);
    defer a.free(again);
    try std.testing.expectEqual(@as(?u64, null), try differs(s, again, stored_w, stored_w));
}

/// Image offset of KDA layer `li`'s copy `h`.
fn copyAt(li: u64, h: u64) u64 {
    return Words.bytes + sh.kda.rec + (h * sh.kda.layers + li) * sh.kda.rec_layer;
}

test "glm canon: a byte of a layer's live copy that differs refuses; one in its stale copy does not" {
    const s = try place(stored_w, 0x11);
    defer a.free(s);
    const f = try place(fresh_w, 0x11); // the stale copies alike on both sides
    defer a.free(f);
    const live = copyAt(1, 0) + 5; // layer 1 holds copy 0 in the stored state (0b101), copy 1 in the fresh one
    s[@intCast(live)] +%= 1;
    try std.testing.expectEqual(@as(?u64, live), try differs(s, f, stored_w, fresh_w));
    s[@intCast(live)] -%= 1;
    const stale = copyAt(1, 1) + 5; // the stored state's other copy of layer 1
    s[@intCast(stale)] +%= 1;
    try std.testing.expectEqual(@as(?u64, null), try differs(s, f, stored_w, fresh_w));
}

test "glm canon: ring rows both states hold are compared by position; a row one of them dropped is not" {
    var parts: std.ArrayList(image.Part) = .empty;
    defer parts.deinit(a);
    try image.parts(sh, at, &parts, a);
    var ring: u64 = 0; // the first DSA layer's first ring: after the words, the prefix and its latent rows
    for (parts.items[0..3]) |p| ring += p.bytes;
    const s = try place(stored_w, 0x11);
    defer a.free(s);
    const f = try place(fresh_w, 0x11);
    defer a.free(f);
    const both = ring + (25 - stored_w.base) * sh.ring_row; // position 25: both rings hold it
    s[@intCast(both)] +%= 1;
    try std.testing.expectEqual(@as(?u64, both), try differs(s, f, stored_w, fresh_w));
    s[@intCast(both)] -%= 1;
    const dropped = ring + (17 - stored_w.base) * sh.ring_row; // position 17: the fresh ring was rebased past it
    s[@intCast(dropped)] +%= 1;
    try std.testing.expectEqual(@as(?u64, null), try differs(s, f, stored_w, fresh_w));
    // the words' head rows differ: refused at the words, nothing past them lined up
    var other = fresh_w;
    other.mtp_len -= 1;
    const g = try place(other, 0x11);
    defer a.free(g);
    try std.testing.expectEqual(@as(?u64, 16), try differs(s, g, stored_w, other));
}

/// The image offset of the prefix's first byte past the KDA layers' copies: a decode window's scratch (State.proj).
const scratch: u64 = Words.bytes + sh.kda.rec + 2 * sh.kda.layers * sh.kda.rec_layer;

test "glm heldDiff: one state with other window scratch and stale bytes is the same state; raw bytes part at proj" {
    const s = try place(stored_w, 0x11);
    defer a.free(s);
    const f = try a.dupe(u8, s);
    defer a.free(f);
    @memset(f[@intCast(scratch)..][0..@intCast(Words.bytes + sh.prefix - scratch)], 0x77); // another window's rows
    try std.testing.expectEqual(@as(?u64, scratch), std.mem.indexOfDiff(u8, s, f)); // glm-lineage-check's failure shape
    try std.testing.expectEqual(@as(?u64, null), try image.heldDiff(sh, at, s, f, a));
    const g = try place(stored_w, 0x77); // every byte the state does not hold another
    defer a.free(g);
    try std.testing.expectEqual(@as(?u64, null), try image.heldDiff(sh, at, s, g, a));
}

test "glm heldDiff: a held byte that differs is found (conv, a live KDA copy, a latent row, a ring base word)" {
    const s = try place(stored_w, 0x11);
    defer a.free(s);
    const f = try place(stored_w, 0x77);
    defer a.free(f);
    const latent = Words.bytes + sh.prefix + 3 * sh.latent + 1; // the first DSA layer's row 3
    for ([_]u64{ Words.bytes + sh.kda.conv + 7, copyAt(1, 0) + 5, latent }) |off| {
        f[@intCast(off)] +%= 1;
        try std.testing.expectEqual(@as(?u64, off), try image.heldDiff(sh, at, s, f, a));
        f[@intCast(off)] -%= 1;
    }
    var other = stored_w;
    other.base -= 2; // the same rows in a ring placed elsewhere: its words say so, and the images are not alike
    const g = try place(other, 0x11);
    defer a.free(g);
    try std.testing.expectEqual(@as(?u64, 24), try image.heldDiff(sh, at, s, g, a));
}
