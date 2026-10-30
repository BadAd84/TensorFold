//! The RoCE gather's rails: which HCAs a rank stripes over, how two ranks pair them, and where the proxy runs.

const std = @import("std");
const builtin = @import("builtin");
const ovn = @import("ovn.zig");

/// The most HCAs a rank stripes over: four with TF_GLM_OVN_ROCE_RAILS=2 (a triangle's two rails a peer).
pub const max_hcas = 4;
/// The HCAs a rank stripes over otherwise: the first two names (one rail a peer on a triangle).
pub const default_hcas = 2;

/// TF_GLM_OVN_ROCE_RAILS: 1 or unset keeps the first two HCAs, 2 takes up to four (two rails a peer); others refused.
pub fn wanted() error{BadOvnKnob}!bool {
    return switch (try ovn.int("ROCE_RAILS", 1)) {
        1 => false,
        2 => true,
        else => error.BadOvnKnob,
    };
}

/// TF_GLM_OVN_ROCE_RAILS_KB: with two rails, a shard under this many KiB a rank takes one link (0: every one stripes).
pub fn stripeMinKb() error{BadOvnKnob}!u32 {
    const kb = try ovn.int("ROCE_RAILS_KB", 0);
    if (kb >= 1 << 20) return error.BadOvnKnob;
    return @intCast(kb);
}

/// The HCAs to stripe over, at most `limit`: TF_ROCE_HCA, else NCCL_IB_HCA without "=^" prefixes or ":port" suffixes.
pub fn hcaNames(io: std.Io, gpa: std.mem.Allocator, names: *[max_hcas][64]u8, lens: *[max_hcas]usize, limit: usize) !u32 {
    _ = io;
    _ = gpa;
    const raw = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA") orelse return error.NoHca;
    var n: u32 = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
    while (it.next()) |w| {
        if (n == limit) break;
        var s = std.mem.trimStart(u8, std.mem.trim(u8, w, " "), "=^");
        if (std.mem.indexOfScalar(u8, s, ':')) |c| s = s[0..c];
        if (s.len == 0 or s.len > 63) continue;
        @memcpy(names[n][0..s.len], s);
        lens[n] = s.len;
        n += 1;
    }
    if (n == 0) return error.NoHca;
    return n;
}

/// The HCA's RoCE v2 IPv4 GID (TF_ROCE_GID_INDEX overrides): its index differs between hosts' tables.
pub fn gidIndex(io: std.Io, gpa: std.mem.Allocator, name: []const u8) !c_int {
    if (std.c.getenv("TF_ROCE_GID_INDEX")) |v| return std.fmt.parseInt(c_int, std.mem.span(v), 10) catch error.NoHca;
    const cwd = std.Io.Dir.cwd();
    var i: c_int = 0;
    while (i < 16) : (i += 1) {
        var pbuf: [160]u8 = undefined;
        const tpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ name, i });
        const t = cwd.readFileAlloc(io, tpath, gpa, .limited(64)) catch continue;
        defer gpa.free(t);
        if (!std.mem.startsWith(u8, t, "RoCE v2")) continue;
        const gpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gids/{d}", .{ name, i });
        const gid = cwd.readFileAlloc(io, gpath, gpa, .limited(64)) catch continue;
        defer gpa.free(gid);
        if (std.mem.startsWith(u8, gid, "0000:0000:0000:0000:0000:ffff:")) return i;
    }
    return error.NoHca;
}

/// Whether two IPv4-mapped RoCE GIDs share a /24, the cable joining two ports (IPv6 GIDs never: index-h pairing).
pub fn sameSubnet(a: [16]u8, b: [16]u8) bool {
    const mapped = [12]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff };
    if (!std.mem.eql(u8, a[0..12], &mapped) or !std.mem.eql(u8, b[0..12], &mapped)) return false;
    return a[12] == b[12] and a[13] == b[13] and a[14] == b[14];
}

/// Rank a's HCAs paired with b's by a shared /24 (b's each once, in a's order, at most `limit`), else index-h.
pub fn pairs(a: anytype, b: anytype, limit: usize, mine: *[max_hcas]u8, theirs: *[max_hcas]u8) usize {
    var n: usize = 0;
    for (0..a.n_hca) |h| {
        var q: usize = 0;
        var best: ?usize = null;
        while (q < b.n_hca) : (q += 1) {
            if (!sameSubnet(a.gid[h], b.gid[q])) continue;
            var taken = false;
            for (theirs[0..n]) |u| {
                if (@as(usize, u) == q) taken = true;
            }
            if (taken) continue;
            best = q;
            break;
        }
        if (best) |pq| {
            mine[n] = @intCast(h);
            theirs[n] = @intCast(pq);
            n += 1;
            if (n == limit) break;
        }
    }
    if (n == 0) { // no shared subnet: keep the plain index-h pairing
        n = @min(@as(usize, a.n_hca), limit);
        for (0..n) |i| {
            mine[i] = @intCast(i);
            theirs[i] = @intCast(i);
        }
    }
    return n;
}

/// The fewest links any two ranks pair: every rank holds every blob, so every rank finds the same count.
pub fn leastLinks(all: anytype, limit: usize) usize {
    var least = limit;
    for (all, 0..) |a, r| {
        for (all, 0..) |b, q| {
            if (r == q) continue;
            var mine: [max_hcas]u8 = undefined;
            var theirs: [max_hcas]u8 = undefined;
            least = @min(least, pairs(a, b, limit, &mine, &theirs));
        }
    }
    return least;
}

/// The first `k` of `n` pairs in the lower rank's HCA order, so both ends of every cable keep the same pairs.
pub fn keep(mine: *[max_hcas]u8, theirs: *[max_hcas]u8, n: usize, k: usize, peer_lower: bool) usize {
    if (n <= k) return n;
    if (peer_lower) { // mine ascend already; the peer keeps its first k, so order by its HCAs
        var i: usize = 1;
        while (i < n) : (i += 1) {
            var j = i;
            while (j > 0 and theirs[j - 1] > theirs[j]) : (j -= 1) {
                std.mem.swap(u8, &theirs[j - 1], &theirs[j]);
                std.mem.swap(u8, &mine[j - 1], &mine[j]);
            }
        }
    }
    return k;
}

/// A core set as cpu_set_t holds it on a 64-bit host: cores 0 to 1023.
pub const Cpus = [16]u64;

/// TF_GLM_OVN_ROCE_CPU: the cores the RoCE proxy thread runs on ("4", "4,5", "10-13"); unset: null (anywhere).
pub fn proxyCpus() error{BadOvnKnob}!?Cpus {
    const s = ovn.text("ROCE_CPU") orelse return null;
    return parseCpus(s) orelse error.BadOvnKnob;
}

/// A comma list of cores and ranges as a set, or null when it names none or does not parse.
fn parseCpus(s: []const u8) ?Cpus {
    var set: Cpus = @splat(0);
    var any = false;
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |w| {
        const dash = std.mem.indexOfScalar(u8, w, '-');
        const lo = std.fmt.parseInt(u16, if (dash) |d| w[0..d] else w, 10) catch return null;
        var hi = lo;
        if (dash) |d| hi = std.fmt.parseInt(u16, w[d + 1 ..], 10) catch return null;
        if (hi < lo or hi >= 1024) return null;
        for (lo..@as(usize, hi) + 1) |c| set[c / 64] |= @as(u64, 1) << @intCast(c % 64);
        any = true;
    }
    return if (any) set else null;
}

/// Pins the calling thread to `cpus` on Linux (elsewhere nothing); a refusal is a warning, the thread runs on.
pub fn pin(cpus: Cpus) void {
    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var set: linux.cpu_set_t = @splat(0);
        const n = @min(@sizeOf(linux.cpu_set_t), @sizeOf(Cpus));
        @memcpy(std.mem.asBytes(&set)[0..n], std.mem.asBytes(&cpus)[0..n]);
        linux.sched_setaffinity(0, &set) catch std.log.warn("RoCE all-gather: the proxy could not be pinned to TF_GLM_OVN_ROCE_CPU", .{});
    }
}

const TestBlob = struct { n_hca: u32, gid: [max_hcas][16]u8 };

fn testBlob(subnets: []const u8) TestBlob {
    var b: TestBlob = .{ .n_hca = @intCast(subnets.len), .gid = @splat(@splat(0)) };
    for (subnets, 0..) |net, h| {
        b.gid[h][10] = 0xff;
        b.gid[h][11] = 0xff;
        b.gid[h][12] = 10;
        b.gid[h][13] = net;
        b.gid[h][14] = 1;
        b.gid[h][15] = @intCast(h + 1);
    }
    return b;
}

test "RoCE rails: a crossed triangle pairs one link a peer over two HCAs and two over four" {
    // HCAs in NCCL_IB_HCA order: port 0 and port 1 of one PCIe half, then of the other; cables in subnets 1, 2, 3
    const four = [_]TestBlob{ testBlob(&.{ 1, 3, 1, 3 }), testBlob(&.{ 2, 1, 2, 1 }), testBlob(&.{ 3, 2, 3, 2 }) };
    const two = [_]TestBlob{ testBlob(&.{ 1, 3 }), testBlob(&.{ 2, 1 }), testBlob(&.{ 3, 2 }) };
    try std.testing.expectEqual(@as(usize, 1), leastLinks(&two, default_hcas));
    try std.testing.expectEqual(@as(usize, 2), leastLinks(&four, max_hcas));
    var mine: [max_hcas]u8 = undefined;
    var theirs: [max_hcas]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 2), pairs(four[0], four[1], max_hcas, &mine, &theirs));
    try std.testing.expectEqualSlices(u8, &.{ 0, 2 }, mine[0..2]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 3 }, theirs[0..2]);
}

test "RoCE rails: truncating to the fewest links keeps the same pairs on both ends" {
    // two cables between ranks 0 and 1 whose subnets come in opposite HCA orders; rank 2 shares one with each
    const all = [_]TestBlob{ testBlob(&.{ 5, 6, 7, 1 }), testBlob(&.{ 6, 5, 8, 2 }), testBlob(&.{ 7, 8, 3, 4 }) };
    const k = leastLinks(&all, max_hcas);
    try std.testing.expectEqual(@as(usize, 1), k);
    var m0: [max_hcas]u8 = undefined;
    var t0: [max_hcas]u8 = undefined;
    var m1: [max_hcas]u8 = undefined;
    var t1: [max_hcas]u8 = undefined;
    const n0 = keep(&m0, &t0, pairs(all[0], all[1], max_hcas, &m0, &t0), k, false);
    const n1 = keep(&m1, &t1, pairs(all[1], all[0], max_hcas, &m1, &t1), k, true);
    try std.testing.expectEqual(n0, n1);
    for (0..n0) |i| {
        try std.testing.expectEqual(m0[i], t1[i]);
        try std.testing.expectEqual(t0[i], m1[i]);
    }
}

test "RoCE rails: the proxy's cores parse as a list of cores and ranges" {
    const s = parseCpus("4,10-12").?;
    try std.testing.expectEqual(@as(u64, 1 << 4 | 1 << 10 | 1 << 11 | 1 << 12), s[0]);
    try std.testing.expect(parseCpus("") == null);
    try std.testing.expect(parseCpus("3-1") == null);
    try std.testing.expect(parseCpus("x") == null);
    try std.testing.expect(parseCpus("1024") == null);
    try std.testing.expectEqual(@as(u64, 1), parseCpus("64").?[1]);
}
