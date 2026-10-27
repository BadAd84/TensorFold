//! The gather region's layout and the peer index maps: the arithmetic the region and the kernel share.
const std = @import("std");

const max_hcas = 2;
pub const max_world_roce = 4; // the most ranks this gather serves (a region holds a slot per peer)
const max_peers = max_world_roce - 1;
const slots = 2;
const flag_stride = 128;

/// Peer p of rank `rank` in a `world` (peers 0..world-2 in ring order; both directions agree on the mapping).
pub fn peerRank(rank: usize, p: usize, world: usize) usize {
    return (rank + 1 + p) % world;
}

/// This rank's index in peer `r`'s peer list (where its writes land in r's region).
pub fn myIndexAt(me: usize, r: usize, world: usize) usize {
    return (me + world - 1 - r) % world;
}

/// Whether two IPv4-mapped GIDs share a /24, the cable joining adjacent ports; IPv6 GIDs never match.
pub fn sameSubnet(a: [16]u8, b: [16]u8) bool {
    const mapped = [12]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff };
    if (!std.mem.eql(u8, a[0..12], &mapped) or !std.mem.eql(u8, b[0..12], &mapped)) return false;
    return a[12] == b[12] and a[13] == b[13] and a[14] == b[14];
}

// -- the region's layout (the same on every rank) ---------------------------------------------------------------------
pub const Layout = struct {
    slot_bytes: usize,
    world: usize,
    recv: usize,
    flags: usize,
    send: usize,
    ctrl: usize,
    total: usize,
    recv_peer_stride: usize, // one peer's receive slots: slots * slot_bytes
    flag_peer_stride: usize, // one peer's flags: slots * max_hcas * flag_stride

    pub fn of(max_bytes: usize, world: usize) Layout {
        const s = std.mem.alignForward(usize, max_bytes, 4096);
        const peers = world - 1;
        const recv = peers * slots * s;
        const flags = recv + peers * slots * max_hcas * flag_stride;
        const send = flags + peers * slots * max_hcas * flag_stride; // after the flags, not past a gap of their size
        return .{ .slot_bytes = s, .world = world, .recv = 0, .flags = flags, .send = send, .ctrl = send + slots * s, .total = send + slots * s + flag_stride, .recv_peer_stride = slots * s, .flag_peer_stride = slots * max_hcas * flag_stride };
    }
};
// ctrl words
pub const c_seq = 0;
pub const c_nbytes = 1; // + slot
pub const c_error = 3;
pub const c_hca = 4;

/// What a rank tells every peer to connect: its region, and per HCA its key, address and queue pair.
pub const Blob = extern struct {
    ok: u32,
    n_hca: u32,
    slot_bytes: u64,
    max_bytes: u64, // the largest shard this rank would send this way (every rank uses the smallest)
    addr: u64,
    rkey: [max_hcas]u32,
    qpn: [max_peers][max_hcas]u32, // peer p's queue pair on HCA h: the peer at index p connects to qpn[p][h]
    mtu: [max_hcas]u32,
    lid: [max_hcas]u16,
    gid: [max_hcas][16]u8,
};

test "RoCE gather: every rank agrees one cap" {
    // 511 KiB and 512 KiB round to the same slot; a 512 KiB shard must take the same path on every rank
    try std.testing.expectEqual(Layout.of(511 << 10, 2).slot_bytes, Layout.of(512 << 10, 2).slot_bytes);
    try std.testing.expectEqual(Layout.of(511 << 10, 3).slot_bytes, Layout.of(512 << 10, 4).slot_bytes);
    const a = @min(@as(u64, 511 << 10), @as(u64, 512 << 10));
    const b = @min(@as(u64, 512 << 10), @as(u64, 511 << 10));
    try std.testing.expectEqual(a, b);
    try std.testing.expectEqual(@as(u64, 511 << 10), a);
}

test "RoCE gather: peer and slot indices agree at every world" {
    for (2..max_world_roce + 1) |w| for (0..w) |me| {
        var seen: [max_world_roce]bool = @splat(false);
        for (0..w - 1) |p| {
            const r = peerRank(me, p, w);
            try std.testing.expect(r != me);
            try std.testing.expect(r < w);
            seen[r] = true;
            // a write to rank r lands in the slot r's peer list holds for me
            try std.testing.expectEqual(me, peerRank(r, myIndexAt(me, r, w), w));
        }
        // A rank is its own peer never, so the w-1 peers must cover exactly the other ranks.
        for (seen[0..w], 0..) |s, i| if (i != me) try std.testing.expect(s) else try std.testing.expect(!s);
    };
}
