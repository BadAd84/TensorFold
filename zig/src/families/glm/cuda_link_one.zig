//! TF_GLM_OVN_LINK1: a step's header and up to put_ids ids cross as one fixed block; LINK_ROCE: by the RoCE gather.

const std = @import("std");
const cuda = @import("cuda");
const follow = @import("cuda_follow.zig");

const Link = follow.Link;
const Header = follow.Header;
const Words = follow.Words;
const Kind = follow.Kind;
const put_ids = follow.put_ids;

/// The fixed block: a step's header and room for `put_ids` ids (what tf_glm_put_words fills).
pub const block_bytes = @sizeOf(Words);

comptime {
    std.debug.assert(block_bytes == 64 + put_ids * 4 and block_bytes % 16 == 0);
}

/// Turns the link to one block a step; with TF_GLM_OVN_LINK_ROCE also makes the gather's rows (world blocks).
pub fn setup(l: *Link, group: *const cuda.tp_group.Group) !void {
    const roce = cuda.ovn.on("LINK_ROCE");
    if (!roce and !cuda.ovn.on("LINK1")) return;
    if (l.buf.len < block_bytes or l.ids.len < put_ids) return error.LinkTooSmall;
    l.one = true;
    if (!roce) return;
    l.ring = try cuda.DeviceBuffer.alloc(l.d, group.world * block_bytes);
    l.group = group;
}

/// A decode round's steps: rank 0 sends the next one at once, so a follower's RoCE wait for it never spans idle time.
pub fn quickNext(kind: u32) bool {
    return kind == @backingInt(Kind.block) or kind == @backingInt(Kind.verify) or kind == @backingInt(Kind.shared) or kind == @backingInt(Kind.draft);
}

/// The block from rank 0 to every rank: by the gather (rank 0's row read) when the last step promised it, else NCCL.
fn block(l: *Link, roce: bool) !void {
    if (roce) return l.group.?.allGather(l.stream, l.buf.ptr, l.ring.?.ptr, block_bytes);
    try l.post(block_bytes);
}

/// Rank 0: the step as one block when its ids fit (put, or uploaded and waited); more ids follow by a second broadcast.
pub fn send(l: *Link, put: cuda.Function, hh: Header, ids: []const u32) !void {
    const roce = l.next_roce;
    var w: Words = undefined;
    @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
    const fits = ids.len <= put_ids;
    if (fits) @memcpy(w.w[16..][0..ids.len], ids);
    if (fits and !l.wait) {
        try l.putWords(put, w, 16 + ids.len);
        try block(l, roce);
    } else {
        const words: usize = if (fits) 16 + ids.len else 16;
        try l.buf.uploadAsync(0, std.mem.sliceAsBytes(w.w[0..words]), l.stream.handle);
        try block(l, roce);
        if (!fits) {
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
            try l.post(64 + ids.len * 4);
        }
        try l.stream.synchronize(); // the uploads read `w` and `ids`
    }
    l.next_roce = l.ring != null and quickNext(hh.kind);
}

/// Other ranks: the block (one download), then the ids past `put_ids` by a second broadcast when the header has more.
pub fn receive(l: *Link) !follow.Step {
    var w: Words = undefined;
    if (l.next_roce) {
        const g = l.group.?;
        try g.allGather(l.stream, l.buf.ptr, l.ring.?.ptr, block_bytes);
        try l.stream.synchronize();
        try g.check(); // a gather that timed out or failed holds nothing valid
        try l.ring.?.download(0, std.mem.asBytes(&w)); // rank 0's row comes first
    } else {
        try l.bcast(block_bytes);
        try l.buf.download(0, std.mem.asBytes(&w));
    }
    var h: Header = undefined;
    @memcpy(std.mem.asBytes(&h), std.mem.sliceAsBytes(w.w[0..16]));
    if (h.n > l.ids.len) return error.TooManyIds;
    if (h.n <= put_ids) {
        @memcpy(l.ids[0..h.n], w.w[16..][0..h.n]);
    } else {
        try l.bcast(64 + h.n * 4);
        try l.buf.download(64, std.mem.sliceAsBytes(l.ids[0..h.n]));
    }
    l.next_roce = l.ring != null and quickNext(h.kind);
    return .{ .h = h, .ids = l.ids[0..h.n] };
}

test "one block holds a step's header and put_ids ids; a decode round's steps promise the next by RoCE" {
    try std.testing.expectEqual(@as(usize, 384), block_bytes);
    try std.testing.expect(quickNext(@backingInt(Kind.block)) and quickNext(@backingInt(Kind.verify)));
    try std.testing.expect(quickNext(@backingInt(Kind.shared)) and quickNext(@backingInt(Kind.draft)));
    inline for (.{ Kind.idle, Kind.end, Kind.keep, Kind.release, Kind.begin, Kind.chunk, Kind.prefill, Kind.disk }) |k| {
        try std.testing.expect(!quickNext(@backingInt(k)));
    }
}
