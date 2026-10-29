//! Connect rank liveness before resolving page settings or constructing the engine.
const std = @import("std");

/// Every rank resolves its page mode within the same watched failure boundary as engine construction.
pub fn open(io: std.Io, p: anytype, text: ?[]const u8, initial: anytype, watch: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    if (p.world > 1) try watch.start(io, p.rank, p.world);
    return resolved(p, text, initial, args, after) catch |err| {
        watch.fail(@errorName(err));
        return err;
    };
}

fn resolved(p: anytype, text: ?[]const u8, initial: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    var settings = initial;
    settings.pages = paging(text, p.world);
    return after(args, settings);
}

/// TF_KV_PAGES: auto pages from two ranks (BF16 or FP8 KV alike: cuda_state.onPages), off or 0 keeps windows.
pub fn paging(text: ?[]const u8, world: usize) bool {
    const v = text orelse "auto";
    if (std.mem.eql(u8, v, "auto")) return world >= 2;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return false;
    return true;
}
