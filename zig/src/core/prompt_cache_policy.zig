//! Snapshot eviction order, excluding entries borrowed by unfinished prompt passes.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const prefix = @import("prompt_prefix.zig");

/// The entry to free first: a superseded one, oldest; else the oldest (none while `spares_only`: a latest state stays).
pub fn victimBut(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    var best: ?usize = null;
    var oldest: ?usize = null;
    var oldest_rewind = false;
    for (s.entries.items, 0..) |e, i| { // one prefix compare an entry at most (TF_GLM_OVN_KEPT_WALK: only where it decides)
        var on = prefix.Lazy.of(prompt, e.tokens);
        const rewind = !e.expired and e.at == s.active_rewind_at and on.get(); // the active rewind state goes last
        if (e.borrowers == 0 and e != skip and (oldest == null or (if (rewind != oldest_rewind) oldest_rewind else e.used < s.entries.items[oldest.?].used))) {
            oldest = i;
            oldest_rewind = rewind;
        }
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or rewind) continue;
        if (best != null and e.used >= s.entries.items[best.?].used) continue; // not older than the superseded one found
        const now = born > e.born and prompt.len > e.tokens.len; // a longer request's tokens extend it (`on`)
        // gated, the links first (a branch point needs no compare); off, as it always asked
        if (if (prefix.gated()) extended(s, e) or (now and on.get()) else (now and on.get()) or extended(s, e)) best = i;
    }
    return best orelse if (s.spares_only) null else oldest;
}

/// Only live continuations make a prefix a branch point or a superseded state.
pub fn extended(s: *const pc.Store, e: *const pc.Entry) bool {
    if (s.links_lost) return extendedScan(s, e);
    for (e.ext.items) |o| if (!o.expired) return true;
    return false;
}

/// `extended` over every entry, for a store whose links an allocation failure left incomplete.
pub fn extendedScan(s: *const pc.Store, e: *const pc.Entry) bool {
    for (s.entries.items) |o| if (!o.expired and o != e and o.born > e.born and o.tokens.len > e.tokens.len and
        prefix.starts(o.tokens, e.tokens)) return true;
    return false;
}

/// A new entry linked to those it extends and those extending it, later born (tokens and born never change).
pub fn link(s: *pc.Store, x: *pc.Entry) void {
    const proved = prefix.gated();
    if (proved) prove(s, x);
    for (s.entries.items) |o| {
        if (o == x or o.born == x.born or o.tokens.len == x.tokens.len) continue;
        const short, const long = if (o.tokens.len < x.tokens.len) .{ o, x } else .{ x, o };
        if (long.born < short.born) continue;
        if (!(if (proved and short == o) o.seen == s.walks else prefix.starts(long.tokens, short.tokens))) continue;
        short.ext.append(s.gpa, long) catch {
            s.links_lost = true;
        };
    }
}

/// TF_GLM_OVN_KEPT_WALK: marks link's shorter entries that begin `x`, longest first; one a marked one extends is free.
fn prove(s: *pc.Store, x: *const pc.Entry) void {
    s.walks += 1;
    for (s.by_position.items) |o| {
        if (o.born >= x.born) continue; // link asks only of earlier-born entries (and a longer one is never a prefix)
        const by_link = for (o.ext.items) |y| {
            if (y.seen == s.walks) break true;
        } else false;
        if (by_link or prefix.starts(x.tokens, o.tokens)) o.seen = s.walks;
    }
}

/// An entry leaving the store leaves every link to it.
pub fn unlink(s: *pc.Store, x: *pc.Entry) void {
    for (s.entries.items) |o| if (std.mem.indexOfScalar(*pc.Entry, o.ext.items, x)) |i| {
        _ = o.ext.swapRemove(i);
    };
}
