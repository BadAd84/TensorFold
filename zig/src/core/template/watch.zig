//! A render that stops once its output has run `slack` bytes past where it first differs from a reference text.
const std = @import("std");
const v = @import("value.zig");

/// Output is append-only, so a stopped render's text and spans are a prefix of the full render's.
pub const Watch = struct {
    against: []const u8,
    slack: usize,
    out: ?*const v.Buffer = null,
    diverged: ?usize = null,
    stopped: bool = false,

    /// The root output grew from `from` bytes: fails the render, stopped, once it is `slack` past the difference.
    pub fn grew(w: *Watch, rt: *v.Rt, out: *const v.Buffer, from: usize) v.Error!void {
        const text = out.text.items;
        if (w.diverged == null) {
            const ref = w.against[@min(from, w.against.len)..];
            const new = text[from..];
            const n = @min(new.len, ref.len);
            if (std.mem.indexOfDiff(u8, new[0..n], ref[0..n])) |i| {
                w.diverged = from + i;
            } else if (new.len > ref.len) w.diverged = from + ref.len;
        }
        const d = w.diverged orelse return;
        if (text.len < d + w.slack) return;
        w.stopped = true;
        return rt.fail("render stopped {d} bytes past its first difference", .{w.slack});
    }
};
