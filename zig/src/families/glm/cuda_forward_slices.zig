//! Forward.run's parts: its rows in (`enter`), the pass it takes (`cuts`), and (TF_GLM_OVN_FILL_SLICE_MS) a prompt
//! chunk's pieced layers lo..hi (`layers`), so a chunk can run in layer slices with decode rounds between them.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const prof_mod = @import("cuda_prof.zig");
const shared = @import("cuda_shared.zig");
const pieces_mod = @import("cuda_forward_pieces.zig");
const Forward = fw.Forward;

/// Forward.run's start: every rank's rows readied (its sequence, or a shared window's segments), its ids in, the rows
/// embedded and an image's rows spliced over their placeholders.
pub fn enter(f: *const Forward, ids: []const i32) !void {
    const R = ids.len;
    for (f.ranks) |r| {
        if (R == 0 or R > r.b.rows) return error.BadWindow;
        if (r.segs.len == 0) {
            try r.st.ready(f.stream, R);
            r.st.pending = R;
            r.st.prompt = r.b.prefill;
        } else try shared.begin(r, R, f.stream);
        try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
    }
    const c = &f.ranks[0].w.config;
    const embed_stamp = prof_mod.span(f.prof, f.stream, .embed);
    for (f.ranks) |r| try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
    if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
    embed_stamp.end();
    if (f.prof) |p| p.mark(f.stream, .embed);
}

/// The row pieces of a pass of `R` rows when it takes the pieced pass (their count), else null (Forward.whole).
pub fn cuts(f: *const Forward, R: usize, cut: *[fw.max_pieces][2]usize) ?usize {
    const pieces = if (f.pieces) |p| (if (f.ranks[0].b.prefill) fw.ranges(R, p.count, cut) else 0) else 0;
    // overlap.Overlap.begin: pieced with two pieces or more, and split takes a chunk of one piece too
    return if (pieces > 1 or (pieces == 1 and f.splits(R))) pieces else null;
}

/// Whether a prompt chunk of `R` rows can run in slices: one rank's buffers, no shared window, the pieced pass.
pub fn sliceable(f: *const Forward, R: usize) bool {
    var cut: [fw.max_pieces][2]usize = undefined;
    return f.ranks.len == 1 and f.ranks[0].segs.len == 0 and cuts(f, R, &cut) != null;
}

/// Layers lo..hi of the pieced pass of `R` rows (`enter` ran): the calls one pass over every layer makes, in order.
pub fn layers(f: *const Forward, R: usize, lo: usize, hi: usize) !void {
    var cut: [fw.max_pieces][2]usize = undefined;
    const n = cuts(f, R, &cut) orelse return error.NotPieced;
    try pieces_mod.piecedLayers(f, f.pieces.?, R, cut[0..n], lo, hi);
}
