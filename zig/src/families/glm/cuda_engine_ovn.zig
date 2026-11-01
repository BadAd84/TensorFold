//! The engine's speed-hunt knobs made at load (TF_GLM_OVN_SHARED_PROF, _DRAW1, _SEG_KDA, _SEG_COMMIT, _SEG_DSA,
//! _FILL_SLICE_MS, _ONE_CHUNK, _LINEAGE).
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;

/// After the window buffers, the sampler and the profilers: every knob that is on gets its scratch.
pub fn open(e: *Engine, gpa: std.mem.Allocator, io: std.Io) !void {
    errdefer close(e);
    const slice_ms = try cuda.ovn.int("FILL_SLICE_MS", 0); // with FILL_FAST_MS: cuda_prompt_slices
    if (slice_ms > 0) {
        e.slices = try @import("cuda_prompt_slices.zig").Slices.create(gpa, e.d, slice_ms, try cuda.ovn.int("FILL_FAST_MS", 0), e.w.layers.len);
        @import("cuda_prompt_slices.zig").on = true;
    }
    if (cuda.ovn.on("ONE_CHUNK")) { // the bound sequence's shapes size its captures
        e.one = try @import("cuda_one_chunk.zig").Capture.create(gpa, e.d, &e.cur.st, e.c.i_dim, e.c.hidden);
        @import("cuda_one_chunk.zig").on = true;
    }
    if (cuda.ovn.on("LINEAGE")) @import("cuda_lineage.zig").on = true; // kept and live rows' tags (cuda_lineage)
    if (cuda.ovn.on("SHARED_PROF")) e.shared_prof = try @import("cuda_engine_prof.zig").Shared.create(gpa, e.d, io);
    // DRAW1 and SEG_DSA are sized by the shared rows: made after the page pool (cuda_engine_shared.ensure)
    if (e.mtp and cuda.ovn.on("SEG_KDA")) e.seg_kda = try @import("cuda_kda_seg.zig").SegKda.init(e.d, e.m.mods[0]);
    if (cuda.ovn.on("KDA_PIECES")) { // a multi-prompt chunk's KDA chains in one launch (cuda_kda_pieces); e.bp's table
        e.pieces_kda = try @import("cuda_kda_pieces.zig").PiecesKda.init(e.d, e.m.mods[0]);
        e.bp.seg.pieces = &e.pieces_kda.?;
    }
    if (e.mtp and cuda.ovn.on("SEG_COMMIT")) e.seg_commit = try @import("cuda_commit_seg.zig").SegCommit.init(e.d, e.m.mods[0], e.m.mods[6]);
}

/// Frees what `open` made (before the buffers and the sampler go).
pub fn close(e: *Engine) void {
    if (@import("cuda_one_chunk.zig").on) if (e.one) |c| {
        c.destroy();
        e.one = null;
    };
    if (@import("cuda_prompt_slices.zig").of(e)) |s| {
        s.destroy(e.gpa);
        e.slices = null;
    }
    if (e.seg_dsa) |*x| {
        x.deinit();
        e.seg_dsa = null;
        if (e.bs) |*b| b.seg.dsa = null;
    }
    if (e.seg_commit) |*x| {
        x.deinit();
        e.seg_commit = null;
    }
    if (e.seg_kda) |*x| {
        x.deinit();
        e.seg_kda = null;
        if (e.bs) |*b| b.seg.kda = null;
    }
    if (e.pieces_kda) |*x| {
        x.deinit();
        e.pieces_kda = null;
        e.bp.seg.pieces = null;
    }
    if (e.draw1) |*x| {
        x.deinit();
        e.draw1 = null;
    }
    if (e.shared_prof) |s| {
        s.destroy();
        e.shared_prof = null;
    }
}
