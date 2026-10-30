//! The engine's speed-hunt knobs made at load (TF_GLM_OVN_SHARED_PROF, _DRAW1, _SEG_KDA, _SEG_COMMIT, _SEG_DSA,
//! _FILL_SLICE_MS).
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
    if (cuda.ovn.on("SHARED_PROF")) e.shared_prof = try @import("cuda_engine_prof.zig").Shared.create(gpa, e.d, io);
    if (cuda.ovn.on("DRAW1")) e.draw1 = try @import("cuda_draw1.zig").Draw1.init(e);
    if (e.bs != null and cuda.ovn.on("SEG_KDA")) { // glm_kda is fw.Modules.load's module 0, glm_glue its 6
        e.seg_kda = try @import("cuda_kda_seg.zig").SegKda.init(e.d, e.m.mods[0]);
        e.bs.?.seg.kda = &e.seg_kda.?;
    }
    if (e.bs != null and cuda.ovn.on("SEG_COMMIT")) e.seg_commit = try @import("cuda_commit_seg.zig").SegCommit.init(e.d, e.m.mods[0], e.m.mods[6]);
    if (e.bs != null and cuda.ovn.on("SEG_DSA")) { // glm_glue is module 6, glm_ops module 1
        const b = &e.bs.?;
        e.seg_dsa = try @import("cuda_dsa_seg.zig").SegDsa.init(gpa, e.d, e.m.mods[6], e.m.mods[1], &e.w, b.rows, b.nch, e.capacity);
        b.seg.dsa = &e.seg_dsa.?;
    }
}

/// Frees what `open` made (before the buffers and the sampler go).
pub fn close(e: *Engine) void {
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
    if (e.draw1) |*x| {
        x.deinit();
        e.draw1 = null;
    }
    if (e.shared_prof) |s| {
        s.destroy();
        e.shared_prof = null;
    }
}
