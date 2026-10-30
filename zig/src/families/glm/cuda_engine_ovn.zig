//! The engine's speed-hunt knobs made at load (TF_GLM_OVN_SHARED_PROF, _DRAW1, _SEG_KDA, _SEG_COMMIT); null when off.
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;

/// After the window buffers, the sampler and the profilers: every knob that is on gets its scratch.
pub fn open(e: *Engine, gpa: std.mem.Allocator, io: std.Io) !void {
    errdefer close(e);
    if (cuda.ovn.on("SHARED_PROF")) e.shared_prof = try @import("cuda_engine_prof.zig").Shared.create(gpa, e.d, io);
    if (cuda.ovn.on("DRAW1")) e.draw1 = try @import("cuda_draw1.zig").Draw1.init(e);
    if (e.bs != null and cuda.ovn.on("SEG_KDA")) { // glm_kda is fw.Modules.load's module 0, glm_glue its 6
        e.seg_kda = try @import("cuda_kda_seg.zig").SegKda.init(e.d, e.m.mods[0]);
        e.bs.?.seg_kda = &e.seg_kda.?;
    }
    if (e.bs != null and cuda.ovn.on("SEG_COMMIT")) e.seg_commit = try @import("cuda_commit_seg.zig").SegCommit.init(e.d, e.m.mods[0], e.m.mods[6]);
}

/// Frees what `open` made (before the buffers and the sampler go).
pub fn close(e: *Engine) void {
    if (e.seg_commit) |*x| {
        x.deinit();
        e.seg_commit = null;
    }
    if (e.seg_kda) |*x| {
        x.deinit();
        e.seg_kda = null;
        if (e.bs) |*b| b.seg_kda = null;
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
