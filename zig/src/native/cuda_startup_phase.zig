//! Native startup phase wiring and host construction; the observer never owns calibration or transfers.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const kv_pages = @import("cuda_kv_pool.zig");
const family_memory = @import("cuda_family_memory.zig");
const getenv = @import("cuda_device.zig").getenv;
pub const Hook = ?*const fn (*anyopaque, cuda.startup_watch.Phase) void;

/// A 0 or 1 setting; any other value refuses the start rather than falling back to the default in silence.
fn flag(name: [:0]const u8) !?bool {
    const v = getenv(name) orelse return null;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    std.log.err("{s}={s}: 0 or 1", .{ name, v });
    return error.BadCopySetting;
}

/// The copy policy from TENSORFOLD_COPY_*; every setting unset leaves the engine's own widening rule.
fn copyPolicy(h: anytype) !void {
    if (getenv("TENSORFOLD_COPY_MISS")) |v| h.cfg.copy_miss = std.meta.stringToEnum(lanes.Config.CopyMiss, v) orelse {
        std.log.err("TENSORFOLD_COPY_MISS={s}: first or keep", .{v});
        return error.BadCopySetting;
    };
    if (getenv("TENSORFOLD_COPY_RAMP")) |v| h.cfg.copy_ramp = std.meta.stringToEnum(lanes.Config.CopyRamp, v) orelse {
        std.log.err("TENSORFOLD_COPY_RAMP={s}: double or jump", .{v});
        return error.BadCopySetting;
    };
    if (try flag("TENSORFOLD_COPY_SHARED")) |v| h.cfg.copy_shared = v;
    if (try flag("TENSORFOLD_COPY_RESYNC")) |v| h.host.resync = v;
    if (try flag("TENSORFOLD_COPY_RESYNC_PROMPT_ONLY")) |v| h.host.resync_prompt_only = v;
    std.log.info("copy policy: miss {s}, ramp {s}, shared rounds {s}, resync {s}", .{ @tagName(h.cfg.copy_miss), @tagName(h.cfg.copy_ramp), if (h.cfg.copy_shared) "ramp prompt copies" else "first width", if (!h.host.resync) "off" else if (h.host.resync_prompt_only) "prompt copies" else "every copy" });
}

pub fn readiness(h: anytype, comptime F: type) void {
    if (@hasDecl(F, "startup_phase")) h.startup_phase = F.startup_phase;
    if (@hasDecl(F, "healthy")) h.health = .{ .ctx = h.family, .read = F.healthy };
}
fn note(h: anytype, phase: cuda.startup_watch.Phase) void {
    if (h.startup_phase) |callback| callback(h.family, phase);
}

pub fn serve(h: anytype, io: std.Io, facts: lanes.Model, rows: u32, info: api.Info, explain: ?api.Explain, callbacks: anytype) !void {
    note(h, .calibration);
    errdefer note(h, .failed);
    h.cfg = try lanes.Config.init(h.gpa, facts, rows, rows - 1);
    errdefer h.cfg.deinit(h.gpa);
    if (h.served.prepare != null) try @import("cuda_startup_costs.zig").open(h, callbacks.bound);
    // a drafter's trees: the planner picks a lone stream's lanes, as deep as it drafts
    if (facts.head_trees) h.cfg.head_lanes = rows - 1;
    if (facts.head_trees) h.cfg.head_depth = @max(1, facts.drafts);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(h.gpa, &h.cfg, callbacks.backend(h), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(h.gpa, io, &h.core, info);
    try copyPolicy(h);
    h.host.on_thread = .{ .ctx = h, .run = callbacks.initThread };
    h.host.memory = if (h.gpu != null) .{ .read = family_memory.read } else null;
    h.host.explain = explain;
    h.host.health = @import("cuda_readiness.zig").source(h);
    h.host.cache = h.store;
    h.host.warm_quiet = if (info.warm_turns) nemotron.native.warm_quiet else 0;
    if (h.lone != null) h.host.lone = .{ .ctx = h, .run = callbacks.loneRun, .sampled = true, .logprobs = h.inner.vtable.first_row != null, .structures = h.inner.vtable.first_masked != null, .min_budget = nemotron.native.lone_min_budget, .resumes = true };
    if (h.learn != null) h.host.learner = callbacks.learner(h);
    kv_pages.install(&h.kv, &h.host); // the pool's first census, before the engine thread owns the pool
    errdefer kv_pages.unwatch(&h.kv); // a host that never starts leaves the pool reporting to no one
    if (h.kv_count) |p| h.host.countPool(p);
    errdefer h.host.uncountPool();
    try h.host.start();
    note(h, .ready);
}
