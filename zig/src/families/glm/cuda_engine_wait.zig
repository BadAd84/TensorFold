//! Engine.gate's heat wait over the ranks (moved out of cuda_engine.zig to keep it under 600 lines).
const std = @import("std");
const heat = @import("heat");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Cancel = engine.Cancel;

/// heat.Gate.waitUntil's hooks over the ranks: gathered max reading, rank 0's stop: all pause together.
pub const Wait = struct {
    e: *Engine,
    cancel: ?Cancel,
    failed: ?anyerror = null,

    pub fn read(ptr: *anyopaque) anyerror!f64 {
        const w: *Wait = @ptrCast(@alignCast(ptr));
        const local: f64 = blk: {
            var dir = std.Io.Dir.cwd().openDir(w.e.io, w.e.heat_gate.root, .{ .iterate = true }) catch break :blk -std.math.inf(f64);
            defer dir.close(w.e.io);
            break :blk heat.hottest(w.e.io, dir) catch -std.math.inf(f64);
        };
        return w.e.shared(local);
    }

    pub fn sleep(ptr: *anyopaque) void {
        const w: *Wait = @ptrCast(@alignCast(ptr));
        std.Io.sleep(w.e.io, .fromMilliseconds(2000), .awake) catch {};
    }

    pub fn stop(ptr: *anyopaque) bool {
        const w: *Wait = @ptrCast(@alignCast(ptr));
        const mine: f64 = if (w.cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0;
        const any = w.e.shared(mine) catch |err| {
            w.failed = err;
            return true;
        };
        return any > 0;
    }
};
