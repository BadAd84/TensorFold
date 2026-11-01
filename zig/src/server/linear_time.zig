//! Timing for linear-time tests: two sizes run in turn on the thread's CPU clock, each keeping its fastest run.
const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

/// The thread's CPU clock where it counts to the microsecond, so other work on the machine adds nothing; else awake.
pub fn clock(io: std.Io) std.Io.Clock {
    const r = std.Io.Clock.cpu_thread.resolution(io) catch return .awake;
    return if (r.nanoseconds > 0 and r.nanoseconds <= std.time.ns_per_us) .cpu_thread else .awake;
}

/// On Windows the thread CPU clock is a coarse tick, so measure exact CPU cycles instead (immune to machine load).
pub const cycles = builtin.os.tag == .windows;

const win = if (cycles) struct {
    extern "kernel32" fn GetCurrentThread() callconv(.winapi) windows.HANDLE;
    extern "kernel32" fn QueryThreadCycleTime(thread: windows.HANDLE, out: *u64) callconv(.winapi) windows.BOOL;
    fn now() u64 {
        var out: u64 = 0;
        _ = QueryThreadCycleTime(GetCurrentThread(), &out);
        return out;
    }
} else struct {};

/// The fastest of `rounds` runs each of `run(ctx, 0)` and `run(ctx, 1)`, taken in turn so load slows both alike.
pub fn fastest(rounds: usize, ctx: anytype, comptime run: fn (@TypeOf(ctx), usize) anyerror!void) ![2]u64 {
    var ns: [2]u64 = @splat(std.math.maxInt(u64));
    if (cycles) {
        for (0..rounds) |_| for (&ns, 0..) |*best, i| {
            const start = win.now();
            try run(ctx, i);
            best.* = @min(best.*, win.now() - start);
        };
        return ns;
    }
    const io = std.testing.io;
    const c = clock(io);
    for (0..rounds) |_| for (&ns, 0..) |*best, i| {
        const start = c.now(io);
        try run(ctx, i);
        best.* = @min(best.*, @as(u64, @intCast(@max(0, start.durationTo(c.now(io)).nanoseconds))));
    };
    return ns;
}

test "the CPU clock leaves out a sleep that the awake clock counts" {
    const io = std.testing.io;
    if (clock(io) != .cpu_thread) return error.SkipZigTest;
    const cpu = std.Io.Clock.cpu_thread.now(io);
    const awake = std.Io.Clock.awake.now(io);
    try std.Io.sleep(io, .fromMilliseconds(50), .awake);
    try std.testing.expect(awake.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 50 * std.time.ns_per_ms);
    try std.testing.expect(cpu.durationTo(std.Io.Clock.cpu_thread.now(io)).nanoseconds < 25 * std.time.ns_per_ms);
}
