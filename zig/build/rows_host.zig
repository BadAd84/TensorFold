//! Disk spill lifecycle runs without shaders, checkpoints or devices.
const std = @import("std");
pub fn tests(b: *std.Build, api: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const filter = b.option([]const u8, "rows-filter", "One spill lifecycle host regression") orelse "rows_";
    const run = b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{filter} }));
    // the file format and the reader pool live in lanes, where every rank's part of a state (DISK_LOCAL) reaches them
    const files = b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{ filter, "disk_readers" } }));
    const step = b.step("test-rows-host", "Spill lifecycle and file exactness without a GPU");
    step.dependOn(&run.step);
    step.dependOn(&files.step);
    all.dependOn(&run.step);
    all.dependOn(&files.step);
}
