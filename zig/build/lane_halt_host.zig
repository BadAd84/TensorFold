//! CPU checks for halt ordering, with the real lane host and fake target.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const interrupted = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_interrupted_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    b.step("test-lane-interrupted-host", "Interrupted prompt retention and clean retries without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = interrupted, .filters = &.{"interrupted prefill:"} })).step);
    const halt = b.addTest(.{ .root_module = api, .filters = &.{ "halt (", "halt fails" } });
    b.step("test-lane-halt-host", "Halt ordering with a fake backend and no GPU dependencies").dependOn(&b.addRunArtifact(halt).step);
    b.step("test-lane-lone-host", "Existing lone-driver handoff and reclaim with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"lone driver"} })).step);
    b.step("test-lane-api-host", "Shared lane host and engine API CPU regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    b.step("test-snapshot-index-host", "Prompt-cache lookup index against the whole-store scan, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"snapshot index:"} })).step);
    b.step("test-kept-walk-host", "TF_GLM_OVN_KEPT_WALK: every kept-state walk decides the same, comparing fewer lists").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"kept walk:"} })).step);
    const spark = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu });
    inline for (.{ .{ b.graph.host, "kept-walk-bench", "compile-kept-walk-bench-host" }, .{ spark, "kept-walk-bench-spark", "compile-kept-walk-bench-spark" } }) |v| {
        const fast_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = v[0], .optimize = .fast, .link_libc = true });
        const walk_bench = b.addExecutable(.{ .name = v[1], .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/prompt_prefix_bench.zig"), .target = v[0], .optimize = .fast, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = fast_lanes }} }) });
        b.step(v[2], "Build the CPU-only kept-state walk timing probe (TF_GLM_OVN_KEPT_WALK) without running it").dependOn(&b.addInstallArtifact(walk_bench, .{}).step);
    }
    b.step("test-rows-prefetch-host", "Disk states read ahead for queued requests: equality, cancellation, fallbacks").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefetch:"} })).step);
    b.step("test-embedding-lane-host", "Prompt-only embedding jobs on the actual lane worker without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"embedding lane:"} })).step);
    b.step("test-prefill-service-host", "The opt-in prompt-row budget beside decoding streams on the lane host with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefill service:"} })).step);
    const copies = b.step("test-root-copy-host", "Root copy precedence, drafter absorption and real worker checks without GPU");
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"root copy:"} })).step);
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"root copy:"} })).step);
}
