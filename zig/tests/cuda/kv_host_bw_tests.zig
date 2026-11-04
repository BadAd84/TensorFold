//! M1 of docs/research/kv-host-port-plan-2026-11-04.md: a whole-buffer checksum read from device
//! memory against the same bytes in mapped pinned host memory, the two arenas TF_GLM_KV_HOST would
//! place a pool's latent rows in.
//!
//! Kernel choice: no compiled kernel in our tree is a fair proxy. glm_ops.cu holds only
//! tf_glm_dsa_tokens(_i32), tf_glm_pools_i32, tf_glm_add_i32, tf_glm_put_slot and tf_glm_copy_rows
//! (zig/kernels/cuda/glm_ops.cu:32-70): none reads a whole buffer and reduces, pools_i32 and
//! copy_rows write half or all of what they read, and the l2pf/l2pace kernels prefetch without
//! consuming a value. So this check JITs the 30-line PTX checksum below through
//! cuda.Module.loadPtx (zig/src/cuda/module.zig:25), the runtime_tests.zig "ptx" precedent: no
//! fatbin rebuild and no kernel contract for a bench. The kernel reads 16-byte vectors strided
//! across threads, the load shape of upstream's tf_glm_kvh_copy (the KV_HOST gather's host-page
//! read, glm_ops.cu:77 in commit 4bddc9459), xors them into four lane accumulators and reduces
//! once per thread, so reads dominate the traffic and a wrong read changes the checksum.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const gib = 1 << 30;
const n_vec: u64 = gib / 16; // whole 16-byte vectors, no tail to handle
const blocks: u32 = 1024;
const threads: u32 = 256;
const warmups = 3;
const reps = 10;

const sum_ptx =
    \\.version 8.0
    \\.target sm_80
    \\.address_size 64
    \\
    \\.visible .entry tf_kvh_bw_sum(
    \\    .param .u64 p_src,
    \\    .param .u64 p_n,
    \\    .param .u64 p_out
    \\)
    \\{
    \\    .reg .pred %p1;
    \\    .reg .b32 %r<6>;
    \\    .reg .b32 %a<4>;
    \\    .reg .b64 %d<10>;
    \\    ld.param.u64 %d0, [p_src];
    \\    ld.param.u64 %d1, [p_n];
    \\    ld.param.u64 %d2, [p_out];
    \\    mov.u32 %r0, %ctaid.x;
    \\    mov.u32 %r1, %ntid.x;
    \\    mov.u32 %r2, %tid.x;
    \\    mad.lo.u32 %r3, %r0, %r1, %r2;
    \\    mov.u32 %r4, %nctaid.x;
    \\    mul.lo.u32 %r4, %r4, %r1;
    \\    cvt.u64.u32 %d3, %r3;
    \\    cvt.u64.u32 %d4, %r4;
    \\    cvta.to.global.u64 %d0, %d0;
    \\    cvta.to.global.u64 %d2, %d2;
    \\    mov.b32 %a0, 0;
    \\    mov.b32 %a1, 0;
    \\    mov.b32 %a2, 0;
    \\    mov.b32 %a3, 0;
    \\loop:
    \\    setp.ge.u64 %p1, %d3, %d1;
    \\    @%p1 bra done;
    \\    shl.b64 %d5, %d3, 4;
    \\    add.u64 %d5, %d0, %d5;
    \\    ld.global.v4.u32 {%r0, %r1, %r2, %r5}, [%d5];
    \\    xor.b32 %a0, %a0, %r0;
    \\    xor.b32 %a1, %a1, %r1;
    \\    xor.b32 %a2, %a2, %r2;
    \\    xor.b32 %a3, %a3, %r5;
    \\    add.u64 %d3, %d3, %d4;
    \\    bra loop;
    \\done:
    \\    mov.b64 %d6, {%a1, %a0};
    \\    mov.b64 %d7, {%a3, %a2};
    \\    xor.b64 %d6, %d6, %d7;
    \\    red.global.add.u64 [%d2], %d6;
    \\    ret;
    \\}
    \\
;

const Arm = struct { gbps: f64, sum: u64 };

/// Warm-up launches, then `reps` timed launches of the checksum over `src`, both events on the
/// stream so only the kernel is timed; the checksum the last launch left in `out` comes home.
fn arm(gpu: check.Gpu, f: cuda.Function, s: cuda.Stream, src: u64, out: *cuda.DeviceBuffer) !Arm {
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    const xs = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(xs);
    for (0..warmups + reps) |r| {
        try out.fill8(0, s.handle); // the reset rides the stream, before the timed window
        if (r >= warmups) try start.record(s);
        var a: cuda.Args = .{};
        a.add(src);
        a.add(n_vec);
        a.add(out.ptr);
        try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, s, &a);
        if (r >= warmups) {
            try end.record(s);
            try s.synchronize();
            xs[r - warmups] = @as(f64, gib) / (@as(f64, try start.elapsedMs(end)) / 1e3) / 1e9;
        }
    }
    try s.synchronize();
    var sum: u64 = 0;
    try out.download(0, std.mem.asBytes(&sum));
    return .{ .gbps = check.median(xs), .sum = sum };
}

/// `kvh-bw`: 1 GiB read by the same kernel from device memory and from mapped pinned host memory.
pub fn run(gpu: check.Gpu) !void {
    const d = gpu.d;
    const gpa = gpu.gpa;
    var log: [4096]u8 = undefined;
    var m = try cuda.Module.loadPtx(d, sum_ptx, &log);
    defer m.unload();
    const f = try m.function("tf_kvh_bw_sum");
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    const pattern = try gpa.alloc(u8, gib);
    defer gpa.free(pattern);
    for (pattern, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7); // l2pace_tests' generator
    var dev = try cuda.DeviceBuffer.fromHost(d, pattern);
    defer dev.free();
    var mapped = try cuda.HostBuffer.allocMapped(d, gib); // cuda/memory.zig:253: pinned, device-mapped
    defer mapped.free();
    @memcpy(mapped.bytes, pattern); // the CPU fill faults every page in before any timing
    const mapped_dev = try mapped.device(); // the device VA TF_GLM_KV_HOST's kernels would read
    var out = try cuda.DeviceBuffer.alloc(d, 8);
    defer out.free();
    const device = try arm(gpu, f, s, dev.ptr, &out);
    const host = try arm(gpu, f, s, mapped_dev, &out);
    try check.expect(device.sum == host.sum, "kvh-bw: the checksums differ: device {x:0>16}, mapped {x:0>16}: a wrong read cannot pass as fast", .{ device.sum, host.sum });
    std.debug.print("RESULT kvh-bw: device {d:.1} GB/s, mapped {d:.1} GB/s, sums equal\n", .{ device.gbps, host.gbps });
    check.pass("kvh-bw: 1 GiB checksum read, {d} reps' median each: device {d:.1} GB/s, mapped pinned {d:.1} GB/s, checksum {x:0>16} on both", .{ reps, device.gbps, host.gbps, device.sum });
}
