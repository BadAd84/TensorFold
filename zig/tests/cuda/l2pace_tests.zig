//! TF_GLM_OVN_L2PF_PACED's kernels (#1580: l2pf segments, l2pace paced) leave the bytes they prefetch unchanged.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const chunk: u32 = 32 << 10;

pub fn run(gpu: check.Gpu) !void {
    const n: usize = (3 << 20) + 4096;
    const host = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(host);
    for (host, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
    var buf = try cuda.DeviceBuffer.fromHost(gpu.d, host);
    defer buf.free();
    const rows = [_][2]i64{
        .{ @intCast(buf.ptr), 1 << 20 }, // a whole MiB in 32 pieces
        .{ @intCast(buf.ptr + (1 << 20) + 16), (2 << 20) - 32 }, // a 16-byte offset and a short last piece
        .{ @intCast(buf.ptr + (3 << 20)), 4096 }, // one short piece
    };
    var table = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&rows));
    defer table.free();
    var seg_mod = try cuda.Module.load(gpu.d, cuda.kernels.l2pf);
    defer seg_mod.unload();
    var pace_mod = try cuda.Module.load(gpu.d, cuda.kernels.l2pace);
    defer pace_mod.unload();
    const segments = try seg_mod.function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij");
    const paced = try pace_mod.function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var a: cuda.Args = .{};
    a.add(table.ptr);
    a.add(@as(c_int, rows.len));
    a.add(chunk);
    try cuda.launch.launch(segments, .{ .grid = .{ .x = 2 }, .block = .{ .x = 128 } }, s, &a);
    var p: cuda.Args = .{};
    p.add(table.ptr);
    p.add(@as(c_int, rows.len));
    p.add(chunk);
    p.add(@as(u64, (chunk + 75) / 150)); // 150 GB/s, as the knob's default
    p.add(@as(u64, 0));
    try cuda.launch.launch(paced, .{ .grid = .{ .x = 2 }, .block = .{ .x = 32 } }, s, &p);
    try s.synchronize();
    const back = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(back);
    try buf.download(0, back);
    try check.expect(std.mem.eql(u8, host, back), "l2pace: the prefetched bytes are unchanged", .{});
    check.pass("l2pace: segments and paced prefetch of 3 MiB leave every byte unchanged", .{});
}
