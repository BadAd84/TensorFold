const std = @import("std");
const cuda = @import("cuda");
const mm = @import("mm");
const kernels: mm.Kernels = .{
    .by_rows = .{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } },
    .packed_rows = @splat(.{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } }),
    .prompt = .{ .id = 512 },
    .reduce = .{ .id = 1024 },
    .tile = .{},
    .tile_packed = @splat(.{}),
    .pdl = false,
};
test "verify and draft windows keep shape-selected split K at every width" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 31, 32 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expectEqual(@as(u32, 4), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
    };
}
test "prompt flag chooses one K slice even for one row, alone or many" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 3, 16, 64, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
    };
}

test "fixed K prompt experiment uses the same shape split across bucket, offset and tail boundaries" {
    var k = kernels;
    k.prompt_fixed_k = true;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096 + 7 * 2048 * 2, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 2 * ((rows + 63) / 64)), cuda.count);
        var i: usize = 0;
        while (i < cuda.count) : (i += 2) {
            try std.testing.expectEqual(@as(u32, 4), cuda.configs[i].grid.z);
            try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[i + 1]);
        }
    };
    try std.testing.expect(!kernels.prompt_fixed_k);
}

fn ovnKernels() mm.Kernels {
    var k = kernels;
    k.ovn.wide = @splat(.{ .id = 2000 });
    k.ovn.cluster = @splat(.{ .{ .id = 3016 }, .{ .id = 3032 }, .{ .id = 3064 } });
    k.ovn.cluster_from = 32 << 20;
    k.ovn.dectile = @splat(.{ .id = 4000 });
    k.ovn.dectile_below = 96;
    k.ovn.swiglu = .{ .id = 5000 };
    return k;
}

test "ovn: every knob unset keeps today's launches" {
    const k = kernels;
    try std.testing.expect(k.ovn.wide == null and k.ovn.cluster == null and k.ovn.dectile == null and k.ovn.swiglu == null);
    cuda.count = 0;
    try std.testing.expect(!try mm.swigluReduce(&k, .{}, 4096, 4096, .{ .w = 8192, .n = 1408, .k = 4096 }, 16384, 32768, 2, 1.0, false));
    try std.testing.expectEqual(@as(usize, 0), cuda.count);
}

test "ovn MM_WIDE: prompts past the 0083 rule take one launch of 128 x 128 bands, the rest today's blocks" {
    const k = ovnKernels();
    for ([_]u8{ 16, 4, 6, 8 }) |bits| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, 4096, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 2000), cuda.symbols[0]);
        try std.testing.expectEqual(@as(u32, 32 * 32), cuda.configs[0].grid.x);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
        cuda.count = 0; // 64 blocks of 128 x 128: under the rule
        try mm.matmul(&k, .{}, 4096, 4096, .{ .w = 8192, .s = 12288, .n = 160, .k = 4096, .bits = bits }, 16384, 32768, 4096, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, if (bits == 16) 512 else 128), cuda.symbols[0]);
    }
}

test "ovn MM_CLUSTER: decode windows of a weight past the threshold sum their slices in a cluster, no reduce" {
    const k = ovnKernels();
    for ([_]usize{ 1, 2, 16, 17, 33, 64 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, 4096, .{ .w = 8192, .n = 8192, .k = 4096 }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, if (rows <= 16) 3016 else if (rows <= 32) 3032 else 3064), cuda.symbols[0]);
        try std.testing.expectEqual(@as(u32, 2), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 2), cuda.configs[0].cluster.?.z);
        cuda.count = 0; // 8 Mi values: the buffer and glm_mm_reduce
        try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .n = 4096, .k = 2048 }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
        try std.testing.expect(cuda.configs[0].cluster == null);
    }
}

test "ovn MM_DECTILE: a 16-row window under the block count takes 32-column tiles with today's slices" {
    const k = ovnKernels();
    cuda.count = 0; // shared down 4096 x 704: 64 blocks, one slice
    try mm.matmul(&k, .{}, 4096, 704, .{ .w = 8192, .s = 12288, .n = 4096, .k = 704, .bits = 4 }, 16384, 32768, 2, true, false);
    try std.testing.expectEqual(@as(usize, 1), cuda.count);
    try std.testing.expectEqual(@as(u32, 4000), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u32, 128), cuda.configs[0].grid.x);
    try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
    cuda.count = 0; // 17 rows: the 32-row bucket, today's tile
    try mm.matmul(&k, .{}, 4096, 704, .{ .w = 8192, .s = 12288, .n = 4096, .k = 704, .bits = 4 }, 16384, 32768, 17, true, false);
    try std.testing.expectEqual(@as(u32, 32), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u32, 64), cuda.configs[0].grid.x);
}

test "ovn SWIGLU_REDUCE: decode gate/up takes the matmul and one fused kernel; prompts and clusters decline" {
    var k = ovnKernels();
    cuda.count = 0; // the shared expert's 1408 x 4096: 8 slices
    try std.testing.expect(try mm.swigluReduce(&k, .{}, 4096, 4096, .{ .w = 8192, .n = 1408, .k = 4096 }, 16384, 32768, 2, 1.0, false));
    try std.testing.expectEqual(@as(usize, 2), cuda.count);
    try std.testing.expectEqual(@as(u32, 8), cuda.configs[0].grid.z);
    try std.testing.expectEqual(@as(u32, 5000), cuda.symbols[1]);
    cuda.count = 0;
    try std.testing.expect(!try mm.swigluReduce(&k, .{}, 4096, 4096, .{ .w = 8192, .n = 1408, .k = 4096 }, 16384, 32768, 2, 1.0, true));
    try std.testing.expect(!try mm.swigluReduce(&k, .{}, 4096, 4096, .{ .w = 8192, .n = 8192, .k = 4096 }, 16384, 32768, 2, 1.0, false));
    try std.testing.expectEqual(@as(usize, 0), cuda.count);
    k.ovn.cluster = null; // the dense gate/up without clusters: 2 slices, fused
    try std.testing.expect(try mm.swigluReduce(&k, .{}, 4096, 4096, .{ .w = 8192, .n = 8192, .k = 4096 }, 16384, 32768, 2, 1.0, false));
    try std.testing.expectEqual(@as(usize, 2), cuda.count);
}
