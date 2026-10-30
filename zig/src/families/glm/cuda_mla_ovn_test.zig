//! TF_GLM_OVN_MLA_FUSEMERGE, _ABSORB_VEC, _KVB8 and _LATENT_MMA launch choices under a recording driver; off is today's.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const t = std.testing;

const Rec = struct {
    var n: usize = 0;
    var ids: [8]usize = undefined;
    var gz: [8]c_uint = undefined;
    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, z: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        ids[n] = @intFromPtr(f);
        gz[n] = z;
        n += 1;
        return 0;
    }
};

/// A kernel's handle in `kernels`: its field's index plus one.
fn id(comptime name: []const u8) usize {
    return std.meta.fieldIndex(glue.Kernels, name).? + 1;
}

fn kernels(d: *cuda.Driver) glue.Kernels {
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 0..) |name, i| @field(ks, name) = .{ .d = d, .handle = @ptrFromInt(i + 1) };
    return ks;
}

const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };

/// The kernels one dense call (hb-head blocks) and one sparse call of `rows` rows launch, in order.
fn attention(set: glue.Set, s: cuda.Stream, scales: u64, rows: usize, hb: usize) ![]const usize {
    Rec.n = 0;
    try latent.attentionFormat(set, s, 256, 512, scales, .{}, 768, p, 4096, 64, rows, 22, 512, 3, 0.0625, hb);
    try latent.sparseAttentionFormat(set, s, 256, 512, scales, .{}, 1024, 2048, 2051, p, 4096, rows, 22, 512, 0.0625);
    return Rec.ids[0..Rec.n];
}

test "TF_GLM_OVN MLA knobs off: today's chunk, merge, absorb and expand kernels" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const off: glue.Set = .{ .glue = &ks };
    try t.expectEqualSlices(usize, &.{ id("chunks_32"), id("merge"), id("chunks_32"), id("merge") }, try attention(off, s, 0, 70, 32));
    try t.expectEqualSlices(usize, &.{ id("chunks_16_fp8"), id("merge"), id("chunks_16_fp8"), id("merge") }, try attention(off, s, 64, 2, 16));
    Rec.n = 0;
    try latent.absorbQ(off, s, 256, 4096, 0, 512, 2, 22, 256, 512);
    try latent.expandV(off, s, 256, 4096, 0, 512, 2, 22, 256, 512);
    try t.expectEqualSlices(usize, &.{ id("absorb_q"), id("expand_v") }, Rec.ids[0..Rec.n]);
}

test "TF_GLM_OVN_MLA_FUSEMERGE: 32-head launches from fuse_rows rows fused, the rest bounded" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const on: glue.Set = .{ .glue = &ks, .mla = .{ .fuse_rows = 64, .skip = true } };
    try t.expectEqualSlices(usize, &.{ id("chunks_32_fused"), id("chunks_32_fused") }, try attention(on, s, 0, 70, 32));
    try t.expectEqual(@as(c_uint, 1), Rec.gz[0]); // one block a row and head block walks the chunks
    try t.expectEqualSlices(usize, &.{ id("chunks_32_fp8_fused"), id("chunks_32_fp8_fused") }, try attention(on, s, 64, 64, 32));
    try t.expectEqualSlices(usize, &.{ id("chunks_32_fp8_skip"), id("merge"), id("chunks_32_fp8_skip"), id("merge") }, try attention(on, s, 64, 63, 32));
    try t.expectEqual(@as(c_uint, 3), Rec.gz[0]); // a chunk a block, as today
    try t.expectEqualSlices(usize, &.{ id("chunks_16_skip"), id("merge"), id("chunks_16_skip"), id("merge") }, try attention(on, s, 0, 2, 16));
    const every: glue.Set = .{ .glue = &ks, .mla = .{ .fuse_rows = 1, .skip = true } }; // never a 16-head block
    try t.expectEqualSlices(usize, &.{ id("chunks_16_skip"), id("merge"), id("chunks_16_skip"), id("merge") }, try attention(every, s, 0, 4, 16));
}

test "TF_GLM_OVN_ABSORB_VEC and _KVB8: 16-byte forms when aligned, E4M3 forms whenever kv_b has scales" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const vec: glue.Set = .{ .glue = &ks, .mla = .{ .vec = true } };
    Rec.n = 0;
    try latent.absorbQ(vec, s, 256, 4096, 0, 512, 70, 22, 256, 512);
    try latent.expandV(vec, s, 256, 4096, 0, 512, 70, 22, 256, 512);
    try latent.absorbQ(vec, s, 256, 4104, 0, 512, 70, 22, 256, 512); // kv_b off 16 bytes: today's kernel
    try latent.expandV(vec, s, 264, 4096, 0, 512, 70, 22, 256, 512); // o off 16 bytes
    try latent.absorbQ(vec, s, 256, 4096, 8192, 512, 2, 22, 256, 512);
    try latent.expandV(.{ .glue = &ks }, s, 256, 4096, 8192, 512, 2, 22, 256, 512);
    try t.expectEqualSlices(usize, &.{ id("absorb_q_vec"), id("expand_v_vec"), id("absorb_q"), id("expand_v"), id("absorb_q8"), id("expand_v8") }, Rec.ids[0..Rec.n]);
    try t.expectEqual(@as(c_uint, 2), Rec.gz[0]); // 70 rows: two 64-row blocks
    try t.expectError(error.LatentShape, latent.absorbQ(vec, s, 256, 4100, 8192, 512, 2, 22, 256, 512));
}

test "TF_GLM_OVN_LATENT_MMA: every row of a prompt buffer on the mma forms, decode buffers as before" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const prompt: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .mma = true, .vec = true } };
    const decode: glue.Set = .{ .glue = &ks, .mla = .{ .mma = true, .vec = true } };
    Rec.n = 0;
    try latent.absorbQ(prompt, s, 256, 4096, 0, 512, 4096, 22, 256, 512);
    try latent.expandV(prompt, s, 256, 4096, 0, 512, 3, 22, 256, 512); // a resumed tail of 3 rows: the same kernel
    try latent.absorbQ(prompt, s, 256, 4096, 8192, 512, 70, 22, 256, 512);
    try latent.expandV(prompt, s, 256, 4096, 8192, 512, 70, 22, 256, 512);
    try latent.absorbQ(decode, s, 256, 4096, 0, 512, 2, 22, 256, 512);
    try latent.expandV(decode, s, 256, 4096, 8192, 512, 2, 22, 256, 512);
    try t.expectEqualSlices(usize, &.{ id("absorb_mma"), id("expand_mma"), id("absorb_mma8"), id("expand_mma8"), id("absorb_q_vec"), id("expand_v8") }, Rec.ids[0..Rec.n]);
    try t.expectEqualSlices(c_uint, &.{ 64, 1, 2, 2 }, Rec.gz[0..4]); // 64-row blocks: 4,096, a 3-row tail, 70
    const off: glue.Set = .{ .glue = &ks, .prompt = true }; // knob off: a prompt buffer keeps today's kernels
    Rec.n = 0;
    try latent.absorbQ(off, s, 256, 4096, 0, 512, 4096, 22, 256, 512);
    try latent.expandV(off, s, 256, 4096, 0, 512, 4096, 22, 256, 512);
    try t.expectEqualSlices(usize, &.{ id("absorb_q"), id("expand_v") }, Rec.ids[0..Rec.n]);
    try t.expectError(error.LatentShape, latent.absorbQ(prompt, s, 264, 4096, 0, 512, 64, 22, 256, 512)); // q off 16 B
}
