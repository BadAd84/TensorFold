//! TF_GLM_OVN_MLA_FUSEMERGE, _ABSORB_VEC, _KVB8, _LATENT_MMA and _SEG_PASS launch choices under a recording driver.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const t = std.testing;

const Rec = struct {
    var n: usize = 0;
    var ids: [8]usize = undefined;
    var gy: [8]c_uint = undefined;
    var gz: [8]c_uint = undefined;
    var bx: [8]c_uint = undefined;
    var smem: [8]c_uint = undefined;
    fn run(f: cuda.abi.Function, _: c_uint, y: c_uint, z: c_uint, x: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        ids[n] = @intFromPtr(f);
        gy[n] = y;
        gz[n] = z;
        bx[n] = x;
        smem[n] = shared;
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

test "TF_GLM_OVN_LATENT_MMA_EXACT: the knob reads, and a prompt buffer's absorb and expand run the exact kernels" {
    inline for (.{ "MLA_FUSEMERGE", "ABSORB_VEC", "LATENT_MMA", "SEG_PASS", "KVB8", "DECODE_SHARD", "PK8" }) |k| cuda.ovn.fake(k, null);
    cuda.ovn.fake("LATENT_MMA_EXACT", "1");
    try t.expect((try glue.Mla.read()).mma_exact);
    try t.expect(!(try glue.Mla.read()).mma); // the exact knob does not turn on the tensor-core form
    try t.expect(glue.Mla.wanted());
    cuda.ovn.fake("LATENT_MMA_EXACT", null);
    try t.expect(!(try glue.Mla.read()).mma_exact);
    try t.expect(!glue.Mla.wanted());

    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const prompt: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .mma_exact = true } };
    const decode: glue.Set = .{ .glue = &ks, .mla = .{ .mma_exact = true } };
    Rec.n = 0;
    try latent.absorbQ(prompt, s, 256, 4096, 0, 512, 4096, 22, 256, 512);
    try latent.expandV(prompt, s, 256, 4096, 0, 512, 3, 22, 256, 512); // a resumed tail of 3 rows: the same kernel
    try latent.absorbQ(decode, s, 256, 4096, 0, 512, 2, 22, 256, 512); // decode keeps today's kernels
    try latent.expandV(decode, s, 256, 4096, 0, 512, 2, 22, 256, 512);
    try t.expectEqualSlices(usize, &.{ id("absorb_exact"), id("expand_exact"), id("absorb_q"), id("expand_v") }, Rec.ids[0..Rec.n]);
    try t.expectEqualSlices(c_uint, &.{ 64, 1 }, Rec.gz[0..2]); // 64-row blocks: 4,096, then a 3-row tail
    const both: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .mma = true, .mma_exact = true } };
    Rec.n = 0;
    try latent.absorbQ(both, s, 256, 4096, 0, 512, 64, 22, 256, 512); // the exact form wins over the tensor-core form
    try t.expectEqual(id("absorb_exact"), Rec.ids[0]);
    Rec.n = 0;
    try latent.absorbQ(prompt, s, 256, 4096, 8192, 512, 2, 22, 256, 512); // KVB8 (sk != 0): the bf16-only exact kernel steps aside
    try t.expectEqual(id("absorb_q8"), Rec.ids[0]);
    try latent.absorbQ(prompt, s, 256, 4104, 0, 512, 64, 22, 256, 512); // wk off 16 bytes: base absorb_q keeps the same bytes
    try t.expectEqual(id("absorb_q"), Rec.ids[Rec.n - 1]);
}

test "TF_GLM_OVN_SEG_PASS: decode chunk launches on the seg pass (24-head blocks for 17-24 heads), prompts as today" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const on: glue.Set = .{ .glue = &ks, .mla = .{ .seg = true } };
    try t.expectEqualSlices(usize, &.{ id("chunks_seg24"), id("merge"), id("chunks_seg24"), id("merge") }, try attention(on, s, 0, 2, 16));
    try t.expectEqualSlices(c_uint, &.{ 1, 22, 1, 22 }, Rec.gy[0..4]); // one block holds every head; merge as today
    try t.expectEqualSlices(c_uint, &.{ 384, 128, 384, 128 }, Rec.bx[0..4]);
    try t.expectEqualSlices(c_uint, &.{ 3, 5 }, &.{ Rec.gz[0], Rec.gz[2] }); // nch chunks, sparse.width's five
    try t.expectEqual(glue.seg_smem[1], Rec.smem[0]);
    try t.expectEqualSlices(usize, &.{ id("chunks_seg24_fp8"), id("merge"), id("chunks_seg24_fp8"), id("merge") }, try attention(on, s, 64, 70, 32));
    for ([_][3]usize{ .{ 32, 2, 0 }, .{ 16, 1, 0 }, .{ 64, 4, 64 } }) |x| { // 16-head blocks outside 17-24 heads
        Rec.n = 0;
        try latent.attentionFormat(on, s, 256, 512, x[2], .{}, 768, p, 4096, 64, 2, x[0], 512, 3, 0.0625, 16);
        try t.expectEqual(if (x[2] == 0) id("chunks_seg16") else id("chunks_seg16_fp8"), Rec.ids[0]);
        try t.expectEqual(@as(c_uint, @intCast(x[1])), Rec.gy[0]);
        try t.expectEqual(glue.seg_threads[0], Rec.bx[0]);
        try t.expectEqual(glue.seg_smem[0], Rec.smem[0]);
    }
    Rec.n = 0; // TP=2's 32 heads on today's one 32-head block: two seg blocks would double the CTAs, so today's kernel
    try latent.attentionFormat(on, s, 256, 512, 0, .{}, 768, p, 4096, 64, 8, 32, 512, 3, 0.0625, 32);
    try t.expectEqual(id("chunks_32"), Rec.ids[0]);
    try t.expectEqual(@as(c_uint, 1), Rec.gy[0]);
    const prompt: glue.Set = .{ .glue = &ks, .prompt = true, .mla = .{ .seg = true } }; // prompt buffers: today's kernels
    try t.expectEqualSlices(usize, &.{ id("chunks_32"), id("merge"), id("chunks_32"), id("merge") }, try attention(prompt, s, 0, 70, 32));
    try t.expectEqualSlices(c_uint, &.{ 256, 128, 256, 128 }, Rec.bx[0..4]);
    const fused: glue.Set = .{ .glue = &ks, .mla = .{ .fuse_rows = 64, .skip = true, .seg = true } }; // fused launches first
    try t.expectEqualSlices(usize, &.{ id("chunks_32_fused"), id("chunks_32_fused") }, try attention(fused, s, 0, 70, 32));
    try t.expectEqualSlices(usize, &.{ id("chunks_seg24"), id("merge"), id("chunks_seg24"), id("merge") }, try attention(fused, s, 0, 63, 32));
}

const Ev = struct {
    var live: usize = 0;
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        live += 1;
        out.* = @ptrFromInt(live);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
};

test "TF_GLM_PROF=1: a decode's mla stage split at its chunk pass and merge; fused launches one part, prompts none" {
    const prof = @import("cuda_prof.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    d.api.cuEventCreate = Ev.create;
    d.api.cuEventDestroy_v2 = Ev.destroy;
    d.api.cuEventRecord = Ev.record;
    const ks = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const pr = try prof.Prof.init(&d, t.io);
    defer pr.deinit();
    pr.open = true;
    for ([_]glue.Set{ .{ .glue = &ks }, .{ .glue = &ks, .mla = .{ .seg = true } } }) |decode| {
        pr.n = 0;
        Rec.n = 0;
        try latent.attentionProfile(decode, pr, s, 256, 512, 0, .{}, 768, p, 4096, 64, 2, 22, 512, 3, 0.0625, 16);
        try latent.sparseAttentionProfile(decode, pr, s, 256, 512, 0, .{}, 1024, 2048, 2051, p, 4096, 2, 22, 512, 0.0625);
        try t.expectEqualSlices(prof.Section, &.{ .mla, .mla_chunks, .mla_merge, .mla, .mla_chunks, .mla_merge }, pr.ends[0..pr.n]);
    }
    pr.n = 0;
    Rec.n = 0;
    const fused: glue.Set = .{ .glue = &ks, .mla = .{ .fuse_rows = 64, .skip = true } };
    try latent.attentionProfile(fused, pr, s, 256, 512, 0, .{}, 768, p, 4096, 64, 70, 22, 512, 3, 0.0625, 32);
    try t.expectEqualSlices(prof.Section, &.{ .mla, .mla_chunks }, pr.ends[0..pr.n]);
    pr.n = 0;
    Rec.n = 0;
    const prompt: glue.Set = .{ .glue = &ks, .prompt = true };
    try latent.attentionProfile(prompt, pr, s, 256, 512, 0, .{}, 768, p, 4096, 64, 70, 22, 512, 3, 0.0625, 32);
    try t.expectEqual(@as(usize, 0), pr.n);
    pr.open = false;
}
