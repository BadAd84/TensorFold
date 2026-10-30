//! TF_GLM_OVN_HC_ROWS (16, 32 prompt rows a block) and _HC_DEC (hc.cu's decode layout): today's bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const glue = glm.glue;

const d = 4096;
const s = 4;
const wide = s * d;
const mixes = 24;
const most = 1366;
const counts = [_]usize{ 1, 2, 4, 5, 8, 16, 32, 63, 64, 65, 70, 130, 683, most };

const Outs = struct { out: u64, xs: u64, post: u64, comb: u64, part: u64 };
const per = [5]usize{ d * 2, (d / 64) * 4, s * 4, s * s * 4, glue.hcPartBytes(1) };
const what = [5][]const u8{ "normed rows", "group sums", "post", "comb", "partials" };

fn pre(r: *Rig, set: glue.Set, x: u64, w: [3]u64, norm: glue.Norm, rows: usize) !Outs {
    const o: Outs = .{ .out = try r.zeros(rows * d * 2), .xs = try r.zeros(rows * (d / 64) * 4), .post = try r.zeros(rows * s * 4), .comb = try r.zeros(rows * s * s * 4), .part = try r.zeros(glue.hcPartBytes(rows)) };
    try glue.hcPre(set, r.s, x, w[0], w[1], w[2], norm, o.out, o.xs, o.post, o.comb, o.part, 1e-6, 1e-6, 20, rows, d, s);
    return o;
}

/// `rows` rows of `set`'s call from x row `at` equal the reference's rows at .. at + rows (all five outputs).
fn equal(r: *Rig, label: []const u8, set: glue.Set, want: Outs, x: u64, w: [3]u64, norm: glue.Norm, at: usize, rows: usize) !void {
    const o = try pre(r, set, x + at * wide * 2, w, norm, rows);
    var name: [160]u8 = undefined;
    for ([5]u64{ o.out, o.xs, o.post, o.comb, o.part }, [5]u64{ want.out, want.xs, want.post, want.comb, want.part }, per, what) |got, all, n, field|
        try ref.sameRows(r, try std.fmt.bufPrint(&name, "{s}: {d} rows from row {d}, {s} equal the {d}-row call's", .{ label, rows, at, field, most }), all + at * n, got, rows * n);
}

/// Each form against today's hc_partial on 1,366 rows: every count from row 0, three from row 7; then hcRow's forms.
pub fn run(r: *Rig, set: glue.Set, mod: cuda.Module) !void {
    errdefer cuda.completion.drain(r.gpu.d, r.s, "GLM HC_ROWS / HC_DEC fixture left work pending");
    var off = set.glue.*;
    try glue.hcLoad(&off, mod, 0, false);
    var k16 = set.glue.*;
    try glue.hcLoad(&k16, mod, 16, true);
    var k32 = set.glue.*;
    try glue.hcLoad(&k32, mod, 32, false);
    const base = try r.a.alloc(f32, mixes);
    for (base) |*b| b.* = (r.rng.float(f32) * 2 - 1) * 0.5;
    const scale = [3]f32{ 0.7, 0.9, 1.1 };
    const w = [3]u64{ try r.dev(std.mem.sliceAsBytes(try r.bfs(mixes * wide, 0.02))), try r.dev(std.mem.sliceAsBytes(base)), try r.dev(std.mem.asBytes(&scale)) };
    const norm: glue.Norm = .{ .bf16 = try r.dev(std.mem.sliceAsBytes(try r.bfs(d, 1.0))) };
    const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(most * wide, 1.0)));
    const want = try pre(r, .{ .glue = &off }, x, w, norm, most); // today's tf_glm_hc_partial, 4 rows a block
    const forms = [_]struct { []const u8, glue.Set, usize }{
        .{ "hc_partial8 / hc_partial (knobs off)", .{ .glue = &off }, most },
        .{ "HC_ROWS=16, tf_glm_hc_partial16", .{ .glue = &k16, .prompt = true }, most },
        .{ "HC_ROWS=32, tf_glm_hc_partial32", .{ .glue = &k32, .prompt = true }, most },
        .{ "HC_DEC, tf_glm_hc_partial48", .{ .glue = &k16 }, 63 }, // hcPre's rows under 64 only
        .{ "HC_DEC with PDL_EARLY, tf_glm_hc_partial48_pdl", .{ .glue = &k16, .pdl_early = true }, 63 },
    };
    for (forms) |f| {
        for (counts) |m| if (m <= f[2]) try equal(r, f[0], f[1], want, x, w, norm, 0, m);
        for ([_]usize{ 5, 63, 70 }) |m| if (m <= f[2]) try equal(r, f[0], f[1], want, x, w, norm, 7, m);
    }
    check.pass("TF_GLM_OVN_HC_ROWS and _HC_DEC: partials and outputs equal tf_glm_hc_partial's at 1 to 1,366 rows", .{});
    try @import("glm_ref_ovn.zig").rowForms(r, .{ .glue = &off }, &.{ .{ .glue = &k16, .prompt = true }, .{ .glue = &k32, .prompt = true } }, &counts);
}
