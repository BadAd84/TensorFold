//! TF_GLM_OVN_TOPK_SUM and _COMBINE_SY against the kernels they replace, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const glue = glm.glue;

/// fp32 inputs in [-scale, scale).
fn f32s(r: *Rig, n: usize, scale: f32) ![]f32 {
    const out = try r.a.alloc(f32, n);
    for (out) |*v| v.* = (r.rng.float(f32) * 2 - 1) * scale;
    return out;
}

/// selectSum on the router's K slices: the picks and weights of sliceSum + select, at decode rows and a prompt's.
fn topkSum(r: *Rig, set: glue.Set) !void {
    const ne = 256;
    const top_k = 8;
    const bias = try r.dev(std.mem.sliceAsBytes(try f32s(r, ne, 0.05)));
    var name: [128]u8 = undefined;
    for ([_]usize{ 1, 2, 3, 16, 300 }) |m| for ([_]bool{ true, false }) |norm| {
        const part = try r.dev(std.mem.sliceAsBytes(try f32s(r, glue.router_ks * m * ne, 0.5)));
        const logits = try r.zeros(m * ne * 4);
        const bytes = m * (top_k + 1) * 4;
        const pick = [2]u64{ try r.zeros(bytes), try r.zeros(bytes) };
        const wts = [2]u64{ try r.zeros(bytes), try r.zeros(bytes) };
        try glue.sliceSum(set, r.s, part, logits, m * ne, glue.router_ks, true); // glue.router's sum of the same slices
        try glue.select(set, r.s, logits, bias, pick[0], wts[0], m, top_k, ne, 2.5, norm);
        try glue.selectSum(set, r.s, part, bias, pick[1], wts[1], m, top_k, ne, 2.5, norm);
        const how = if (norm) "normed" else "not normed";
        try check.sameBytes(try std.fmt.bufPrint(&name, "topk_sum {d} rows ({s}): picks", .{ m, how }), try r.back(pick[1], bytes), try r.back(pick[0], bytes));
        try check.sameBytes(try std.fmt.bufPrint(&name, "topk_sum {d} rows ({s}): weights", .{ m, how }), try r.back(wts[1], bytes), try r.back(wts[0], bytes));
    };
    check.pass("topk_sum: picks and weights equal slice_sum + topk's at 1, 2, 3, 16 and 300 rows", .{});
}

/// combineSy against put_slot + combine: fp32 slot rows (decode, EXL3 prompts) and bf16 ones (NVFP4 prompts).
fn combineSy(r: *Rig, set: glue.Set, put_slot: cuda.Function) !void {
    const d = 4096;
    const slots = 9;
    var name: [128]u8 = undefined;
    for ([_]usize{ 1, 2, 3, 40 }) |rows| for ([_]bool{ true, false }) |f32_slots| {
        const ey = if (f32_slots) try r.dev(std.mem.sliceAsBytes(try f32s(r, rows * slots * d, 1.0))) else try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * slots * d, 1.0)));
        const sy = try r.dev(std.mem.sliceAsBytes(try f32s(r, rows * d, 1.0)));
        const w = try f32s(r, rows * slots, 1.0);
        for (0..rows) |i| w[i * slots + slots - 1] = 1; // the shared expert's weight
        const wts = try r.dev(std.mem.sliceAsBytes(w));
        const out = [2]u64{ try r.zeros(rows * d * 4), try r.zeros(rows * d * 4) };
        // first: ey's last slot still holds noise, which combine_sy must not read
        try glue.combineSy(set, r.s, ey, f32_slots, sy, wts, out[1], rows, d, slots);
        var a: cuda.Args = .{};
        a.add(sy);
        a.add(ey);
        for ([_]usize{ rows, d, slots, slots - 1, @intFromBool(!f32_slots) }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(put_slot, .{ .grid = .{ .x = @intCast((rows * d + 255) / 256) }, .block = .{ .x = 256 } }, r.s, &a);
        try glue.combine(set, r.s, ey, f32_slots, wts, out[0], rows, d, slots);
        const kind = if (f32_slots) "fp32" else "bf16";
        try check.sameBytes(try std.fmt.bufPrint(&name, "combine_sy {d} rows, {s} slots", .{ rows, kind }), try r.back(out[1], rows * d * 4), try r.back(out[0], rows * d * 4));
    };
    check.pass("combine_sy: sums equal put_slot + combine's at 1, 2, 3 and 40 rows, fp32 and bf16 slots", .{});
}

/// The knobs' kernels against the default path's, on the glue set and glm_ops' put_slot.
pub fn run(r: *Rig, set: glue.Set, ops: cuda.Module) !void {
    try topkSum(r, set);
    try combineSy(r, set, try ops.function("tf_glm_put_slot"));
}
