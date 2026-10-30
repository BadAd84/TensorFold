//! glm-draw1 (GPU): TF_GLM_OVN_DRAW1's draws against the full gather's, token for token, on a three-rank split of
//! each row: greedy picks by one tf_glm_top1_rows launch against tf_glm_top1 a row, and sampled tokens from each rank's
//! top k (tf_draw_cands, tf_draw_place, tf_draw_rows) against tf_draw on the whole rows, random and tie-heavy rows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const glm = @import("glm_cuda");

const world = 3;
const rows = 8;
const Rule = cuda.sample.Rule;

const Kind = enum { spread, peaked, ties, flat, steps };

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

/// A row by a fixed generator in bf16; `odd` 1 puts -inf in every fifth column, 2 also NaN in every thirteenth.
fn fill(kind: Kind, salt: u64, odd: u2, out: []u16) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (salt *% 0xD1B5_4A32_D192_ED03);
    for (out, 0..) |*h, i| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const r: f32 = @as(f32, @floatFromInt(x >> 40)) / 16777216.0;
        const v: f32 = switch (kind) {
            .spread => -3.0 * r,
            .peaked => if (i % 2711 == 7) 2.0 + 6.0 * r else -12.0 * r,
            .ties => -3.0 * @floor(r * 16.0) / 16.0,
            .flat => 0.0,
            .steps => -@floor(r * 4.0), // four values: every rank's top k and the global one cut inside one tie
        };
        h.* = bf16(v);
        if (odd >= 1 and i % 5 == 1) h.* = 0xFF80;
        if (odd == 2 and i % 13 == 3) h.* = 0x7FC0;
    }
}

const Fns = struct { draw: cuda.Function, rows: cuda.Function, cands: cuda.Function, place: cuda.Function, top1: cuda.Function, top1_rows: cuda.Function };

fn launch(f: cuda.Function, grid: usize, s: cuda.Stream, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 1024 } }, s, a);
}

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    const c = glm.config.Config{};
    const V: usize = c.vocab;
    var spans: [world]glm.shard.Span = undefined;
    for (&spans, 0..) |*sp, q| sp.* = (try glm.shard.Share.of(&c, q, world)).vocab;
    var sm = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer sm.unload();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    const fns: Fns = .{ .draw = try sm.function("tf_draw"), .rows = try sm.function("tf_draw_rows"), .cands = try sm.function("tf_draw_cands"), .place = try sm.function("tf_draw_place"), .top1 = try gm.function("tf_glm_top1"), .top1_rows = try gm.function("tf_glm_top1_rows") };
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const host = try gpa.alloc(u16, rows * V);
    defer gpa.free(host);
    const part = try gpa.alloc(u16, rows * V);
    defer gpa.free(part);
    var full = try cuda.DeviceBuffer.alloc(gpu.d, rows * V * 2);
    defer full.free();
    var local = try cuda.DeviceBuffer.alloc(gpu.d, rows * V * 2); // rank q's rows packed at rows * spans[q].lo
    defer local.free();
    var cand = try cuda.DeviceBuffer.alloc(gpu.d, world * rows * 1024 * 8);
    defer cand.free();
    var placed = try cuda.DeviceBuffer.alloc(gpu.d, rows * V * 2);
    defer placed.free();
    var ctl = try cuda.DeviceBuffer.alloc(gpu.d, rows * (@sizeOf(Rule) + 8) + @sizeOf(Rule) + 16);
    defer ctl.free();
    var toks = try cuda.DeviceBuffer.alloc(gpu.d, 2 * rows * 4);
    defer toks.free();
    var picks = try cuda.DeviceBuffer.alloc(gpu.d, 2 * world * rows * 16);
    defer picks.free();
    try picks.fill8(0, null);
    const rules_at = ctl.ptr; // [rows]Rule, then metas [rows]i32, src [rows]u32, the one rule, its meta
    const metas_at = rules_at + rows * @sizeOf(Rule);
    const src_at = metas_at + rows * 4;
    const rule_at = src_at + rows * 4;
    const meta_at = rule_at + @sizeOf(Rule);
    var src: [rows]u32 = undefined;
    for (&src, 0..) |*x, r| x.* = @intCast(r);
    try ctl.uploadAsync(src_at - ctl.ptr, std.mem.sliceAsBytes(&src), s.handle);
    const temps = [_]f64{ 0.6, 1.0, 1.5 };
    const ks = [_]u32{ 1, 2, 20, 40, 100, 1024 };
    const ps = [_]f64{ 0.9, 0.95, 1.0 };
    const mins = [_]f64{ 0.0, 0.05 };
    var greedy_rows: usize = 0;
    var greedy_differ: usize = 0;
    var draws: usize = 0;
    var differ: usize = 0;
    for (std.enums.values(Kind), 0..) |kind, ki| for ([_]u2{ 0, 1, 2 }) |odd| {
        for (0..rows) |r| fill(kind, ki * 131 + r + 977 * @as(u64, odd), odd, host[r * V ..][0..V]);
        for (spans) |sp| for (0..rows) |r| @memcpy(part[rows * sp.lo + r * sp.len() ..][0..sp.len()], host[r * V + sp.lo ..][0..sp.len()]);
        try full.uploadAsync(0, std.mem.sliceAsBytes(host), s.handle);
        try local.uploadAsync(0, std.mem.sliceAsBytes(part), s.handle);
        // greedy: rank q's picks a row at a time (the engine's loop), then all rows in one launch, beside them
        for (spans, 0..) |sp, q| {
            const base = local.ptr + rows * sp.lo * 2;
            for (0..rows) |r| {
                var a: cuda.Args = .{};
                a.add(base + r * sp.len() * 2);
                a.add(@as(i64, @intCast(sp.len())));
                a.add(@as(u32, @intCast(sp.lo)));
                a.add(picks.ptr + (q * rows + r) * 16);
                try launch(fns.top1, 1, s, &a);
            }
            var a: cuda.Args = .{};
            a.add(base);
            a.add(@as(i64, @intCast(sp.len())));
            a.add(@as(u32, @intCast(sp.lo)));
            a.add(@as(i64, @intCast(sp.len())));
            a.add(picks.ptr + (world + q) * rows * 16);
            try launch(fns.top1_rows, rows, s, &a);
        }
        try s.synchronize();
        var got_picks: [2 * world * rows * 16]u8 = undefined;
        try picks.download(0, &got_picks);
        greedy_rows += world * rows;
        if (!std.mem.eql(u8, got_picks[0 .. world * rows * 16], got_picks[world * rows * 16 ..])) greedy_differ += 1;
        if (odd == 2) continue; // NaN logits: greedy only (the engine's sampled rows hold none)
        for (temps) |t| for (ks) |k| for (ps) |p| for (mins) |mp| {
            const sm_rule: lanes.Sampling = .{ .seed = 0x5EED_0000 + ki * 7 + draws, .temperature = t, .top_k = k, .top_p = p, .min_p = mp };
            const rule = Rule.from(lanes.gpu_rule.rule(sm_rule));
            const first: i32 = @intCast(100 + draws % 997);
            const rules: [rows]Rule = @splat(rule);
            var metas: [rows]i32 = undefined;
            for (&metas, 0..) |*m, r| m.* = first + @as(i32, @intCast(r));
            try ctl.uploadAsync(0, std.mem.sliceAsBytes(&rules), s.handle);
            try ctl.uploadAsync(metas_at - ctl.ptr, std.mem.sliceAsBytes(&metas), s.handle);
            try ctl.uploadAsync(rule_at - ctl.ptr, std.mem.asBytes(&rule), s.handle);
            try ctl.uploadAsync(meta_at - ctl.ptr, std.mem.asBytes(&[4]i32{ first, 0, 0, 0 }), s.handle);
            var a: cuda.Args = .{}; // the full gather's draw: tf_draw on whole rows
            a.add(full.ptr);
            a.add(@as(u32, @intCast(V)));
            a.add(rule_at);
            a.add(meta_at);
            a.add(@as(i32, 0));
            a.add(toks.ptr);
            a.add(@as(u64, 0));
            try launch(fns.draw, rows, s, &a);
            for (spans, 0..) |sp, q| {
                var b: cuda.Args = .{};
                b.add(local.ptr + rows * sp.lo * 2);
                b.add(@as(u32, @intCast(sp.len())));
                b.add(@as(u32, @intCast(sp.lo)));
                b.add(rules_at);
                b.add(src_at);
                b.add(k);
                b.add(cand.ptr + q * rows * k * 8);
                try launch(fns.cands, rows, s, &b);
            }
            var pl: cuda.Args = .{};
            pl.add(cand.ptr);
            pl.add(@as(u32, world));
            pl.add(@as(u32, rows));
            pl.add(k);
            pl.add(placed.ptr);
            pl.add(@as(u32, @intCast(V)));
            try launch(fns.place, rows, s, &pl);
            var d: cuda.Args = .{};
            d.add(placed.ptr);
            d.add(@as(u32, @intCast(V)));
            d.add(rules_at);
            d.add(metas_at);
            d.add(toks.ptr + rows * 4);
            try launch(fns.rows, rows, s, &d);
            try s.synchronize();
            var got: [2 * rows]u32 = undefined;
            try toks.download(0, std.mem.sliceAsBytes(&got));
            for (0..rows) |r| {
                draws += 1;
                if (got[r] == got[rows + r]) continue;
                differ += 1;
                std.debug.print("DIFFER {s} odd {d} row {d} T {d} top_k {d} top_p {d} min_p {d}: full {d} candidates {d}\n", .{ @tagName(kind), odd, r, t, k, p, mp, got[r], got[rows + r] });
            }
        };
    };
    std.debug.print("RESULT glm-draw1: {d} greedy rows ({d} sets differ), {d} sampled draws, {d} differ\n", .{ greedy_rows, greedy_differ, draws, differ });
    try check.expect(greedy_differ == 0, "glm-draw1: tf_glm_top1_rows differs from tf_glm_top1 in {d} sets", .{greedy_differ});
    try check.expect(differ == 0, "glm-draw1: {d} of {d} candidate draws differ from the full gather's", .{ differ, draws });
    check.pass("glm-draw1: {d} greedy picks and {d} sampled draws equal the full gather's", .{ greedy_rows, draws });
}
