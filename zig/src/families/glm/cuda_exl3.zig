//! GLM-5.3-Flash's EXL3 routed experts on CUDA (Mia-AiLab's checkpoint: 4-bit mcg trellis experts, the rest bf16).

const std = @import("std");
const modelopt = @import("core").modelopt;
const shard = @import("cuda_shard.zig");
const Span = shard.Span;
const Config = @import("config.zig").Config;
const cuda = @import("cuda");
const experts = @import("cuda_experts.zig");

/// config.json's quantization_config (top level, else text_config's), when it has one.
fn block(root: std.json.Value) ?std.json.ObjectMap {
    if (root != .object) return null;
    if (root.object.get("quantization_config")) |q| if (q == .object) return q.object;
    if (root.object.get("text_config")) |t| if (t == .object) if (t.object.get("quantization_config")) |q| if (q == .object) return q.object;
    return null;
}

/// Whether config.json names EXL3 (quant_method "exl3"): `admit` then says whether this family reads it.
pub fn isExl3(gpa: std.mem.Allocator, config_json: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json, .{}) catch return false;
    defer parsed.deinit();
    const q = block(parsed.value) orelse return false;
    const m = q.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "exl3");
}

/// The EXL3 checkpoints the kernels read: 4-bit routed experts on the "mcg" codebook (everything else bf16).
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *modelopt.Why) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer parsed.deinit();
    const q = block(parsed.value) orelse {
        why.set("no quantization_config", .{});
        return error.UnsupportedQuant;
    };
    const bits = q.get("bits") orelse std.json.Value{ .null = {} };
    if (bits != .integer or bits.integer != 4) {
        why.set("EXL3 at {f} bits: the CUDA family reads 4-bit EXL3 experts", .{std.json.fmt(bits, .{})});
        return error.UnsupportedQuant;
    }
    const book = q.get("codebook") orelse std.json.Value{ .null = {} };
    if (book != .string or !std.mem.eql(u8, book.string, "mcg")) {
        why.set("EXL3 codebook {f}: the CUDA family reads \"mcg\"", .{std.json.fmt(book, .{})});
        return error.UnsupportedQuant;
    }
}

/// One layer's EXL3 experts on a rank: trellis words [E][K/16][N/16][32] and fp16 scales (suh [E][K], svh [E][N]).
pub const Experts = struct {
    gt: u64,
    ut: u64,
    dt: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    k2: u64, // independent per-expert projection widths in the common module
    shx: bool, // every expert's gate and up share one input rotation: a prompt rotates a row once, not once a pick
    gb: u64 = 0, // the gate trellis of every expert back to back (gt's pointers are gb + e * one expert's bytes)
    ub: u64 = 0,
    db: u64 = 0,
    ovn: cuda.exl3_experts.Ovn = .{}, // the TF_GLM_OVN_X3_* decode variants, read once at load
    rot: common.Rot = .{}, // TF_GLM_OVN_X3_ROTEXPERT: per-expert share widths (gb + rot.before(e) columns' words)
};

/// TF_GLM_OVN_X3_NC (1, 2 or w), TF_GLM_OVN_X3_FUSEDOWN and TF_GLM_OVN_X3_XROW: the decode variants (unset: none).
fn ovnKnobs() !cuda.exl3_experts.Ovn {
    const nc = cuda.ovn.text("X3_NC") orelse "0";
    const eql = std.mem.eql;
    const loads: cuda.exl3_experts.Ovn.Loads = if (eql(u8, nc, "0")) .off else if (eql(u8, nc, "1")) .nc1 else if (eql(u8, nc, "2")) .nc2 else if (eql(u8, nc, "w")) .word else return error.BadOvnKnob;
    return .{ .loads = loads, .fuse_down = cuda.ovn.on("X3_FUSEDOWN"), .xrow = cuda.ovn.on("X3_XROW"), .items = cuda.ovn.on("ITEMS") };
}

/// TF_GLM_OVN_X3_ROTEXPERT (a numerics trade): rank r holds share (r - e mod world) mod world of expert e, the
/// standard split's column ranges, so each rank holds the wide share of a world-th of the experts. Off: none.
pub fn rotation(inter: usize, rank: usize, world: usize) !common.Rot {
    if (!cuda.ovn.on("X3_ROTEXPERT")) return .{};
    return rotationOf(inter, rank, world) catch |err| {
        std.log.err("TF_GLM_OVN_X3_ROTEXPERT: {d} expert columns over {d} ranks split into more than one wide share", .{ inter, world });
        return err;
    };
}

/// rotation() with the knob on: one wide share (the split's first), the rest equal; an even split rotates nothing.
pub fn rotationOf(inter: usize, rank: usize, world: usize) !common.Rot {
    const wide = (try shard.span(inter, 128, 0, world)).len();
    const narrow = (try shard.span(inter, 128, world - 1, world)).len();
    if (wide == narrow) return .{};
    for (1..world) |s| if ((try shard.span(inter, 128, s, world)).len() != narrow) return error.RotExpertShares;
    return .{ .m = world, .r = rank, .wn = narrow, .wx = wide - narrow };
}

/// Expert e's columns on `rank` of `world`: the standard split's share (rank - e mod world) mod world under `rot`.
pub fn expertCut(inter: usize, rank: usize, world: usize, e: usize, rot: common.Rot) !Span {
    return shard.span(inter, 128, if (rot.on()) (rank + world - e % world) % world else rank, world);
}

/// The routed width a rank's kernels and buffers take: its widest share (TF_GLM_OVN_EXPERT_ROTATE: over the layers,
/// a layer's own on that layer's Share.at), or the wide share under TF_GLM_OVN_X3_ROTEXPERT.
pub fn routedWidth(share: shard.Share, c: *const Config) !usize {
    const rot = try rotation(c.moe_inter, share.rank, share.world);
    return if (rot.on()) rot.wn + rot.wx else share.widest(c)[0];
}

fn cutOf(inter: usize, share: shard.Share, e: usize, rot: common.Rot) !Span {
    return if (rot.on()) expertCut(inter, share.rank, share.world, e, rot) else share.moe;
}

const tile = 128; // a 16 x 16 tile's bytes at 4 bits

/// One expert's share of a K x N trellis: gate, up (`by_cols`) every tile row's tile columns of `cut`, down its rows.
pub fn trellisShare(gpa: std.mem.Allocator, host: *std.ArrayList(u8), bytes: []const u8, k: usize, n: usize, by_cols: bool, cut: Span) !void {
    const kt = (if (by_cols) k else cut.len()) / 16;
    const nt = (if (by_cols) cut.len() else n) / 16;
    if (by_cols) {
        const row = (n / 16) * tile;
        for (0..kt) |r| try host.appendSlice(gpa, bytes[r * row + (cut.lo / 16) * tile ..][0 .. nt * tile]);
    } else try host.appendSlice(gpa, bytes[(cut.lo / 16) * nt * tile ..][0 .. kt * nt * tile]);
}

/// The bytes trellisShare appends.
pub fn shareBytes(k: usize, n: usize, by_cols: bool, cut: Span) usize {
    return (if (by_cols) k else cut.len()) / 16 * ((if (by_cols) cut.len() else n) / 16) * tile;
}

/// One expert's `cut` of an fp16 scale vector, zero-padded to `pad` values (the wide share's under the rotation).
pub fn scaleShare(gpa: std.mem.Allocator, host: *std.ArrayList(u8), bytes: []const u8, cut: Span, pad: usize) !void {
    try host.appendSlice(gpa, bytes[cut.lo * 2 .. cut.hi * 2]);
    try host.appendNTimes(gpa, 0, (pad - cut.len()) * 2);
}

/// Layer `i`'s EXL3 experts through `L` (cuda_weights' loader): gate, up by each expert's share's tile columns, down
/// by its rows (`share.moe`, or cutOf's per-expert share under TF_GLM_OVN_X3_ROTEXPERT, scales padded to the widest).
pub fn load(L: anytype, i: u32, count: usize, dims: usize, inter: usize, share: shard.Share) !Experts {
    const cut = share.moe;
    if (dims % 128 != 0 or cut.lo % 128 != 0 or cut.hi % 128 != 0) return error.NotExl3Mcg4;
    const rot = try rotation(inter, share.rank, share.world);
    const pad = if (rot.on()) rot.wn + rot.wx else cut.len(); // every expert's gate/up svh and down suh row
    var out: Experts = undefined;
    out.shx = true;
    const Mat = struct { k: usize, n: usize, by_cols: bool };
    const mats = [_]Mat{
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = inter, .n = dims, .by_cols = false },
    };
    var first_suh: ?[]const u8 = null;
    inline for (mats, .{ "gate_proj", "up_proj", "down_proj" }, .{ "gt", "ut", "dt" }, .{ "gb", "ub", "db" }, .{ "suh_g", "suh_u", "suh_d" }, .{ "svh_g", "svh_u", "svh_d" }) |m, proj, tw, tb, su, sv| {
        for (0..count) |e| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".trellis", .{ i, e });
            if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != m.k / 16 or t.dim(1) != m.n / 16 or t.dim(2) != 64) return error.NotExl3Mcg4;
            const marker = L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".mcg", .{ i, e }) catch return error.NotExl3Mcg4;
            if (marker.dtype != .i32 or marker.numel() != 1 or marker.bytes.len != 4 or
                std.mem.readInt(u32, marker.bytes[0..4], .little) != @import("core").exl3_load.markers[0].value)
                return error.NotExl3Mcg4;
            try trellisShare(L.gpa, &L.host, t.bytes, m.k, m.n, m.by_cols, try cutOf(inter, share, e, rot));
        }
        const base = try L.put("layers.{d}.moe.exl3." ++ tw, .{i});
        @field(out, tb) = base;
        var at: u64 = base;
        for (0..count) |e| {
            const ptr = at;
            at += shareBytes(m.k, m.n, m.by_cols, try cutOf(inter, share, e, rot));
            try L.host.appendSlice(L.gpa, std.mem.asBytes(&ptr));
        }
        @field(out, tw) = try L.put("layers.{d}.moe.exl3." ++ tw ++ "_ptrs", .{i});
        for (0..count) |e| { // suh: the input's scales (gate, up: every input; down: the expert's share)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".suh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.k) return error.NotExl3Mcg4;
            if (m.by_cols) { // one rotation shared by the layer's every gate and up input?
                if (first_suh) |f| (if (!std.mem.eql(u8, f, t.bytes)) {
                    out.shx = false;
                }) else first_suh = t.bytes;
                try L.host.appendSlice(L.gpa, t.bytes);
            } else try scaleShare(L.gpa, &L.host, t.bytes, try cutOf(inter, share, e, rot), pad);
        }
        @field(out, su) = try L.put("layers.{d}.moe.exl3." ++ su, .{i});
        for (0..count) |e| { // svh: the output's scales (gate, up: the expert's share; down: every output)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".svh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.n) return error.NotExl3Mcg4;
            if (m.by_cols) try scaleShare(L.gpa, &L.host, t.bytes, try cutOf(inter, share, e, rot), pad) else try L.host.appendSlice(L.gpa, t.bytes);
        }
        @field(out, sv) = try L.put("layers.{d}.moe.exl3." ++ sv, .{i});
    }
    for (0..count) |_| {
        const bits: i32 = 8;
        try L.host.appendSlice(L.gpa, std.mem.asBytes(&bits));
    }
    out.k2 = try L.put("layers.{d}.moe.exl3.k2", .{i});
    out.ovn = try ovnKnobs();
    out.rot = rot;
    if (out.ovn.items and rot.on()) { // the item plan's grouped kernels have no per-expert share (rotexpert) path
        std.log.err("TF_GLM_OVN_ITEMS with TF_GLM_OVN_X3_ROTEXPERT: unsupported together", .{});
        return error.BadOvnKnob;
    }
    return out;
}

/// The selected common EXL3 owner. This file adapts the family's checkpoint shard and slot layout only.
pub const Kernels = cuda.exl3_experts.Experts;
const common = cuda.exl3_experts;
pub const Scratch = struct { picks: u64, xg: u64, xu: u64, z: u64 };
pub const top_k = 8;
/// Rows a call to the common experts takes: each call reads every expert's weights in 64-row tiles, so a prompt chunk
/// (cuda_state.prefill_rows) runs as one call, where two calls of half a chunk would read every expert twice.
pub const chunk_rows = @import("cuda_state.zig").prefill_rows;
fn layer(w: Experts, count: usize, dims: usize, width: usize, limit: f32) common.Layer {
    return .{ .gate = w.gt, .up = w.ut, .down = w.dt, .gate_k2 = w.k2, .up_k2 = w.k2, .down_k2 = w.k2, .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = count, .dims = dims, .width = width, .codebook = .mcg, .limit = limit, .act_mode = 0, .uniform_k2 = 8, .shared_input = w.shx, .gate_base = w.gb, .up_base = w.ub, .down_base = w.db, .rot = w.rot };
}
fn shape(dims: usize, width: usize) common.Layer {
    const w: Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = false };
    return layer(w, cuda.grouped.max_experts, dims, width, 0);
}
/// A chunk's compacted picks ahead of the common scratch, 256-aligned; the usize keeps @min from narrowing to u7.
fn picksBytes(rows: usize) usize {
    const cap: usize = @min(rows, chunk_rows);
    return std.mem.alignForward(usize, cap * top_k * 4, 256);
}
/// The prompt path's TF_GLM_OVN_* switches (EY_DIRECT, X3_PROMPT_K64, _ORDER, X3_YBF16, X3_ROTEXPERT); all off.
pub const Ovn = common.prompt.Ovn;
/// `o`: the switches the kernels run with (Kernels' `prompt.ovn`; Ovn.fromEnv() before they load).
pub fn scratchBytes(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool, o: Ovn) [4]usize {
    const n: usize = @min(rows, chunk_rows);
    var l = shape(dims, width);
    l.shared_input = shx;
    if (prefill and common.prompt.eligible(l)) return .{ picksBytes(rows) + (common.prompt.bytes(l, n, top_k, o) catch unreachable), 4, 4, 4 };
    return .{ picksBytes(rows) + (common.scratchBytes(shape(dims, width), n, top_k) catch unreachable), 4, 4, 4 };
}
/// scratchBytes for every routed width `share` holds (TF_GLM_OVN_EXPERT_ROTATE: per layer), each size the largest.
pub fn heldScratch(rows: usize, dims: usize, share: shard.Share, c: *const Config, prefill: bool, shx: bool, o: Ovn) ![4]usize {
    var out = scratchBytes(rows, dims, try routedWidth(share, c), prefill, shx, o);
    if (!share.rot.on) return out;
    for (0..c.layers) |i| {
        const got = scratchBytes(rows, dims, (try share.at(c, i)).moe.len(), prefill, shx, o);
        for (&out, got) |*x, y| x.* = @max(x.*, y);
    }
    return out;
}
/// TF_GLM_OVN_X3_YBF16 (a numerics trade): a fused EXL3 prompt chunk's slot rows in bf16, as an NVFP4 chunk's.
pub fn bf16Slots(k: *const Kernels, dims: usize, width: usize) bool {
    return k.prompt.ovn.ybf16 and common.prompt.eligible(shape(dims, width));
}
/// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst`, in one call.
fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
}
pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64, xd: u64) !void {
    _ = xd;
    if (slots != top_k + 1) return error.UnsupportedTopK;
    if (dims % 256 != 0 or width % 128 != 0) return error.NotExl3Mcg4;
    const l = layer(w, count, dims, width, limit);
    const cap: usize = @min(rows, chunk_rows);
    const arena = sc.picks + picksBytes(rows);
    const fused = prompt and common.prompt.eligible(l);
    const o = k.prompt.ovn;
    const eb: usize = if (fused and o.ybf16) 2 else 4; // a slot value's bytes (bf16Slots)
    var at: usize = 0;
    while (at < rows) : (at += chunk_rows) {
        const n: usize = @min(chunk_rows, rows - at);
        if (!prompt and (w.ovn.any() or w.rot.on())) { // TF_GLM_OVN_X3_ROTEXPERT runs only the variants' kernels
            try decodeOvn(k, stream, l, w.ovn, sc.picks, arena, cap, x + at * x_stride * 2, x_stride, picks + at * slots * 4, n, ey + at * slots * dims * 4, &ek.router, plan_buf);
            continue;
        }
        try copyRows(stream, sc.picks, top_k * 4, picks + at * slots * 4, slots * 4, top_k * 4, n, "EXL3 compact route rows");
        const output = if (fused) blk: {
            const scratch = try common.prompt.scratch(arena, l, cap, top_k, o);
            const to: common.prompt.Out = if (o.ey_direct) .{ .y = ey + at * slots * dims * eb, .slots = slots } else .{ .y = scratch.y, .slots = top_k };
            try k.prompt.run(k, ek.router, stream, l, x + at * x_stride * 2, x_stride, sc.picks, n, scratch, plan_buf, to);
            break :blk scratch.y;
        } else blk: {
            const scratch = try common.scratch(arena, l, cap, top_k);
            try k.runSlots(stream, l, x + at * x_stride * 2, x_stride, .bf16, sc.picks, n, scratch);
            break :blk scratch.y;
        };
        if (fused and o.ey_direct) continue; // EY_DIRECT: the down kernel wrote ey's routed slots in place
        try copyRows(stream, ey + at * slots * dims * eb, slots * dims * eb, output, top_k * dims * eb, top_k * dims * eb, n, "EXL3 routed output slots");
    }
}

/// A decode window's routed experts through the TF_GLM_OVN_X3_* variants: ey's routed slot rows get the same bytes.
fn decodeOvn(k: *const Kernels, stream: cuda.Stream, l: common.Layer, o: common.Ovn, compact: u64, arena: u64, cap: usize, x: u64, x_stride: usize, picks: u64, n: usize, ey: u64, router: *const cuda.grouped.Router, plan: cuda.grouped.Plan) !void {
    const slots = top_k + 1;
    const d = l.dims;
    // TF_GLM_OVN_ITEMS: route the 9-slot picks and run the grouped kernels on items; the 9-slot layout (no y)
    // fits the arena the plain 8-slot path was sized for, and the down epilogue writes ey's slot rows in place.
    if (o.items) {
        if ((try common.fusedBytes(l, cap, slots)) > (try common.scratchBytes(l, cap, top_k))) return error.TileNotBuilt;
        const scratch = try common.fusedScratch(arena, l, cap, slots);
        return k.runWith(stream, l, x, x_stride, .bf16, picks, 0, n, 0, scratch, o, ey, .{ .router = router, .plan = plan });
    }
    // the fused layout (slots 9, no y) fits inside the plain one this arena was sized for
    if (o.fuse_down and common.fusable(l) and (try common.fusedBytes(l, cap, slots)) <= (try common.scratchBytes(l, cap, top_k))) {
        const scratch = try common.fusedScratch(arena, l, cap, slots);
        return k.runWith(stream, l, x, x_stride, .bf16, picks, 0, n, 0, scratch, o, ey, null);
    }
    var plain = o;
    plain.fuse_down = false;
    try copyRows(stream, compact, top_k * 4, picks, slots * 4, top_k * 4, n, "EXL3 compact route rows");
    const scratch = try common.scratch(arena, l, cap, top_k);
    try k.runWith(stream, l, x, x_stride, .bf16, compact, 0, n, 0, scratch, plain, 0, null);
    try copyRows(stream, ey, slots * d * 4, scratch.y, top_k * d * 4, top_k * d * 4, n, "EXL3 routed output slots");
}

test "EXL3 admission: Mia-AiLab's block passes; other widths and codebooks are refused by name" {
    const gpa = std.testing.allocator;
    const mia =
        \\{"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}}
    ;
    try std.testing.expect(isExl3(gpa, mia));
    var why: modelopt.Why = .{};
    try admit(gpa, mia, &why);
    try std.testing.expect(!isExl3(gpa, "{\"quantization_config\": {\"quant_method\": \"modelopt\"}}"));
    try std.testing.expect(!isExl3(gpa, "{}"));
    try std.testing.expectError(error.UnsupportedQuant, admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3, \"codebook\": \"mcg\"}}", &why));
    try std.testing.expectEqualStrings("EXL3 at 3 bits: the CUDA family reads 4-bit EXL3 experts", why.text());
    try std.testing.expectError(error.UnsupportedQuant, admit(gpa, "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"mul1\"}}}", &why));
    try std.testing.expectEqualStrings("EXL3 codebook \"mul1\": the CUDA family reads \"mcg\"", why.text());
}

test "a chunk's compacted picks end before the common scratch at every row count" {
    for (1..3 * chunk_rows) |rows| {
        const cap: usize = @min(rows, chunk_rows);
        try std.testing.expect(picksBytes(rows) >= cap * top_k * 4);
        try std.testing.expectEqual(@as(usize, 0), picksBytes(rows) % 256);
    }
}

test "GLM's routed experts run on the common EXL3 module's kernels" {
    try std.testing.expect(Kernels == cuda.exl3_experts.Experts);
}

test "TF_GLM_OVN_X3_ROTEXPERT at TP=3: each expert's three shares are the standard ones, rank r wide where e mod 3 == r" {
    const t = std.testing;
    const inter = 2048;
    var held: [3]usize = @splat(0);
    for (0..3) |r| {
        const rot = try rotationOf(inter, r, 3);
        try t.expectEqual(common.Rot{ .m = 3, .r = r, .wn = 640, .wx = 128 }, rot);
        var at: usize = 0; // the loader's back-to-back layout, as the kernels address it (rot.before)
        for (0..288) |e| {
            const c = try expertCut(inter, r, 3, e, rot);
            try t.expectEqual(c, try shard.span(inter, 128, (r + 3 - e % 3) % 3, 3)); // the unrotated rank's share
            try t.expectEqual(rot.cols(e), c.len());
            try t.expectEqual((e % 3 == r), c.len() == 768);
            try t.expectEqual(at, rot.before(e));
            at += c.len();
        }
        held[r] = at;
    }
    for (0..288) |e| { // the three ranks' shares of an expert cover its 16 blocks once
        var covered: [16]u8 = @splat(0);
        for (0..3) |r| {
            const c = try expertCut(inter, r, 3, e, try rotationOf(inter, r, 3));
            for (c.lo / 128..c.hi / 128) |b| covered[b] += 1;
        }
        for (covered) |n| try t.expectEqual(@as(u8, 1), n);
    }
    try t.expectEqual([3]usize{ 196608, 196608, 196608 }, held); // 96 x 768 + 192 x 640; unrotated 288 x 768 on rank 0
    try t.expectEqual(common.Rot{}, try rotationOf(inter, 1, 2)); // even splits: nothing to spread
    try t.expectEqual(common.Rot{}, try rotationOf(inter, 3, 4));
    try t.expectEqual(common.Rot{}, try rotationOf(inter, 0, 1));
    try t.expectError(error.RotExpertShares, rotationOf(inter, 0, 6)); // 3/3/3/3/2/2 blocks: four wide shares
    try t.expectEqual(try shard.span(inter, 128, 1, 3), try expertCut(inter, 1, 3, 5, .{})); // off: the rank's share
}

test "an expert's share of a trellis and of its scales is the plain loader's bytes; scales zero-padded to the widest" {
    const t = std.testing;
    const gpa = t.allocator;
    var src: [2 * 16 * tile]u8 = undefined; // gate/up: K 32, N 256 (2 x 16 tiles); down: K 256, N 32 (16 x 2 tiles)
    for (&src, 0..) |*b, j| b.* = @truncate(j *% 7 +% j / 251);
    const cut: Span = .{ .lo = 128, .hi = 256 };
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(gpa);
    try trellisShare(gpa, &host, &src, 32, 256, true, cut);
    try t.expectEqual(shareBytes(32, 256, true, cut), host.items.len);
    for (0..2) |r| try t.expectEqualSlices(u8, src[r * 16 * tile + 8 * tile ..][0 .. 8 * tile], host.items[r * 8 * tile ..][0 .. 8 * tile]);
    host.clearRetainingCapacity();
    try trellisShare(gpa, &host, &src, 256, 32, false, cut);
    try t.expectEqual(shareBytes(256, 32, false, cut), host.items.len);
    try t.expectEqualSlices(u8, src[8 * 2 * tile ..][0 .. 8 * 2 * tile], host.items);
    host.clearRetainingCapacity();
    try scaleShare(gpa, &host, src[0..512], cut, 256);
    try t.expectEqualSlices(u8, src[256..512], host.items[0..256]);
    try t.expect(std.mem.allEqual(u8, host.items[256..512], 0));
    host.clearRetainingCapacity();
    try scaleShare(gpa, &host, src[0..512], cut, cut.len()); // off: no padding
    try t.expectEqualSlices(u8, src[256..512], host.items);
}

test "TF_GLM_OVN_ITEMS: off by default, and on it reaches the experts Ovn options" {
    cuda.ovn.fake("ITEMS", null);
    try std.testing.expect(!(try ovnKnobs()).items);
    cuda.ovn.fake("ITEMS", "1");
    try std.testing.expect((try ovnKnobs()).items);
    cuda.ovn.fake("ITEMS", null); // leave it unset for any later test that reads the knob
}

test "TF_GLM_OVN_ITEMS: the decode plan sized for the largest shared window covers every smaller window" {
    const slots = top_k + 1; // 9 (top-8 plus the shared expert)
    const n_exp = 288; // GLM-5.3-Flash's routed experts; the shared id is `n_exp`, so the plan counts n_exp + 1
    const cap = 64; // the largest shared decode window
    // cuda_state sizes the plan members as rows * slots and the items as maxItems(rows * slots, n_exp + 1, 16)
    const members_cap = cap * slots;
    const items_cap = cuda.grouped.maxItems(cap * slots, n_exp + 1, 16);
    for (1..cap + 1) |rows| {
        try std.testing.expect(rows * slots <= members_cap);
        try std.testing.expect(cuda.grouped.maxItems(rows * slots, n_exp + 1, 16) <= items_cap);
    }
}
