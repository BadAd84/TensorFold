//! The speed hunt's DFlash2 knobs (TF_GLM_OVN_DRAFT_*), read once at load on every rank; unset is today's drafter.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");

/// A stream count's chain cut (TF_GLM_OVN_DRAFT_CUTS="1:0.3,2:0.3,4:0.7"): the largest key at most the count applies.
pub const Cuts = struct {
    n: usize = 0,
    keys: [16]usize = undefined,
    cuts: [16]f64 = undefined,

    pub fn parse(text: []const u8) !Cuts {
        var c: Cuts = .{};
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |raw| {
            const item = std.mem.trim(u8, raw, " ");
            const colon = std.mem.indexOfScalar(u8, item, ':') orelse return error.BadOvnKnob;
            if (c.n == c.keys.len) return error.BadOvnKnob;
            const k = std.fmt.parseInt(usize, std.mem.trim(u8, item[0..colon], " "), 10) catch return error.BadOvnKnob;
            const v = std.fmt.parseFloat(f64, std.mem.trim(u8, item[colon + 1 ..], " ")) catch return error.BadOvnKnob;
            if (k == 0 or !(v >= 0 and v <= 1)) return error.BadOvnKnob;
            for (c.keys[0..c.n]) |x| if (x == k) return error.BadOvnKnob;
            c.keys[c.n] = k;
            c.cuts[c.n] = v;
            c.n += 1;
        }
        if (c.n == 0) return error.BadOvnKnob;
        return c;
    }

    /// The cut for `streams` streams in the pass: the largest key at most `streams`, else null (the default cut).
    pub fn at(c: *const Cuts, streams: usize) ?f64 {
        var best: ?usize = null;
        for (c.keys[0..c.n], 0..) |k, i| if (k <= streams and (best == null or k > c.keys[best.?])) {
            best = i;
        };
        return if (best) |i| c.cuts[i] else null;
    }
};

/// The drafter's knobs; every default is the drafter as it runs without them.
pub const Knobs = struct {
    onepass: bool = false, // ONEPASS: a pass's 4-bit linears in launches of q4.many_rows rows
    down4: bool = false, // DOWN4: the MLP's down at 4 bits (production's quantize4)
    mlpsplit: usize = 0, // MLPSPLIT=N: the MLP split over the ranks (world 3+) while at most N streams are in a pass
    edge: f32 = 1.0, // EDGE: the selector's edge weight in a pick
    noisy: bool = false, // BETA set: the chances are softmax((score + noise) / beta), production's noisy_confidence
    noise: f32 = 0, // NOISE: the target's keyed Gumbel draw's weight in a sampled stream's pick (0: off)
    keepwin: bool = false, // KEEPWIN: kept prompt states carry the drafter's window rows and its start
    cuts: ?Cuts = null, // CUTS: the chain cut by streams in the pass (null: TF_GLM_DRAFT_CONFIDENCE, today's)
    fixed: bool = false, // FIXED: verify every draft the cut keeps (production's fnc7:cut), not a cost-sized window

    /// The knobs from the environment (every rank's is the same); `beta` the chances' temperature (1.0 unset).
    pub fn read(beta: *f64) !Knobs {
        var k: Knobs = .{ .onepass = cuda.ovn.on("DRAFT_ONEPASS"), .down4 = cuda.ovn.on("DRAFT_DOWN4"), .keepwin = cuda.ovn.on("DRAFT_KEEPWIN"), .fixed = cuda.ovn.on("DRAFT_FIXED") };
        k.mlpsplit = @intCast(try cuda.ovn.int("DRAFT_MLPSPLIT", 0));
        if (try float("DRAFT_EDGE")) |v| k.edge = @floatCast(v);
        if (try float("DRAFT_BETA")) |v| {
            if (!(v > 0)) return error.BadOvnKnob;
            beta.* = v;
            k.noisy = true;
        }
        if (try float("DRAFT_NOISE")) |v| {
            if (!(v >= 0)) return error.BadOvnKnob;
            k.noise = @floatCast(v);
        }
        if (cuda.ovn.text("DRAFT_CUTS")) |t| k.cuts = try Cuts.parse(t);
        return k;
    }
};

/// `TF_GLM_OVN_<name>` as a finite float (unset or empty: null); anything else is refused.
pub fn float(comptime name: []const u8) !?f64 {
    const s = cuda.ovn.text(name) orelse return null;
    const v = std.fmt.parseFloat(f64, std.mem.trim(u8, s, " ")) catch return error.BadOvnKnob;
    if (!std.math.isFinite(v)) return error.BadOvnKnob;
    return v;
}

/// The target's keyed Gumbel draw for token `id` at position `pos` (sample.cu's race: -log(-log(u)), u below 1).
pub fn gumbel(seed: u64, pos: i64, id: u32) f32 {
    const u = lanes.gpu_full.raceUniform(seed, @intCast(pos), id);
    return -@log(-@log(u));
}

test "a stream count takes the cut of the largest key at most it; no such key is the default" {
    const c = try Cuts.parse("1:0.3, 2:0.3,4:0.7");
    try std.testing.expectEqual(@as(?f64, 0.3), c.at(1));
    try std.testing.expectEqual(@as(?f64, 0.3), c.at(3));
    try std.testing.expectEqual(@as(?f64, 0.7), c.at(4));
    try std.testing.expectEqual(@as(?f64, 0.7), c.at(16));
    const late = try Cuts.parse("4:0.5");
    try std.testing.expectEqual(@as(?f64, null), late.at(2));
    for ([_][]const u8{ "", "1", "0:0.3", "1:x", "1:1.5", "1:-0.1", "1:0.3,1:0.4" }) |bad| try std.testing.expectError(error.BadOvnKnob, Cuts.parse(bad));
}

test "the host's keyed uniform is sample.cu's: (top 24 bits of the splitmix chain + 0.5f) / 2^24 in float32" {
    // bits from sample.cu's mix/uniform arithmetic (u64 wrapping, float32 add rounding to nearest even)
    const known = [_]struct { seed: u64, pos: u32, id: u32, bits: u32 }{
        .{ .seed = 0, .pos = 0, .id = 0, .bits = 0x3e4ffa2e },
        .{ .seed = 7, .pos = 1, .id = 5, .bits = 0x3f04b384 },
        .{ .seed = 7, .pos = 66903620, .id = 1500, .bits = 0x3f800000 }, // the top value: 1.0, which the race clamps
        .{ .seed = 123456789, .pos = 4097, .id = 154855, .bits = 0x3e48b596 },
        .{ .seed = std.math.maxInt(u64), .pos = std.math.maxInt(u32), .id = std.math.maxInt(u32), .bits = 0x3ebc25b3 },
        .{ .seed = 42, .pos = 1000, .id = 77, .bits = 0x3f6de160 },
    };
    for (known) |k| try std.testing.expectEqual(k.bits, @as(u32, @bitCast(lanes.gpu_rule.uniform24(k.seed, k.pos, k.id))));
    try std.testing.expect(std.math.isFinite(gumbel(7, 66903620, 1500))); // the clamped draw stays finite
    var u: f32 = @bitCast(@as(u32, 0x3f04b384)); // runtime, as gumbel computes it
    _ = &u;
    try std.testing.expectEqual(-@log(-@log(u)), gumbel(7, 1, 5));
}
