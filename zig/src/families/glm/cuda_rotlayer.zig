//! TF_GLM_OVN_EXPERT_ROTATE (production's ots0105): each MoE layer's widest expert shares on a rank of its own.
//! At TP=3 an expert's 16 Hadamard blocks split 6 / 5 / 5 and rank 0 holds the 6 in every MoE layer, so its weights
//! run about 8.9 GiB over the others' and the pool's ceiling follows it. Here MoE layer i keeps the standard split's
//! column ranges but rank r holds share (r - owner_i) mod world: each share's partial comes from the same bytes
//! through the same width-keyed kernels, and the layer's glue sums the gathered partials in share order (the owner's
//! first, then on in rank order; glm_glue_rows.cuh ranks8), the order the standard split sums them in. Every bit is
//! the knob-off path's. Values: unset or 0 (the standard split), mod (layer i's owner i mod world), or C0,C1,...
//! (a count a rank over the MoE layers, interleaved by a smooth weighted round robin, ties to the lower rank).
//! Two and four ranks split the experts evenly: nothing rotates.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const Config = cfg.Config;

/// The most ranks a world holds (cuda_ranks.max_world); a schedule's counts take one each.
const max_world = @import("cuda_ranks.zig").max_world;

/// Owners.digest of a value the schedule refused: the ranks' agreement then names the knob before any rank stops.
pub const refused: u32 = 0xffff_ffff;

/// Set once at load (cuda_engine) when the owners rotate; off, the forward reads no Share.rot.
pub var active = false;

/// glm-rotlayer-check's knob-off pass: `read` answers the standard split whatever the knob says.
pub var skip = false;

/// Per layer the rank holding the first (widest) shares of its routed and shared experts.
pub const Owners = struct {
    on: bool = false,
    of: [cfg.max_layers]u8 = @splat(0), // 0 off the MoE layers and past them (the MTP layer keeps the standard split)

    /// The rank whose partial layer `layer`'s glue sums first (0 off the knob and off the MoE layers).
    pub fn first(o: *const Owners, layer: usize) usize {
        return if (o.on and layer < cfg.max_layers) o.of[layer] else 0;
    }

    /// The standard split's share rank `rank` of `world` holds at layer `layer`: (rank - first) mod world.
    pub fn held(o: *const Owners, rank: usize, world: usize, layer: usize) usize {
        return (rank + world - o.first(layer)) % world;
    }

    /// The ranks' agreement word: 0 off, else a CRC-32 of the owners (never 0 or `refused`).
    pub fn digest(o: *const Owners) u32 {
        if (!o.on) return 0;
        const h = std.hash.Crc32.hash(&o.of);
        return if (h == 0 or h == refused) 1 else h;
    }

    /// How many of the first `layers` layers' first shares each rank holds (MoE layers only, `moe`).
    pub fn counts(o: *const Owners, c: *const Config, world: usize) [max_world]usize {
        var n: [max_world]usize = @splat(0);
        for (0..@min(c.layers, cfg.max_layers)) |i| if (c.isMoe(@intCast(i))) {
            n[o.first(i) % world] += 1;
        };
        return n;
    }
};

/// The routed experts' TP unit (cuda_shard: EXL3 tiles and rotations take 128).
pub fn moeUnit(c: *const Config) usize {
    return if (c.exl3) 128 else shard.unit;
}

/// Whether both expert families split into equal shares over `world` ranks (nothing to spread).
fn even(c: *const Config, world: usize) !bool {
    const fams = [_][2]usize{ .{ c.moe_inter, moeUnit(c) }, .{ c.shared_inter, shard.unit } };
    var same = true;
    for (fams) |f| {
        const w0 = (try shard.span(f[0], f[1], 0, world)).len();
        for (1..world) |q| same = same and (try shard.span(f[0], f[1], q, world)).len() == w0;
    }
    return same;
}

/// The owners `spec` names for `c` over `world` ranks (null, "0" or "off": none); a bad value is refused.
pub fn parse(spec: ?[]const u8, c: *const Config, world: usize) !Owners {
    const v = spec orelse return .{};
    if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) return .{};
    if (world == 0 or world > max_world or c.layers > cfg.max_layers) return error.BadExpertRotate;
    var n: usize = 0; // the MoE layers the schedule spreads
    for (0..c.layers) |i| n += @intFromBool(c.isMoe(@intCast(i)));
    var counts: [max_world]u64 = @splat(0);
    const by_mod = std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "mod");
    if (!by_mod) {
        var it = std.mem.splitScalar(u8, v, ',');
        var k: usize = 0;
        var sum: u64 = 0;
        while (it.next()) |part| : (k += 1) {
            if (k == world) return error.BadExpertRotate;
            counts[k] = std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10) catch return error.BadExpertRotate;
            sum += counts[k];
        }
        if (k != world or sum != n) return error.BadExpertRotate;
    }
    if (world == 1 or try even(c, world)) return .{};
    var o: Owners = .{ .on = true };
    var credit: [max_world]i64 = @splat(0);
    for (0..c.layers) |i| if (c.isMoe(@intCast(i))) {
        if (by_mod) {
            o.of[i] = @intCast(i % world);
            continue;
        }
        for (credit[0..world], counts[0..world]) |*x, m| x.* += @intCast(m);
        var best: usize = 0;
        for (1..world) |q| if (credit[q] > credit[best]) {
            best = q;
        };
        credit[best] -= @intCast(n);
        o.of[i] = @intCast(best);
    };
    return o;
}

/// TF_GLM_OVN_EXPERT_ROTATE's owners for `c` over `world` ranks, read once at load (`skip`: the standard split).
pub fn read(c: *const Config, world: usize) !Owners {
    if (skip) return .{};
    const spec = cuda.ovn.text("EXPERT_ROTATE");
    const o = parse(spec, c, world) catch |err| {
        std.log.err("TF_GLM_OVN_EXPERT_ROTATE={s}: 0, mod, or {d} counts (one a rank) summing to the MoE layers", .{ spec orelse "", world });
        return err;
    };
    if (o.on and cuda.ovn.on("X3_ROTEXPERT")) {
        std.log.err("TF_GLM_OVN_EXPERT_ROTATE and TF_GLM_OVN_X3_ROTEXPERT both move expert shares: pick one", .{});
        return error.BadExpertRotate;
    }
    return o;
}

/// Rank 0's load line: how many MoE layers' first shares each rank holds.
pub fn note(o: *const Owners, c: *const Config, world: usize) void {
    if (!o.on) return;
    const n = o.counts(c, world);
    var buf: [96]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (n[0..world], 0..) |x, r| w.print("{s}{d}", .{ if (r == 0) "" else "/", x }) catch break;
    std.log.info("glm: expert shares rotated (TF_GLM_OVN_EXPERT_ROTATE={s}): the widest share of each MoE layer on ranks {s}, partials summed in share order (the same bits)", .{ cuda.ovn.text("EXPERT_ROTATE") orelse "", w.buffered() });
}

/// The rank whose partial layer `layer`'s glue sums first on a forward over `s` (0 unless the knob loaded owners).
pub fn first(s: *const shard.Share, layer: usize) usize {
    return if (active) s.rot.first(layer) else 0;
}

/// The glue kernels' world argument: the ranks in the low byte, the rank summed first from bit 8 (0: today's int).
pub fn word(world: usize, first_rank: usize) usize {
    return world | first_rank << 8;
}

/// The partial ranks8 reads k-th (glm_glue_rows.cuh): the first rank's, then on in rank order modulo world.
pub fn slot(k: usize, first_rank: usize, world: usize) usize {
    return (first_rank + k) % world;
}

/// glm-rotlayer-check's hook (null: off): each layer's residual rows after its FFN glue, hashed in glue order.
pub var capture: ?*Capture = null;

pub const Capture = struct {
    d: *const cuda.Driver,
    host: []u8, // one glue call's rows, read back
    sums: [cfg.max_layers]u64 = @splat(0),
    rows: [cfg.max_layers]usize = @splat(0),

    /// Rows lo..hi of residual `x` (bf16 rows of `row` bytes) after layer `j`'s FFN glue on `rank`, `stream` done.
    pub fn at(c: *Capture, stream: cuda.Stream, rank: usize, j: usize, x: u64, lo: usize, hi: usize, row: usize) !void {
        if (j >= cfg.max_layers or hi <= lo) return;
        const bytes = (hi - lo) * row;
        if (bytes > c.host.len) return error.CaptureTooSmall;
        try stream.synchronize();
        try c.d.check(c.d.api.cuMemcpyDtoH_v2(c.host.ptr, x + lo * row, bytes), "cuMemcpyDtoH");
        var h = std.hash.Wyhash.init(c.sums[j]);
        h.update(std.mem.asBytes(&[3]u64{ rank, lo, hi }));
        h.update(c.host[0..bytes]);
        c.sums[j] = h.final();
        c.rows[j] += hi - lo;
    }
};

test "TF_GLM_OVN_EXPERT_ROTATE: mod puts MoE layer i's first shares on rank i mod 3; dense layers and MTP keep 0" {
    const t = std.testing;
    const c = Config{};
    const o = try parse("mod", &c, 3);
    try t.expect(o.on);
    for (0..c.dense_layers) |i| try t.expectEqual(@as(usize, 0), o.first(i));
    for (c.dense_layers..c.layers) |i| try t.expectEqual(i % 3, o.first(i));
    try t.expectEqual(@as(usize, 0), o.first(c.layers)); // the MTP layer
    const n_mod = o.counts(&c, 3);
    try t.expectEqualSlices(usize, &.{ 14, 14, 14 }, n_mod[0..3]);
    try t.expectEqual(o.of, (try parse("1", &c, 3)).of);
}

test "TF_GLM_OVN_EXPERT_ROTATE: counts spread by a smooth round robin, every prefix within one layer of its share" {
    const t = std.testing;
    const c = Config{};
    const want = [3]usize{ 10, 16, 16 };
    const o = try parse("10,16,16", &c, 3);
    const got_counts = o.counts(&c, 3);
    try t.expectEqualSlices(usize, &want, got_counts[0..3]);
    var got: [3]usize = @splat(0);
    var k: usize = 0;
    const n = c.layers - c.dense_layers;
    for (c.dense_layers..c.layers) |i| {
        got[o.first(i)] += 1;
        k += 1;
        for (0..3) |r| { // |got - k C_r / n| < 1, scaled by n
            const ideal = k * want[r];
            const diff = if (got[r] * n > ideal) got[r] * n - ideal else ideal - got[r] * n;
            try t.expect(diff < n);
        }
    }
    // credits 10,16,16: rank 1 (the tie goes low), then 2 (32), then 0 (30 against 6 and 6)
    for ([_]usize{ 1, 2, 0 }, c.dense_layers..) |r, i| try t.expectEqual(r, o.first(i));
}

test "TF_GLM_OVN_EXPERT_ROTATE: off, refused values, and even splits that rotate nothing" {
    const t = std.testing;
    const c = Config{};
    for ([_]?[]const u8{ null, "0", "off", "OFF" }) |v| try t.expect(!(try parse(v, &c, 3)).on);
    for ([_][]const u8{ "x", "10,16", "10,16,16,0", "10,16,15", "-1,27,16", "", "mod2" }) |v| try t.expectError(error.BadExpertRotate, parse(v, &c, 3));
    try t.expect(!(try parse("mod", &c, 2)).on); // 2,048 columns over two or four ranks split evenly
    try t.expect(!(try parse("mod", &c, 4)).on);
    try t.expect(!(try parse("21,21", &c, 2)).on);
    var x = c;
    x.exl3 = true; // 16 blocks of 128: 6 / 5 / 5
    try t.expect((try parse("mod", &x, 3)).on);
    try t.expectEqual(@as(u32, 0), (Owners{}).digest());
    const a = (try parse("mod", &c, 3)).digest();
    try t.expect(a != 0 and a != refused);
    try t.expect(a != (try parse("10,16,16", &c, 3)).digest());
    try t.expectEqual(a, (try parse("MOD", &c, 3)).digest());
}

test "TF_GLM_OVN_EXPERT_ROTATE: each layer's ranks hold the standard shares once, the owner's the first" {
    const t = std.testing;
    var c = Config{};
    c.exl3 = true;
    const o = try parse("10,16,16", &c, 3);
    var plain: [3]shard.Share = undefined;
    var rot: [3]shard.Share = undefined;
    for (0..3) |r| {
        plain[r] = try shard.Share.of(&c, r, 3);
        rot[r] = plain[r];
        rot[r].rot = o;
    }
    for (0..c.layers + 1) |i| {
        var seen: [3]bool = @splat(false);
        for (0..3) |r| {
            const s = try rot[r].at(&c, i);
            const q = o.held(r, 3, i);
            try t.expectEqual(plain[q].moe, s.moe);
            try t.expectEqual(plain[q].shared, s.shared);
            try t.expect(!s.rot.on); // a layer's share is static
            try t.expect(!seen[q]);
            seen[q] = true;
            if (r == o.first(i)) try t.expectEqual(@as(usize, 0), q);
        }
    }
    // the widest share a rank holds: 768 / 704 wherever it owns a layer; the standard split's when off
    for (0..3) |r| try t.expectEqual([2]usize{ 768, 704 }, rot[r].widest(&c));
    try t.expectEqual([2]usize{ 640, 640 }, plain[2].widest(&c));
    try t.expectEqual(plain[1].moe, (try plain[1].at(&c, 7)).moe);
    // the admission bound (Share.part): rank 2 holds a 768-column share on its layers, so its bound rises to 3 / 8
    try t.expectEqual([2]u64{ 768, 2048 }, rot[2].part(&c));
    try t.expectEqual([2]u64{ 4096, 12288 }, plain[2].part(&c));
}

test "TF_GLM_OVN_EXPERT_ROTATE: partials summed from the owner's equal the standard split's sum bit for bit" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(0x0105);
    const rnd = prng.random();
    var mutant_differs = false;
    for (0..4096) |_| {
        var p: [3]f32 = undefined; // share s's partial, the same bits on whichever rank computes it
        for (&p) |*x| x.* = (rnd.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(rnd.intRangeAtMost(i32, -8, 8)));
        const today = (p[0] + p[1]) + p[2]; // ranks8 off the knob: rank 0 first
        for (0..3) |o| {
            var g: [3]f32 = undefined; // rank r's gathered slot holds share (r - o) mod 3
            for (0..3) |r| g[r] = p[(r + 3 - o) % 3];
            var acc = g[slot(0, o, 3)];
            for (1..3) |k| acc = acc + g[slot(k, o, 3)];
            try t.expectEqual(@as(u32, @bitCast(today)), @as(u32, @bitCast(acc)));
            const ranked = (g[0] + g[1]) + g[2]; // the mutant: rank order on rotated slots
            mutant_differs = mutant_differs or @as(u32, @bitCast(ranked)) != @as(u32, @bitCast(today));
        }
    }
    try t.expect(mutant_differs);
    try t.expectEqual(@as(usize, 3), word(3, 0)); // knob off: the launches carry today's int
    try t.expectEqual(@as(usize, 3), word(3, 2) & 255);
    try t.expectEqual(@as(usize, 2), word(3, 2) >> 8);
}
