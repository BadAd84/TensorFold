//! TF_GLM_OVN_DRAW1: a verify's rows, or every part's of a shared round, drawn with one exchange a kind and one sync.
//! Greedy rows: one tf_glm_top1_rows launch, one gather, the host's pick per row. Sampled rows (0 < top_k <= 1,024):
//! each rank's top k (tf_draw_cands) gathered once, set in -inf rows (tf_draw_place), drawn by tf_draw_rows.
const std = @import("std");
const cuda = @import("cuda");
const sampler = @import("cuda_sampler.zig");
const ranks = @import("cuda_ranks.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

/// sample.cu's candidate cap (tf_sample::C): a top_k past it draws from whole rows as before.
pub const max_k = 1024;
const rows_max = engine.max_shared_rows; // the host arrays' rows; the device buffers hold the load's (Draw1.rows)
const pair_bytes = 16;

/// A draw's per-row words as one upload: each sampled row's rule, its position less one, its logits row.
const Ctl = extern struct { rules: [rows_max]sampler.Rule, metas: [rows_max]i32, src: [rows_max]u32 };

/// Device bytes Draw1.init allocates for `rows` rows over `world` ranks (the rooms reserve them at load).
pub fn deviceBytes(world: usize, rows: usize, vocab: usize) u64 {
    return world * rows * pair_bytes + world * rows * max_k * 8 + rows * vocab * 2 + @sizeOf(Ctl) + rows * 4;
}

/// One part's rows: logits rows row0.. of the draw's buffer, tokens at first, first + 1, ...; null sampling: greedy.
pub const Part = struct { row0: usize, rows: usize, first: u64, sampling: ?sampler.Sampling, out: []u32 };

/// Whether a part's rule draws here: greedy, or a top_k inside the candidate cap.
pub fn takes(s: ?sampler.Sampling) bool {
    const x = s orelse return true;
    return x.top_k > 0 and x.top_k <= max_k;
}

pub const Draw1 = struct {
    top1_rows: cuda.Function,
    cands: cuda.Function,
    place: cuda.Function,
    draw_rows: cuda.Function,
    picks: cuda.DeviceBuffer, // greedy: [world][rows] (value bits, id, 0, 0)
    cand: cuda.DeviceBuffer, // sampled: [world][rows][k] (bf16 bits, id) word pairs
    full: cuda.DeviceBuffer, // bf16 [rows][vocab]: -inf but the gathered candidates
    ctl: cuda.DeviceBuffer, // Ctl
    tok: cuda.DeviceBuffer, // u32 [rows]
    rows: usize, // the rows the buffers hold: the shared buffers' (cuda_ranks.bufferRows)

    /// Made at the first verify with the knob on (cuda_engine_shared.ensure); glm_glue is the forward's module 6.
    pub fn init(e: *Engine) !Draw1 {
        const d = e.d;
        const world = e.share.world;
        const rows: usize = try ranks.bufferRows(e.settings);
        std.debug.assert(rows <= rows_max);
        var x: Draw1 = undefined;
        x.rows = rows;
        x.top1_rows = try e.m.mods[6].function("tf_glm_top1_rows");
        x.cands = try e.smp.module.function("tf_draw_cands");
        x.place = try e.smp.module.function("tf_draw_place");
        x.draw_rows = try e.smp.module.function("tf_draw_rows");
        x.picks = try cuda.DeviceBuffer.alloc(d, world * rows * pair_bytes);
        errdefer x.picks.free();
        try x.picks.fill8(0, null); // each pick's padding stays zero
        x.cand = try cuda.DeviceBuffer.alloc(d, world * rows * max_k * 8);
        errdefer x.cand.free();
        x.full = try cuda.DeviceBuffer.alloc(d, rows * e.c.vocab * 2);
        errdefer x.full.free();
        x.ctl = try cuda.DeviceBuffer.alloc(d, @sizeOf(Ctl));
        errdefer x.ctl.free();
        x.tok = try cuda.DeviceBuffer.alloc(d, rows * 4);
        return x;
    }

    pub fn deinit(x: *Draw1) void {
        x.tok.free();
        x.ctl.free();
        x.full.free();
        x.cand.free();
        x.picks.free();
    }
};

/// Every rank's `bytes` at base + rank * bytes into every rank's base, rank order (the greedy pick's gather).
fn gather(e: *Engine, base: u64, bytes: usize) !void {
    const mine = base + e.share.rank * bytes;
    if (e.group) |g| return g.allGather(e.stream, mine, base, bytes);
    try e.lib.check(e.lib.api.ncclAllGather(mine, base, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
}

/// `parts` (each `takes` its rule; no grammar) from `logits` (this rank's bf16 [rows][span]): every rank, same parts.
pub fn draw(e: *Engine, logits: u64, parts: []const Part) !void {
    const x = &e.draw1.?;
    const world = e.share.world;
    const span = e.share.vocab.len();
    var greedy_rows: usize = 0; // the top1 launch covers rows 0 .. the last greedy part's end
    var n: usize = 0; // sampled rows
    var k: u32 = 0;
    var ctl: Ctl = undefined;
    for (parts) |p| {
        if (p.rows == 0 or p.row0 + p.rows > x.rows or p.first < 1) return error.BadDraw;
        const s = p.sampling orelse {
            greedy_rows = @max(greedy_rows, p.row0 + p.rows);
            continue;
        };
        const r = sampler.rule(s);
        for (0..p.rows) |j| {
            ctl.rules[n] = r;
            ctl.metas[n] = @intCast(p.first - 1 + j); // tf_draw_rows keys row n at metas[n] + 1
            ctl.src[n] = @intCast(p.row0 + j);
            n += 1;
        }
        k = @max(k, s.top_k);
    }
    if (greedy_rows > 0) {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i64, @intCast(span)));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(@as(i64, @intCast(span))); // the rows' stride
        a.add(x.picks.ptr + e.share.rank * greedy_rows * pair_bytes);
        try cuda.launch.launch(x.top1_rows, .{ .grid = .{ .x = @intCast(greedy_rows) }, .block = .{ .x = 1024 } }, e.stream, &a);
        try gather(e, x.picks.ptr, greedy_rows * pair_bytes);
    }
    if (n > 0) {
        const slots = (k + 1) / 2 * 2; // even, so a rank's candidates are whole 16-byte packs for the RoCE gather
        const per = n * slots * 8; // one rank's candidates
        try x.ctl.uploadAsync(0, std.mem.asBytes(&ctl), e.stream.handle);
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(span)));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(x.ctl.ptr + @offsetOf(Ctl, "rules"));
        a.add(x.ctl.ptr + @offsetOf(Ctl, "src"));
        a.add(slots);
        a.add(x.cand.ptr + e.share.rank * per);
        try cuda.launch.launch(x.cands, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, e.stream, &a);
        try gather(e, x.cand.ptr, per);
        var b: cuda.Args = .{};
        b.add(x.cand.ptr);
        b.add(@as(u32, @intCast(world)));
        b.add(@as(u32, @intCast(n)));
        b.add(slots);
        b.add(x.full.ptr);
        b.add(@as(u32, @intCast(e.c.vocab)));
        try cuda.launch.launch(x.place, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, e.stream, &b);
        var c: cuda.Args = .{};
        c.add(x.full.ptr);
        c.add(@as(u32, @intCast(e.c.vocab)));
        c.add(x.ctl.ptr + @offsetOf(Ctl, "rules"));
        c.add(x.ctl.ptr + @offsetOf(Ctl, "metas"));
        c.add(x.tok.ptr);
        try cuda.launch.launch(x.draw_rows, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, e.stream, &c);
    }
    try e.stream.synchronize();
    try e.checkGathers(); // the picks and candidates are gathered rows
    var all: [ranks.max_world * rows_max][4]u32 = undefined; // Engine.init refuses a larger world
    if (greedy_rows > 0) try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, x.picks.ptr, world * greedy_rows * pair_bytes), "cuMemcpyDtoH");
    var tok: [rows_max]u32 = undefined;
    if (n > 0) try x.tok.download(0, std.mem.sliceAsBytes(tok[0..n]));
    var m: usize = 0;
    for (parts) |p| {
        if (p.sampling != null) {
            @memcpy(p.out[0..p.rows], tok[m..][0..p.rows]);
            m += p.rows;
            continue;
        }
        for (0..p.rows) |j| p.out[j] = pick(all[0 .. world * greedy_rows], greedy_rows, p.row0 + j);
    }
}

/// cuda_engine_pick.greedy's rule over the ranks' picks of `row` (rank q's at q * rows + row): the largest value,
/// then the lowest id, ranks in order.
fn pick(all: []const [4]u32, rows: usize, row: usize) u32 {
    var best = all[row];
    var q: usize = rows + row;
    while (q < all.len) : (q += rows) {
        const p = all[q];
        const v: f32 = @bitCast(p[0]);
        const b: f32 = @bitCast(best[0]);
        if (v > b or (v == b and p[1] < best[1])) best = p;
    }
    return best[1];
}

test "the ranks' greedy picks: largest value, then the lowest id, as cuda_engine_pick" {
    const f = struct {
        fn w(v: f32, id: u32) [4]u32 {
            return .{ @bitCast(v), id, 0, 0 };
        }
    }.w;
    // three ranks of two rows: row 0 ties at 2.0 (ids 9 and 4), row 1's best is rank 2's
    const all = [_][4]u32{ f(2.0, 9), f(1.0, 1), f(2.0, 4), f(0.5, 100), f(1.0, 7), f(3.0, 200) };
    try std.testing.expectEqual(@as(u32, 4), pick(&all, 2, 0));
    try std.testing.expectEqual(@as(u32, 200), pick(&all, 2, 1));
}

/// A recording driver: each allocation's bytes; functions and fills succeed without a device.
const Fake = struct {
    var sizes: [8]usize = undefined;
    var n: usize = 0;
    fn alloc(p: *cuda.abi.DevicePtr, len: usize) callconv(.c) cuda.abi.Result {
        sizes[n] = len;
        n += 1;
        p.* = 0x1000 * n;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn fill(_: cuda.abi.DevicePtr, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn function(f: *cuda.abi.Function, _: cuda.abi.Module, _: [*:0]const u8) callconv(.c) cuda.abi.Result {
        f.* = @ptrFromInt(0x40);
        return cuda.abi.success;
    }
};

test "TF_GLM_OVN_DRAW1: its buffers hold the load's TF_GLM_OVN_SHARED_ROWS rows; a part past them is refused" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    d.api.cuMemsetD8_v2 = Fake.fill;
    d.api.cuModuleGetFunction = Fake.function;
    var e: Engine = undefined;
    e.d = &d;
    e.share.world = 3;
    e.share.vocab = .{ .lo = 0, .hi = 51627 };
    e.c.vocab = 154880;
    e.m.mods[6] = .{ .d = &d, .handle = null };
    e.smp.module = .{ .d = &d, .handle = null };
    var out: [8]u32 = undefined;
    for ([_]u32{ 32, 48, 64 }) |rows| {
        e.settings = .{ .shared_rows = rows };
        Fake.n = 0;
        e.draw1 = try Draw1.init(&e);
        defer e.draw1.?.deinit();
        try std.testing.expectEqual(@as(usize, rows), e.draw1.?.rows);
        const want = [_]usize{ 3 * rows * pair_bytes, 3 * rows * max_k * 8, rows * 154880 * 2, @sizeOf(Ctl), rows * 4 };
        try std.testing.expectEqualSlices(usize, &want, Fake.sizes[0..Fake.n]);
        var sum: usize = 0;
        for (Fake.sizes[0..Fake.n]) |n| sum += n;
        try std.testing.expectEqual(@as(u64, sum), deviceBytes(3, rows, 154880)); // what the rooms reserve
        // a part ending one row past the load's rows is refused before any launch
        try std.testing.expectError(error.BadDraw, draw(&e, 0, &.{.{ .row0 = rows - 4, .rows = 5, .first = 1, .sampling = null, .out = &out }}));
    }
}
