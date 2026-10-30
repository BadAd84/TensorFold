//! TF_GLM_OVN_SEG_DSA: a shared decode window's DSA steps, one launch each for every segment (glm_glue_seg.cuh,
//! glm_ops_seg.cuh; production's dsa_segments, families/glm5_next/cuda/forward.py:783-831), from a table of each
//! segment's rows, position and every layer's caches filled once a forward; a block runs its segment's launch body.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const window_rows = state.window_rows;

/// The most segments a table holds (Engine.max_shared).
pub const max_segs = 16;

/// Buffers.seg: a shared window's segment tables (TF_GLM_OVN_SEG_KDA, _SEG_DSA), the engine's; null when off.
pub const Tables = struct { kda: ?*@import("cuda_kda_seg.zig").SegKda = null, dsa: ?*SegDsa = null };

/// glm_seg_table.cuh's Seg: rows, device position, dense chunks, pool bucket, score blocks, first score.
const Seg = extern struct { row0: i64, rows: i64, pos: u64, nch: i64, np: i64, nb: i64, soff: i64, pad: i64 = 0 };

/// glm_seg_table.cuh's Cache: one segment's dsa_block.Caches at one layer and its latent rows a page part.
const Cache = extern struct { lc: u64, ls: u64, ik: u64, ig: u64, pk: u64, pt: u64, ps: i64, page_rows: i64 };

/// The segment lists the launches read: by pass, head block (16 or 32 heads a CTA) and select width.
const List = enum(u8) { dense, dense16, dense32, sparse, sparse16, sparse32, select_narrow, select_wide };
const n_lists = @typeInfo(List).@"enum".field_names.len;

/// The table's head: segments, then the lists (i32 segment indices); [layers][segments]Cache follows at caches_at.
const Head = extern struct { seg: [max_segs]Seg, lists: [n_lists][max_segs]i32 };
const caches_at = @sizeOf(Head);

/// A segment as the caller states it: its rows of the window, host and device position, every DSA layer's caches.
pub const Part = struct { row0: usize, rows: usize, pos: usize, pos_dev: u64, caches: []const dsa_block.Caches };

const Fns = struct {
    lat_write: cuda.Function,
    lat_write_fp8: cuda.Function,
    index_write: cuda.Function,
    pool_keys: cuda.Function,
    chunks: [2][2]cuda.Function, // [32-head blocks][E4M3 cache]
    merge: cuda.Function,
    scores: cuda.Function,
    select: [2]cuda.Function, // [1,024 threads]
    tokens: cuda.Function,

    fn load(g: cuda.Module, ops: cuda.Module) !Fns {
        const f: Fns = .{
            .lat_write = try g.function("tf_glm_lat_write_seg"),
            .lat_write_fp8 = try g.function("tf_glm_lat_write_fp8_seg"),
            .index_write = try g.function("tf_glm_index_write_seg"),
            .pool_keys = try g.function(if (sparse.pk8) "tf_glm_pool_keys_seg_pk8" else "tf_glm_pool_keys_seg"),
            .chunks = .{
                .{ try g.function("tf_glm_chunks_16_seg"), try g.function("tf_glm_chunks_16_fp8_seg") },
                .{ try g.function("tf_glm_chunks_32_seg"), try g.function("tf_glm_chunks_32_fp8_seg") },
            },
            .merge = try g.function("tf_glm_merge_seg"),
            .scores = try g.function(if (sparse.pk8) "tf_glm_scores_seg_pk8" else "tf_glm_scores_seg"),
            .select = .{ try g.function("tf_glm_select_rows_seg"), try g.function("tf_glm_select_rows_wide_seg") },
            .tokens = try ops.function("tf_glm_dsa_tokens_seg"),
        };
        for (f.chunks[0]) |x| try x.allowDynamicShared(glue.chunksSmem(latent.heads_tile));
        for (f.chunks[1]) |x| try x.allowDynamicShared(glue.chunksSmem(latent.heads_tile_wide));
        return f;
    }
};

pub const SegDsa = struct {
    f: Fns,
    gpa: std.mem.Allocator,
    l0: dsa_block.Layer, // the DSA layers' shapes the table's lists were made for (takes checks each layer's)
    layers: usize,
    nch: usize, // Buffers.nch: a dense call's most chunks
    pn: usize, // chunk slots a row's partials hold: the dense most or a sparse call's
    slots: usize, // rows x pn x heads
    table: cuda.DeviceBuffer, // Head, then [layers][segs]Cache
    part: cuda.DeviceBuffer, // po fp32 [slots][lw], pm and pl fp32 [slots]: a segment's from row0 x pn x heads
    scores: cuda.DeviceBuffer, // fp32: the sparse segments' [rows][np] score rows back to back
    caches: []Cache, // the host's [layers][segs] before its upload
    head: Head = std.mem.zeroes(Head),
    parts: [max_segs]Part = undefined,
    counts: [n_lists]usize = @splat(0),
    segs: usize = 0, // this forward's segments (0: none, the per-segment loop runs)
    dense_nch: usize = 0, // the dense segments' most chunks
    nb_max: usize = 0, // the sparse segments' most score blocks
    select: bool = false, // the sparse segments' scores fit `scores` (else each selects alone)

    /// Made at load with the knob on for the engine's shared buffers (`rows`, `nch`) and sequences of `capacity`.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: cuda.Module, ops: cuda.Module, w: *const weights.Weights, rows: usize, nch: usize, capacity: usize) !SegDsa {
        const l0 = for (w.layers) |l| {
            if (l.mixer == .mla) break l.mixer.mla;
        } else return error.NoDsaLayer;
        return initShape(gpa, d, g, ops, l0, state.mlaLayers(w), rows, nch, capacity);
    }

    /// As init, from one DSA layer's shapes and the layer count (the GPU check's).
    pub fn initShape(gpa: std.mem.Allocator, d: *const cuda.Driver, g: cuda.Module, ops: cuda.Module, l0: dsa_block.Layer, layers: usize, rows: usize, nch: usize, capacity: usize) !SegDsa {
        const pn = @max(nch, latent.chunksFor(sparse.width));
        const slots = rows * pn * l0.heads;
        const f = try Fns.load(g, ops);
        const caches = try gpa.alloc(Cache, layers * max_segs);
        errdefer gpa.free(caches);
        var table = try cuda.DeviceBuffer.alloc(d, caches_at + caches.len * @sizeOf(Cache));
        errdefer table.free();
        var part = try cuda.DeviceBuffer.alloc(d, slots * (l0.lw * 4 + 8));
        errdefer part.free();
        const scores = try cuda.DeviceBuffer.alloc(d, rows * (capacity / sparse.pool) * 4);
        return .{ .f = f, .gpa = gpa, .l0 = l0, .layers = layers, .nch = nch, .pn = pn, .slots = slots, .table = table, .part = part, .scores = scores, .caches = caches };
    }

    pub fn deinit(x: *SegDsa) void {
        x.scores.free();
        x.part.free();
        x.table.free();
        x.gpa.free(x.caches);
    }

    /// shared.begin (after each sequence's ready, which may move its index rings): the window's segments.
    pub fn fill(x: *SegDsa, segs: []const fw.Seg, stream: cuda.Stream) !void {
        x.segs = 0;
        if (segs.len < 2 or segs.len > max_segs) return;
        var parts: [max_segs]Part = undefined;
        for (segs, parts[0..segs.len]) |g, *p| p.* = .{ .row0 = g.row0, .rows = g.rows, .pos = g.st.pos, .pos_dev = g.st.pos_dev, .caches = g.st.caches };
        try x.load(parts[0..segs.len], stream);
    }

    /// The table for `parts` (in row order), uploaded on `stream`; a mix one launch cannot take (a part past
    /// window_rows rows, a layer's caches in two formats) leaves segs 0, and the per-segment loop runs.
    pub fn load(x: *SegDsa, parts: []const Part, stream: cuda.Stream) !void {
        x.segs = 0;
        const n = parts.len;
        if (n < 2 or n > max_segs) return;
        const l = x.l0;
        for (parts) |p| if (p.rows == 0 or p.rows > window_rows or p.caches.len != x.layers) return;
        for (0..x.layers) |j| {
            const fp8 = parts[0].caches[j].ls != 0;
            for (parts, 0..) |p, i| {
                const c = p.caches[j];
                if ((c.ls != 0) != fp8) return;
                if (c.ps % latentRow(l, c) != 0) return error.LatentShape; // as dsa_block.attend
                x.caches[j * n + i] = .{ .lc = c.lc, .ls = c.ls, .ik = c.ik, .ig = c.ig, .pk = c.pk, .pt = c.pt, .ps = @intCast(c.ps), .page_rows = @intCast(c.ps / latentRow(l, c)) };
            }
        }
        x.counts = @splat(0);
        x.dense_nch = 0;
        x.nb_max = 0;
        var soff: usize = 0;
        for (parts, 0..) |p, i| {
            const np = sparse.poolBucket(p.pos, p.rows, p.caches[0].pk_rows - 2); // selectTokens' np_max
            const nb = (sparse.scoreColumns(p.pos, p.rows, np) + 63) / 64;
            const nch = @min(latent.chunksFor(p.pos + p.rows), x.nch); // shared.mixer's
            x.head.seg[i] = .{ .row0 = @intCast(p.row0), .rows = @intCast(p.rows), .pos = p.pos_dev, .nch = @intCast(nch), .np = @intCast(np), .nb = @intCast(nb), .soff = @intCast(soff) };
            const wide = latent.headBlock(p.rows, l.heads) == latent.heads_tile_wide;
            if (p.pos < l.dense_limit) {
                x.add(.dense, i);
                x.add(if (wide) .dense32 else .dense16, i);
                x.dense_nch = @max(x.dense_nch, nch);
            }
            if (dsa_block.sparseRows(l, p.pos, p.rows)) {
                x.add(.sparse, i);
                x.add(if (wide) .sparse32 else .sparse16, i);
                x.add(if (sparse.wideSelect(p.rows, np)) .select_wide else .select_narrow, i);
                x.nb_max = @max(x.nb_max, nb);
                soff += p.rows * np;
            }
        }
        x.select = soff * 4 <= x.scores.len;
        try x.table.uploadAsync(0, std.mem.asBytes(&x.head), stream.handle);
        try x.table.uploadAsync(caches_at, std.mem.sliceAsBytes(x.caches[0 .. x.layers * n]), stream.handle);
        @memcpy(x.parts[0..n], parts);
        x.segs = n;
    }

    fn add(x: *SegDsa, list: List, i: usize) void {
        const k = @backingInt(list);
        x.head.lists[k][x.counts[k]] = @intCast(i);
        x.counts[k] += 1;
    }

    fn listOf(x: *const SegDsa, list: List) []const i32 {
        return x.head.lists[@backingInt(list)][0..x.counts[@backingInt(list)]];
    }

    fn listPtr(x: *const SegDsa, list: List) u64 {
        return x.table.ptr + @offsetOf(Head, "lists") + @as(u64, @backingInt(list)) * max_segs * 4;
    }

    /// Whether DSA layer `di` of a window of `n` segments takes the table: this forward's, the table's layer shapes,
    /// a decode window, and TF_GLM_OVN_MLA_FUSEMERGE off (its skip and fused chunk forms keep the per-segment loop).
    pub fn takes(x: *const SegDsa, n: usize, set: glue.Set, l: dsa_block.Layer, di: usize) bool {
        const same = l.heads == x.l0.heads and l.lw == x.l0.lw and l.dense_limit == x.l0.dense_limit;
        return x.segs > 1 and x.segs == n and di < x.layers and same and l.lw == 512 and l.index.d == 128 and l.index.heads <= 32 and !set.prompt and set.mla.fuse_rows == 0 and !set.mla.skip;
    }

    /// dsa_block.attend for every segment of layer `di` (s: the window's scratch, every row prepared): each step's
    /// launch covers every segment, in attend's order, and each segment's rows land where attend writes them.
    pub fn attend(x: *const SegDsa, k: dsa_block.Kernels, stream: cuda.Stream, l: dsa_block.Layer, di: usize, s: dsa_block.Scratch) !void {
        const ix = l.index;
        const n = x.segs;
        const segs = x.table.ptr;
        const cache = x.table.ptr + caches_at + di * n * @sizeOf(Cache);
        const fp8 = x.parts[0].caches[di].ls != 0;
        const lw: i64 = @intCast(l.lw);
        if (fp8) try go(x.f.lat_write_fp8, stream, .{ window_rows, n, 1 }, 512, 0, .{ segs, cache, s.lat, lw }) else try go(x.f.lat_write, stream, .{ window_rows, n, 1 }, 128, 0, .{ segs, cache, s.lat, lw, lw });
        try go(x.f.index_write, stream, .{ window_rows, n, 1 }, 32, 0, .{ segs, cache, s.ikr, @as(i64, @intCast(ix.d + ix.heads)), s.igr, ix.ln_w, ix.ln_b, @as(f32, 1e-6) });
        try go(x.f.pool_keys, stream, .{ window_rows / 4 + 2, n, 1 }, 32, 0, .{ segs, cache, ix.ape });
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
        if (x.counts[@backingInt(List.dense)] > 0) {
            try x.chunks(stream, l, cache, s.qa, 0, 0, 0, scale, fp8, false, x.dense_nch);
            try x.merge(stream, l, .dense, s.ol, s.dummy, 0, false);
        }
        if (x.counts[@backingInt(List.sparse)] == 0) return;
        try x.selectTokens(k, stream, l, di, cache, s);
        const nch = latent.chunksFor(sparse.width);
        try x.chunks(stream, l, cache, s.qa, s.tokens, s.counts, sparse.width, scale, fp8, true, nch);
        try x.merge(stream, l, .sparse, s.ol, s.counts, nch, true);
    }

    /// latent's chunk launches for the dense or sparse segments, one a head block size, `nch` chunks a segment.
    fn chunks(x: *const SegDsa, stream: cuda.Stream, l: dsa_block.Layer, cache: u64, qa: u64, tok: u64, cnt: u64, w: usize, scale: f32, fp8: bool, sp: bool, nch: usize) !void {
        const lists = if (sp) [2]List{ .sparse16, .sparse32 } else [2]List{ .dense16, .dense32 };
        const hbs = [2]usize{ latent.heads_tile, latent.heads_tile_wide };
        const smem = [2]u32{ glue.chunksSmem(latent.heads_tile), glue.chunksSmem(latent.heads_tile_wide) };
        for (lists, hbs, smem, 0..) |list, hb, bytes, wide| {
            const count = x.counts[@backingInt(list)];
            if (count == 0) continue;
            const args = .{ x.table.ptr, x.listPtr(list), cache, qa, tok, cnt, x.part.ptr, x.part.ptr + x.slots * l.lw * 4, x.part.ptr + x.slots * (l.lw * 4 + 4), @as(i64, @intCast(l.heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sp)), @as(i64, @intCast(x.pn)), @as(c_int, @intCast(nch)) };
            try go(x.f.chunks[wide][@intFromBool(fp8)], stream, .{ window_rows, (l.heads + hb - 1) / hb, count * nch }, 256, bytes, args);
        }
    }

    /// latent's merge for the listed segments: dense ones over their own chunks, sparse ones over `nch`.
    fn merge(x: *const SegDsa, stream: cuda.Stream, l: dsa_block.Layer, list: List, out: u64, cnt: u64, nch: usize, sp: bool) !void {
        const po = x.part.ptr;
        const args = .{ x.table.ptr, x.listPtr(list), po, po + x.slots * l.lw * 4, po + x.slots * (l.lw * 4 + 4), out, cnt, @as(i64, @intCast(l.heads)), @as(i64, @intCast(x.pn)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sp)) };
        try go(x.f.merge, stream, .{ window_rows, l.heads, x.counts[@backingInt(list)] }, 128, 0, args);
    }

    /// sparse.selectTokens for the sparse segments, each's tokens and counts at its own rows: one scores, select and
    /// tokens launch each, or (a segment the CUDA indexer takes, or scores past the buffer) each segment alone.
    fn selectTokens(x: *const SegDsa, k: dsa_block.Kernels, stream: cuda.Stream, l: dsa_block.Layer, di: usize, cache: u64, s: dsa_block.Scratch) !void {
        const ix = l.index;
        const kw_rows = ix.d + ix.heads;
        const list = x.listOf(.sparse);
        var alone = !x.select;
        if (k.set.index) |q| {
            for (list) |i| {
                const p = x.parts[@intCast(i)];
                alone = alone or q.takes(p.rows, @intCast(x.head.seg[@intCast(i)].np), p.caches[di].pk, p.caches[di].ps);
            }
        }
        if (alone) {
            for (list) |i| { // dsa_block.attend's call, its output at the segment's rows
                const p = x.parts[@intCast(i)];
                const c = p.caches[di];
                const t: latent.Table = .{ .pt = c.pt, .ps = c.ps, .page_rows = c.ps / latentRow(l, c) };
                try sparse.selectTokensProfile(k.set, k.ops, stream, s.qi + p.row0 * ix.heads * ix.d * 2, s.ikr + p.row0 * kw_rows * 2 + ix.d * 2, kw_rows, c.pk, t, p.pos_dev, p.pos, p.rows, ix.heads, ix.d, c.pk_rows - 2, s.select, s.tokens + p.row0 * sparse.width * 4, s.counts + p.row0 * 4, k.split, k.prof);
            }
            return;
        }
        const dscale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(ix.d), -0.5));
        try go(x.f.scores, stream, .{ window_rows, x.nb_max, list.len }, 128, 0, .{ x.table.ptr, x.listPtr(.sparse), cache, s.qi, s.ikr + ix.d * 2, @as(i64, @intCast(kw_rows)), x.scores.ptr, dscale, sparse.headScale(ix.heads), @as(i64, @intCast(ix.heads)) });
        for ([2]List{ .select_narrow, .select_wide }, [2]u32{ 128, 1024 }, 0..) |sel, threads, wide| {
            const count = x.counts[@backingInt(sel)];
            if (count == 0) continue;
            try go(x.f.select[wide], stream, .{ window_rows, count, 1 }, threads, 0, .{ x.table.ptr, x.listPtr(sel), x.scores.ptr, s.select.pools, @as(c_int, sparse.topk_pools) });
        }
        try go(x.f.tokens, stream, .{ window_rows, list.len, 1 }, 256, 0, .{ x.table.ptr, x.listPtr(.sparse), cache, s.select.pools, @as(c_int, sparse.topk_pools), s.tokens, s.counts });
    }
};

/// A latent row's bytes in cache `c`: E4M3 codes (scales apart) or BF16 (dsa_block.attend's `row`).
fn latentRow(l: dsa_block.Layer, c: dsa_block.Caches) u64 {
    return if (c.ls != 0) l.lw else l.lw * 2;
}

fn go(f: cuda.Function, stream: cuda.Stream, grid: [3]usize, threads: u32, shared: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = shared }, stream, &a);
}

comptime {
    std.debug.assert(@sizeOf(Seg) == 64 and @sizeOf(Cache) == 64);
    std.debug.assert(caches_at == max_segs * 64 + n_lists * max_segs * 4);
}
