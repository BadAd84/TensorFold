//! GLM-5.3-Flash's DSA block on CUDA, a rank's latent-path share (the Python family's dsa_block) to an fp32 partial.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const onepass = @import("cuda_onepass.zig");
const prof = @import("cuda_prof.zig");
pub const index_shard = @import("cuda_index_shard.zig");

/// A prompt chunk's rows one dense latent attention call takes (geometry.MLA_PROMPT_ATT_ROWS).
pub const prompt_att_rows = 512;

/// The indexer's weights, replicated: [wk | weights_proj] [d + heads][dims], wq_b (as stored), gate, LayerNorm, bias.
pub const Index = struct { kw: projection.Proj, gate: u64, ln_w: u64, ln_b: u64, ape: u64, qb: projection.Proj, heads: usize, d: usize };

/// One DSA layer on a rank of `heads` heads: proj, norms, q_b, kv_b per head as wk/wv, o, indexer; shapes from config.
pub const Layer = struct {
    proj: projection.Proj, // [q_a | kv_a], q_b and o as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    q_norm: u64,
    kv_norm: u64,
    q_b: projection.Proj,
    wk: u64,
    wv: u64,
    sk: u64 = 0, // TF_GLM_OVN_KVB8: wk's fp32 scales [heads][qk][lw / 128] (wk then E4M3); 0: wk bf16
    sv: u64 = 0, // and wv's [heads][dv][lw / 128]
    o: projection.Proj,
    index: Index,
    heads: usize,
    qk: usize,
    dv: usize,
    lw: usize,
    q_lora: usize,
    dims: usize,
    eps: f32,
    dense_limit: usize, // the largest context whose top-k selection keeps every visible key (Config.keyWidth)
};

/// The caches: latent bf16 [capacity][lw], index keys/gates bf16 [capacity][128], pooled keys [pk_rows] of
/// sparse.pooledRow bytes (bf16 [128], or TF_GLM_OVN_PK8's 128 E4M3 codes and an FP32 scale).
pub const Caches = struct {
    lc: u64,
    ls: u64 = 0,
    ik: u64,
    ig: u64,
    pk: u64,
    pk_rows: usize,
    pt: u64 = 0, // on pages: the stream's table (i32 page ids), lc and pk then this cache's place in page 0; 0: rows
    ps: u64 = 0, // on pages: bytes from one page to the next
};

/// The block's scratch for up to `rows` rows.
pub const Scratch = struct {
    dp: u64, // bf16 [rows][q_lora + lw]
    qr: u64, // bf16 [rows][q_lora]
    xs_qr: u64, // fp32 [rows][q_lora / 64]
    lat: u64, // bf16 [rows][lw]
    xs_lat: u64, // fp32 [rows][lw / 64]
    q: u64, // bf16 [rows][heads qk]
    ikr: u64, // bf16 [rows][160]
    igr: u64, // fp32 [rows][128]
    router_part: u64, // fp32 [router_ks][rows][128]
    qa: u64, // bf16 [rows][heads][lw]
    ol: u64, // bf16 [rows][heads][lw]
    vn: u64, // bf16 [rows][heads dv]
    qi: u64, // bf16 [rows][index heads d]
    partials: latent.Partials,
    dummy: u64, // an i32
    select: sparse.Scratch,
    tokens: u64, // i32 [rows][sparse.width]
    counts: u64, // i32 [rows]
    mm_part: u64, // fp32 split-K partials
    at: u64, // an i32: a prompt block's first position (pos + its first row)
};

/// The kernels: the captured Triton set and glm_ops (the token lists and position offsets); `shard`: a decode window's
/// indexer pools across the ranks (TF_GLM_OVN_DECODE_SHARD, index_shard.of).
pub const Kernels = struct { set: glue.Set, ops: cuda.Module, prof: ?*prof.Prof = null, split: ?sparse.Split = null, shard: ?index_shard.Shard = null };

/// forward.dsa_block (latent path) of `rows` rows at pos to an fp32 partial; `nch`: 512-key chunks, `part_rows`: block.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize, out: u64) !void {
    try body(k, stream, l, c, s, x, x_stride, rows, pos, host_pos, nch, part_rows);
    try fill(k.set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the DSA block: o_proj of rows lo..hi (s.vn) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const w = l.heads * l.dv;
    try projection.matmul(set, stream, l.o, s.vn + lo * w * 2, w, out, s.mm_part, hi - lo, l.dims, w, true);
}

/// The block up to its out_proj (s.vn holds the expanded rows).
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    try prepareProfile(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof);
    try rest(k, stream, l, c, s, rows, pos, host_pos, nch, part_rows);
}

/// body once its prepare ran on every row (prepareRows, TF_GLM_OVN_FRONTS): attention, then the rows in s.vn.
pub fn rest(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    try attend(k, stream, l, c, s, 0, rows, pos, host_pos, nch, part_rows);
    const stamp = prof.span(k.prof, stream, .projections);
    defer stamp.end();
    try latent.expandV(k.set, stream, s.ol, l.wv, l.sv, s.vn, rows, l.heads, l.dv, l.lw);
}

/// Whether a sequence at `host_pos` has rows past the dense limit among its next `rows` (the sparse pass).
pub fn sparseRows(l: Layer, host_pos: usize, rows: usize) bool {
    return host_pos + rows - 1 >= l.dense_limit;
}

/// The block's row-by-row part for a shared window: projections, norms, q_b, indexer, absorbed query, `qi`'s query.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    return prepareProfile(set, stream, l, s, x, x_stride, rows, qi, null);
}

pub fn prepareProfile(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool, profile: ?*prof.Prof) !void {
    return prepareRows(set, stream, l, s, x, x_stride, 0, rows, qi, profile);
}

/// prepareProfile on rows lo..hi, each output row at its own row; `qi` as the whole chunk's (sparseRows of every row).
pub fn prepareRows(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, qi: bool, profile: ?*prof.Prof) !void {
    const ix = l.index;
    const width = l.q_lora + l.lw;
    const rows = hi - lo;
    const xr = x + lo * x_stride * 2;
    var v = s; // the row buffers from row lo (the split-K and router partials hold only this call's rows)
    v.dp += lo * width * 2;
    v.qr += lo * l.q_lora * 2;
    v.xs_qr += lo * (l.q_lora / 64) * 4;
    v.lat += lo * l.lw * 2;
    v.xs_lat += lo * (l.lw / 64) * 4;
    v.q += lo * l.heads * l.qk * 2;
    v.ikr += lo * (ix.d + ix.heads) * 2;
    v.igr += lo * ix.d * 4;
    v.qa += lo * l.heads * l.lw * 2;
    v.qi += lo * ix.heads * ix.d * 2;
    var stamp = prof.span(profile, stream, .replicated_qakva);
    try projection.matmul(set, stream, l.proj, xr, x_stride, v.dp, v.mm_part, rows, width, l.dims, false);
    try glue.rmsnorm(set, stream, v.dp, width, l.q_norm, v.qr, l.q_lora, v.xs_qr, l.eps, rows, l.q_lora);
    try glue.rmsnorm(set, stream, v.dp + l.q_lora * 2, width, l.kv_norm, v.lat, l.lw, v.xs_lat, l.eps, rows, l.lw);
    stamp.end();
    stamp = prof.span(profile, stream, .projections);
    defer stamp.end();
    try projection.matmul(set, stream, l.q_b, v.qr, l.q_lora, v.q, v.mm_part, rows, l.heads * l.qk, l.q_lora, false);
    try projection.matmul(set, stream, ix.kw, xr, x_stride, v.ikr, v.mm_part, rows, ix.d + ix.heads, l.dims, false);
    try glue.router(set, stream, xr, x_stride, ix.gate, v.igr, v.router_part, rows, l.dims, ix.d);
    try latent.absorbQ(set, stream, v.q, l.wk, l.sk, v.qa, rows, l.heads, l.qk, l.lw);
    if (qi) try projection.matmul(set, stream, ix.qb, v.qr, l.q_lora, v.qi, v.mm_part, rows, ix.heads * ix.d, l.q_lora, false);
}

/// One sequence's rows r0..r0 + rows at pos: latent and index keys written, dense then sparse attention into s.ol.
pub fn attend(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    var v = s; // the scratch from row r0
    v.lat += r0 * l.lw * 2;
    v.ikr += r0 * kw_rows * 2;
    v.igr += r0 * ix.d * 4;
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    const t = try table(l, c);
    var stamp = prof.span(k.prof, stream, .cache_write);
    try latent.latentWriteFormat(set, stream, v.lat, l.lw, c.lc, c.ls, t, pos, rows, l.lw);
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(set, stream, v.ikr, kw_rows, v.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, t, pos, rows);
    stamp.end();
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    if (host_pos < l.dense_limit) {
        stamp = prof.span(k.prof, stream, .mla);
        try denseAttention(k, stream, l, c, v, pos, rows, nch, part_rows, scale);
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        const sharded = if (k.shard) |x| index_shard.takes(x, host_pos, rows) else false; // the same on every rank
        if (sharded) try index_shard.select(set, k.ops, stream, k.shard.?, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, t, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, k.prof) else try sparse.selectTokensProfile(set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, t, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, k.split, k.prof);
        stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        // TF_GLM_OVN_ONEPASS_PIPE: the one-pass pipeline with the default's bits, bit-identical to the chunk programs
        if (set.onepass_pipe) |fp| {
            if (onepass.takesExact(set.prompt, c.ls, l.heads, l.lw, rows)) return onepass.runPipe(fp, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.ol, rows, l.heads, scale);
        }
        // TF_GLM_OVN_SPARSE_ONEPASS: prompt rows in one pass a row (production's kernel and bits), no partials or merge
        if (k.set.onepass) |f| {
            if (onepass.takes(set.prompt, c.ls, l.heads, l.lw)) return onepass.run(f, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.ol, rows, l.heads, scale);
        }
        try latent.sparseAttentionProfile(set, k.prof, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.partials, v.ol, rows, l.heads, l.lw, scale);
    }
}

/// A prompt MTP chunk's block: every row's caches, then the last row's attention and o_proj partial into `out` row 0.
pub fn blockLastRow(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, out: u64) !void {
    if (rows == 0) return error.BadWindow;
    const t = try table(l, c);
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    try prepareProfile(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof); // every row keeps its query, its latent, its keys
    var stamp = prof.span(k.prof, stream, .cache_write);
    try latent.latentWriteFormat(k.set, stream, s.lat, l.lw, c.lc, c.ls, t, pos, rows, l.lw); // all rows, at pos, before any of them attends
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(k.set, stream, s.ikr, kw_rows, s.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, t, pos, rows);
    stamp.end();
    try attendLastRow(k, stream, l, c, s, rows, pos, host_pos, nch, t);
    const pstamp = prof.span(k.prof, stream, .projections);
    defer pstamp.end();
    const off = (rows - 1) * l.heads * l.lw * 2;
    try latent.expandV(k.set, stream, s.ol + off, l.wv, l.sv, s.vn, 1, l.heads, l.dv, l.lw); // its o, into vn's row 0
    try fill(k.set, stream, l, s, 0, 1, out);
}

/// A chunk's last row attending over caches already written: its own query, causal limit and token list.
fn attendLastRow(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64, host_pos: usize, nch: usize, t: latent.Table) !void {
    const ix = l.index;
    const r0 = rows - 1;
    var v = s; // the scratch from the last row (its token lists and partials stay at the front)
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    var at = pos; // the last row's own position on the device: pos + r0, so its causal limit is unchanged
    if (r0 > 0) {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(@as(c_int, @intCast(r0)));
        a.add(v.at);
        try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
        at = v.at;
    }
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    if (host_pos < l.dense_limit) { // the same keys today's whole-chunk call read for this row
        var stamp = prof.span(k.prof, stream, .mla);
        try latent.attentionFormat(k.set, stream, v.qa, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, v.partials, v.ol, v.dummy, 1, l.heads, l.lw, nch, scale, latent.headBlock(rows, l.heads));
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        // one row at host_pos + r0: the same pool bucket, score columns and token list its block computed for it
        try sparse.selectTokensProfile(k.set, k.ops, stream, v.qi, s.ikr + r0 * (ix.d + ix.heads) * 2 + ix.d * 2, ix.d + ix.heads, c.pk, t, at, host_pos + r0, 1, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, null, k.prof);
        var stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        if (k.set.onepass) |f| { // TF_GLM_OVN_SPARSE_ONEPASS: the kernel the whole chunk's row would have taken
            if (onepass.takes(k.set.prompt, c.ls, l.heads, l.lw)) return onepass.run(f, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.ol, 1, l.heads, scale);
        }
        try latent.sparseAttentionFormat(k.set, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.partials, v.ol, 1, l.heads, l.lw, scale);
    }
}

/// The caches' page table: pages whole latent rows apart, a row E4M3 codes (scales apart) or BF16.
fn table(l: Layer, c: Caches) !latent.Table {
    const row = if (c.ls != 0) l.lw else l.lw * 2;
    if (c.ps % row != 0) return error.LatentShape; // the token lists' unit
    return .{ .pt = c.pt, .ps = c.ps, .page_rows = c.ps / row };
}

/// forward.dense_attention: latent.attention in `part_rows`-row blocks, each at its first row's position.
fn denseAttention(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, pos: u64, rows: usize, nch: usize, part_rows: usize, scale: f32) !void {
    const hb = latent.headBlock(rows, l.heads);
    var r0: usize = 0;
    while (r0 < rows) : (r0 += part_rows) {
        const n = @min(part_rows, rows - r0);
        var at = pos;
        if (r0 > 0) {
            var a: cuda.Args = .{};
            a.add(pos);
            a.add(@as(c_int, @intCast(r0)));
            a.add(s.at);
            try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
            at = s.at;
        }
        const off = r0 * l.heads * l.lw * 2;
        try latent.attentionProfile(k.set, k.prof, stream, s.qa + off, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, s.partials, s.ol + off, s.dummy, n, l.heads, l.lw, nch, scale, hb);
    }
}
