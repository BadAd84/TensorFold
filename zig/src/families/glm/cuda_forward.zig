//! GLM-5.3-Flash's TP forward on CUDA (the Python family's forward.compute and commit), partials over NCCL or copied.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const projection = @import("cuda_proj.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const experts = @import("cuda_experts.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const taps_mod = @import("cuda_taps.zig");
const state = @import("cuda_state.zig");
const mm_mod = @import("cuda_mm.zig");
const prof_mod = @import("cuda_prof.zig");
const images_mod = @import("cuda_images.zig");
const exl3 = @import("cuda_exl3.zig");
const l2pf_mod = @import("cuda_l2pf.zig");
const l2pace = @import("cuda_l2pace.zig");
const shared = @import("cuda_shared.zig");
const index_mod = @import("cuda_index.zig");
const moe_side = @import("cuda_moe_side.zig");
const hc_glue = @import("cuda_forward_glue.zig");
const onepass_mod = @import("cuda_onepass.zig");
const kda_chunk = @import("cuda_kda_chunk.zig");
const slices = @import("cuda_forward_slices.zig");
const rotlayer = @import("cuda_rotlayer.zig");

pub const prefill_rows = state.prefill_rows;
pub const window_rows = state.window_rows;
pub const State = state.State;
pub const Buffers = state.Buffers;

/// The modules a forward launches from: our fatbins.
pub const Modules = struct {
    glue: glue.Kernels,
    mods: [8]cuda.Module,
    kda: kda.Kernels,
    lane: cuda.qmmf.Lane, // the NVFP4 linears' decode rows (qlinear)
    prompt: cuda.nvfp4.Prompt, // and their prompt rows
    experts: experts.Kernels,
    exl3: exl3.Kernels, // an EXL3 checkpoint's routed experts
    mm: mm_mod.Kernels, // glm_mm: the bf16 linears on tiled weights
    index: ?index_mod.Kernels, // TF_GLM_INDEX_CUDA: a prompt chunk's indexer scores and pools (glm_index)
    pools32: bool = false, // TF_GLM_OVN_POOLS32: a split prompt indexer gathers its picks as int32
    shard_from: usize = 0, // TF_GLM_OVN_DECODE_SHARD: decode windows from this position shard their indexer pools
    onepass: ?onepass_mod.Kernels, // TF_GLM_OVN_SPARSE_ONEPASS: prompt rows' sparse attention in one pass (glm_onepass)
    onepass_pipe: ?onepass_mod.PipeKernels, // TF_GLM_OVN_ONEPASS_PIPE: the one-pass pipeline with bit-exact switches
    chunked: ?kda_chunk.Kernel, // TF_GLM_OVN_KDA_CHUNKED: prompt chunks' KDA chain in chunked (WY) form
    tiled: bool = false, // eligible bf16 linears are tiled; KDA inputs with uneven strides retain their row-major path
    packs: ?*const mm_mod.Packed = null, // the linears packed to 4, 6 or 8 bits (TF_GLM_DENSE_BITS, _HEAD_BITS)
    copy_rows: cuda.Function,
    put_slot: cuda.Function,
    pack_dense: cuda.Function,
    quant: quant.Kernels,
    q4: q4.Kernels,
    side: ?moe_side.Side, // TF_GLM_OVN_SHARED_SIDE: the stream a decode window's shared experts fork onto
    moe_ovn: moe.Ovn, // TF_GLM_OVN_COMBINE_SY, _TOPK_SUM, _SHARED_FIRST: the MoE block's knobs, read once here
    glue_row: u64, // TF_GLM_OVN_GLUE_ROW=N: a prompt piece's glue of N rows or more in one kernel (0: off)
    pdl_early: u64, // TF_GLM_OVN_PDL_EARLY: 1 the decode windows' glue launches dependent-ready, 2 prompt chunks too
    mla: glue.Mla, // TF_GLM_OVN_MLA_FUSEMERGE, TF_GLM_OVN_ABSORB_VEC: the MLA launch choices, read at load
    l2pace: ?l2pace.Paced, // TF_GLM_OVN_L2PF_PACED: #1580's paced prefetch, opened by the engine after the weights tile

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        const k = cuda.kernels;
        var m: Modules = undefined;
        var n: usize = 0;
        errdefer for (m.mods[0..n]) |*x| x.unload();
        for (&m.mods, [_][]const u8{ k.glm_kda, k.glm_ops, k.experts, k.affine4_pack, k.glm_quant, k.qmm_group, k.glm_glue, k.glm_mm }) |*x, image| {
            x.* = try cuda.Module.load(d, image);
            n += 1;
        }
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer m.prompt.unload();
        m.kda = try kda.Kernels.load(m.mods[0]);
        m.experts = try experts.Kernels.load(d, m.mods[2], @intCast(try ctx.attribute(.multiprocessor_count)));
        errdefer m.experts.unload();
        m.copy_rows = try m.mods[1].function("tf_glm_copy_rows");
        m.put_slot = try m.mods[1].function("tf_glm_put_slot");
        m.pack_dense = try m.mods[3].function("tf_pack_dense");
        m.quant = try quant.Kernels.load(m.mods[4]);
        m.q4 = try q4.Kernels.load(m.mods[5], m.mods[3], ctx);
        m.glue = try glue.Kernels.load(m.mods[6]);
        m.exl3 = try exl3.Kernels.load(d);
        errdefer m.exl3.unload();
        m.mm = try mm_mod.Kernels.load(m.mods[7]);
        m.chunked = if (kda_chunk.enabled()) try kda_chunk.Kernel.load(d) else null;
        errdefer if (m.chunked) |*x| x.unload();
        m.index = null;
        m.mla = try glue.Mla.read();
        errdefer if (m.index) |*x| x.unload();
        const mode = index_mod.Mode.read() catch return error.BadIndexSetting;
        _ = index_mod.splitRows() catch return error.BadIndexSetting; // TF_GLM_OVN_DECODE_INDEX: a typo is refused
        if (mode != .off) m.index = index_mod.Kernels.load(d, k.glm_index, mode) catch |err| blk: {
            std.log.warn("TF_GLM_INDEX_CUDA: the indexer kernels are not used ({t}); prompt chunks use tf_glm_scores", .{err});
            break :blk null;
        };
        m.glue_row = try cuda.ovn.int("GLUE_ROW", 0);
        m.pdl_early = try cuda.ovn.int("PDL_EARLY", 0);
        m.moe_ovn = moe.Ovn.read();
        m.onepass = onepass_mod.open(d, k.glm_onepass, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer if (m.onepass) |*x| x.unload();
        m.onepass_pipe = onepass_mod.openPipe(d, k.glm_onepass_pipe, @intCast(try ctx.attribute(.compute_capability_major)));
        m.pools32 = cuda.ovn.on("POOLS32");
        m.shard_from = try dsa_block.index_shard.from(); // a typo is refused
        m.tiled = false;
        m.packs = null;
        m.l2pace = null; // m starts undefined, so the field defaults do not apply
        m.side = try moe_side.Side.init(d); // last: nothing after it fails
        return m;
    }

    pub fn tools(m: *const Modules) weights.Tools {
        return .{ .pack_dense = m.pack_dense, .lane = &m.lane, .quant = &m.quant, .q4 = &m.q4 };
    }
    pub fn ops(m: *const Modules) cuda.Module {
        return m.mods[1];
    }

    pub fn deinit(m: *Modules) void {
        if (m.side) |*x| x.deinit();
        if (m.l2pace) |*p| p.deinit();
        if (m.chunked) |*x| x.unload();
        if (m.index) |*x| x.unload();
        if (m.onepass) |*x| x.unload();
        if (m.onepass_pipe) |*x| x.unload();
        m.exl3.unload();
        for (&m.mods) |*x| x.unload();
        m.lane.unload();
        m.prompt.unload();
        m.experts.unload();
    }
};

/// overlap.ROW_STEP and PIECES_MAX: piece boundaries on the BF16 prompt matmul's 128-row blocks, at most 16 pieces.
pub const row_step = 128;
pub const max_pieces = 16;

/// overlap.ranges: [lo, hi) row pieces of a chunk of `rows`, about equal on row_step boundaries (fewer when short).
pub fn ranges(rows: usize, pieces: usize, out: *[max_pieces][2]usize) usize {
    const per = (rows + pieces - 1) / pieces;
    const size = (per + row_step - 1) / row_step * row_step;
    var n: usize = 0;
    var lo: usize = 0;
    while (lo < rows) : (lo += size) {
        out[n] = .{ lo, @min(rows, lo + size) };
        n += 1;
    }
    return n;
}

/// A prompt chunk's pieced exchanges (overlap.Overlap): second stream and events; `count`: TF_GLM_OVERLAP_PIECES (3).
pub const Pieces = struct {
    count: usize = 4,
    side: cuda.Stream,
    filled: [max_pieces]cuda.Event,
    gathered: [max_pieces]cuda.Event,
    glues: [max_pieces]cuda.Event, // split: a piece's own rows glued
    shared: [max_pieces]cuda.Event, // split: every rank's glued rows of a piece in
    owns_side: bool = true, // a paired step's second chunk (initTwin) runs on the first's side stream
    peer_first: usize = 0, // TF_GLM_OVN_PEER_FIRST: a split piece's peers' rows fill first, from this many rows a share
    fronts: usize = 0, // TF_GLM_OVN_FRONTS: the next block's front runs in each glue, from this many rows a call

    /// With `pair` (TF_GLM_PROMPT_PAIR) the side stream runs at the highest priority (TF_GLM_OVN_SIDE_PRIO=0: normal).
    pub fn init(d: *const cuda.Driver, count: usize, pair: bool) !Pieces {
        if (count == 0 or count > max_pieces) return error.BadPieces;
        const prio = try cuda.ovn.int("SIDE_PRIO", 1); // 0: a paired side stream at normal priority; unset: the highest
        const high = pair and prio != 0;
        const peer_first = try cuda.ovn.int("PEER_FIRST", 0);
        const fronts = try cuda.ovn.int("FRONTS", 0);
        var p = try make(d, count, if (high) try cuda.Stream.initPriority(d, .highest) else try cuda.Stream.init(d, false), true);
        p.peer_first = @intCast(peer_first);
        p.fronts = @intCast(fronts);
        return p;
    }

    /// A paired step's second chunk on `of`'s side stream: every rank posts both chunks' exchanges in one order.
    pub fn initTwin(d: *const cuda.Driver, of: *const Pieces) !Pieces {
        var p = try make(d, of.count, of.side, false);
        p.peer_first = of.peer_first;
        p.fronts = of.fronts;
        return p;
    }

    fn make(d: *const cuda.Driver, count: usize, side: cuda.Stream, owns_side: bool) !Pieces {
        var p: Pieces = .{ .count = count, .side = side, .filled = undefined, .gathered = undefined, .glues = undefined, .shared = undefined, .owns_side = owns_side };
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.* = try cuda.Event.init(d, false);
            b.* = try cuda.Event.init(d, false);
            g.* = try cuda.Event.init(d, false);
            s.* = try cuda.Event.init(d, false);
        }
        return p;
    }

    pub fn deinit(p: *Pieces) void {
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.deinit();
            b.deinit();
            g.deinit();
            s.deinit();
        }
        if (p.owns_side) p.side.deinit();
        p.* = undefined;
    }
};

/// How the ranks' partials meet: every rank in-process copying into the others' slots, or one rank over NCCL.
pub const Exchange = union(enum) {
    local,
    // group: the main stream's all-gathers through it (RoCE writes for small ones on two ranks); side streams stay raw
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*const cuda.tp_group.Group = null },
};

/// One rank of the forward: its weights, its sequence's caches, the window's buffers.
pub const Rank = struct {
    w: *const weights.Weights,
    st: *State,
    b: *Buffers,
    /// A shared window (Buffers.initShared): each stream's rows in row order; empty: every row on `st`.
    segs: []const Seg = &.{},
};

/// The engine's image rows and the sequence the forward runs on (they splice into that sequence's rows only).
pub const ImageRows = struct { v: *images_mod.Images, slot: u32 };

/// One stream's rows of a shared window: rows row0..row0 + rows, on sequence `st` from its position.
pub const Seg = struct { st: *State, row0: usize, rows: usize };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *const Modules,
    stream: cuda.Stream,
    ranks: []const Rank, // every rank of a local check, or this process's one
    exchange: Exchange,
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE (the Python family's default): decode windows on KDA's three-kernel chain
    pieces: ?*const Pieces = null, // a prompt chunk's exchanges in row pieces on a second stream (null: whole)
    split: bool = true, // TF_GLM_PREFILL_REDUCE=split (the default with send/recv): each rank glues its own rows
    ring: bool = false, // TF_GLM_PREFILL_REDUCE=ring: split, its exchanges passed along the ring (cuda_relay)
    prx: bool = false, // TF_GLM_OVN_PRX takes this pass's split exchanges (cuda_forward_prx.takes): paired, or PRX_ALONE
    split_min: usize = 0, // TF_GLM_SPLIT_MIN: a piece of fewer rows glues whole (its exchanges cost more than they save)
    images: ?ImageRows = null, // a prompt pass's image rows over their placeholders' embeddings (its sequence only)
    prof: ?*prof_mod.Prof = null, // TF_GLM_PROF=1: events between the sections
    l2pf: ?*l2pf_mod.Prefetch = null, // TF_GLM_L2PF: a decode window's weight prefetch beside the stream
    taps: ?taps_mod.Taps = null, // DFlash2 (--drafter): the tapped layers' rows, after each such layer
    index_split: bool = false, // TF_GLM_INDEX_SPLIT: this prompt chunk's indexer rows split across the ranks (rank 0's choice)
    side: ?*const moe_side.Side = null, // TF_GLM_OVN_SHARED_SIDE: this window's shared experts beside its routed ones
    one: ?*@import("cuda_one_chunk.zig").Capture = null, // TF_GLM_OVN_ONE_CHUNK: this prompt chunk's KDA cuts

    /// TF_GLM_OVN_L2PF_PACED on this forward: a decode window of one rank with at most the knob's streams in it.
    pub fn paced(f: *const Forward) ?*const l2pace.Paced {
        return l2pace.Paced.gate(&f.m.l2pace, f.ranks.len, f.ranks[0].b.prefill, f.ranks[0].segs.len);
    }

    pub fn set(f: *const Forward) glue.Set {
        const pdl = f.m.pdl_early >= @as(u64, if (f.ranks[0].b.prefill) 2 else 1);
        return .{ .glue = &f.m.glue, .index = if (f.m.index) |*x| x else null, .mm = if (f.m.tiled) &f.m.mm else null, .packs = f.m.packs, .prompt = f.ranks[0].b.prefill, .linear = .{ .lane = &f.m.lane }, .copy_rows = f.m.copy_rows, .pdl_early = pdl, .mla = f.m.mla, .onepass = if (f.m.onepass) |x| x.f else null, .onepass_pipe = if (f.m.onepass_pipe) |x| x.exact else null };
    }

    /// reduce.sums' gather: every rank's partial of `rows` rows into every rank's gathered buffer, rank order.
    fn gather(f: *const Forward, rows: usize) !void {
        const stamp = prof_mod.span(f.prof, f.stream, .all_gather);
        defer stamp.end();
        const c = &f.ranks[0].w.config;
        const bytes = rows * c.hidden * 4;
        switch (f.exchange) {
            .local => for (f.ranks) |to| for (f.ranks) |from| {
                if (to.w.share.rank == from.w.share.rank) continue;
                const off = from.w.share.rank * bytes;
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(to.b.gathered + off, from.b.gathered + off, bytes, f.stream.handle), "cuMemcpyDtoDAsync");
            },
            .nccl => |n| {
                const r = f.ranks[0];
                if (n.group) |g| return g.allGather(f.stream, r.b.gathered + r.w.share.rank * bytes, r.b.gathered, bytes);
                const count = rows * c.hidden;
                try n.lib.check(n.lib.api.ncclAllGather(r.b.gathered + r.w.share.rank * bytes, r.b.gathered, count, .f32, n.comm, f.stream.handle), "ncclAllGather");
            },
        }
    }

    fn slot(r: Rank, rows: usize) u64 {
        return r.b.gathered + r.w.share.rank * rows * r.w.config.hidden * 4;
    }

    /// forward._pre: hc_pre of every row at a site into b.normed, xs, post and comb.
    pub fn pre(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, rows: usize) !void {
        const stamp = prof_mod.span(f.prof, f.stream, .glue);
        defer stamp.end();
        const c = &r.w.config;
        const b = r.b;
        try glue.hcPre(f.set(), f.stream, b.x, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc);
    }

    /// `first`: the rank whose partial the sum takes first (rotlayer.first: a rotated MoE layer's owner, else 0).
    pub fn post(f: *const Forward, r: Rank, x: u64, rows: usize, first: usize) !void {
        const stamp = prof_mod.span(f.prof, f.stream, .glue);
        defer stamp.end();
        const c = &r.w.config;
        try glue.hcPost(f.set(), f.stream, x, x, r.b.gathered, r.b.post, r.b.comb, rows, c.hidden, c.hc, rotlayer.word(r.w.share.world, first));
    }

    /// forward._mixer: the layer's KDA or DSA block on b.normed; `li`/`di`: its index among loaded KDA / DSA layers.
    pub fn mixer(f: *const Forward, r: Rank, l: weights.Layer, li: usize, di: usize, rows: usize, nch: usize, out: ?u64) !void {
        return f.mixerFrom(r, l, li, di, rows, nch, out, false);
    }

    /// mixer; `fronted` (a prompt chunk, TF_GLM_OVN_FRONTS): the block's front already ran on every row, its rest runs.
    pub fn mixerFrom(f: *const Forward, r: Rank, l: weights.Layer, li: usize, di: usize, rows: usize, nch: usize, out: ?u64, fronted: bool) !void {
        const c = &r.w.config;
        const b = r.b;
        const st = r.st;
        switch (l.mixer) {
            .kda => |k| {
                if (b.prefill) if (f.one) |cap| return @import("cuda_one_chunk.zig").kdaBlock(f, r, k, li, rows, out, fronted, cap);
                const cur = st.cur[li];
                var s = b.kda;
                s.p = if (b.prefill) b.kproj else st.proj + li * st.proj_layer * 2;
                s.saves.state_out = st.recAt(1 - cur, li);
                if (!b.prefill) { // a window's saved rows stay with the layer until its commit
                    const kv = li * window_rows * st.heads * kda.dk;
                    s.saves.k = st.save_k + kv * 4;
                    s.saves.v = st.save_v + kv * 2;
                    s.saves.g = st.save_g + kv * 4;
                    s.saves.b = st.save_b + li * window_rows * st.heads * 4;
                }
                const conv = st.conv + li * 3 * st.conv_c * 2;
                const chunked = if (b.prefill) kda_chunk.prompt(&f.m.chunked, st.pos) else null; // TF_GLM_OVN_KDA_CHUNKED
                if (fronted) try kda_block.restProfile(f.set(), &f.m.kda, f.stream, k, s, rows, conv, st.recAt(cur, li), !b.prefill and f.kda_wide, b.prefill, f.prof, chunked) else try kda_block.bodyProfile(f.set(), &f.m.kda, f.stream, k, s, b.normed, c.hidden, rows, conv, st.recAt(cur, li), !b.prefill and f.kda_wide, b.prefill, f.prof, if (f.paced()) |p| .{ .l2 = p, .layer = li + di } else null, chunked);
                if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
                if (b.prefill) st.cur[li] = 1 - cur; // a prompt chunk keeps every row: the layer commits now
            },
            .mla => |a| {
                const ks: dsa_block.Kernels = .{ .set = f.set(), .ops = f.m.ops(), .prof = f.prof, .split = f.indexSplit(r), .shard = dsa_block.index_shard.of(f.m.shard_from, b.prefill, f.exchange, r.w.share.rank, r.w.share.world) };
                if (fronted) try dsa_block.rest(ks, f.stream, a, st.caches[di], b.dsa, rows, st.pos_dev, st.pos, nch, b.part_rows) else try dsa_block.body(ks, f.stream, a, st.caches[di], b.dsa, b.normed, c.hidden, rows, st.pos_dev, st.pos, nch, b.part_rows);
                if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
            },
        }
    }

    /// A prompt chunk's indexer rows split across the ranks (index_split) over NCCL; in-process ranks pick every row.
    fn indexSplit(f: *const Forward, r: Rank) ?sparse.Split {
        if (!f.index_split or !r.b.prefill) return null;
        return switch (f.exchange) {
            .local => null,
            .nccl => |x| .{ .lib = x.lib, .comm = x.comm, .rank = r.w.share.rank, .world = r.w.share.world, .int32 = f.m.pools32 },
        };
    }

    /// forward._ffn: the dense MLP or MoE block on b.normed into this rank's slot.
    pub fn ffn(f: *const Forward, r: Rank, l: weights.Layer, rows: usize, out: ?u64) !void {
        const stamp = prof_mod.span(f.prof, f.stream, .ffn);
        defer stamp.end();
        const c = &r.w.config;
        const b = r.b;
        switch (l.ffn) {
            .dense => |m| {
                try mlp.body(f.mlpKernels(), f.stream, m, b.dense, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, 0, rows, b.prefill, o);
            },
            .moe => |m| {
                if (f.side) |sd| if (out) |o| return moe_side.block(f.moeKernels(), f.stream, sd, m, b.moe, b.normed, c.hidden, rows, o);
                try moe.run(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, b.prefill, out);
            },
        }
    }

    pub fn mlpKernels(f: *const Forward) mlp.Kernels {
        return .{ .set = f.set(), .linear = .{ .lane = &f.m.lane, .nvfp4_prompt = &f.m.prompt }, .copy_rows = f.m.copy_rows };
    }

    pub fn moeKernels(f: *const Forward) moe.Kernels {
        return .{ .set = f.set(), .experts = &f.m.experts, .exl3 = &f.m.exl3, .put_slot = f.m.put_slot, .prof = f.prof, .ovn = f.m.moe_ovn };
    }

    pub const Site = enum { mixer, ffn };

    /// forward.out_proj's fill of rows lo..hi at a site of layer `l` into `out` (this rank's rows of a piece).
    pub fn fillRows(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize, out: u64) !void {
        const stamp = prof_mod.span(f.prof, f.stream, .output_projection);
        defer stamp.end();
        const c = &r.w.config;
        const b = r.b;
        switch (site) {
            .mixer => switch (l.mixer) {
                .kda => |k| try kda_block.fill(f.set(), f.stream, k, b.kda, lo, hi, out),
                .mla => |a| try dsa_block.fill(f.set(), f.stream, a, b.dsa, lo, hi, out),
            },
            .ffn => switch (l.ffn) {
                .dense => |m| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, lo, hi, b.prefill, out),
                .moe => |m| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, lo, hi, b.prefill, out),
            },
        }
    }

    /// Piece [lo, hi)'s room in a rank's gathered buffer (pieces never share bytes): [world][hi - lo][dims] fp32.
    pub fn room(r: Rank, lo: usize) u64 {
        return r.b.gathered + r.w.share.world * lo * r.w.config.hidden * 4;
    }

    /// reduce.shares: [lo, hi) of the rows of a piece of `n` each rank owns: about equal, the first n % world one more.
    pub fn share(n: usize, world: usize, rank: usize) [2]usize {
        return cuda.rank_scatter.share(n, world, rank);
    }

    /// Whether piece [lo, lo + n) splits (TF_GLM_PREFILL_REDUCE=split): a row for every rank and TF_GLM_SPLIT_MIN rows.
    pub fn splits(f: *const Forward, n: usize) bool {
        return f.split and n >= @max(f.ranks[0].w.share.world, f.split_min);
    }

    pub fn copy(f: *const Forward, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }

    const exchangePieces = @import("cuda_forward_pieces.zig").exchangePieces;

    pub const glueRows = hc_glue.glueRows;

    const pieced = @import("cuda_forward_pieces.zig").pieced;

    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;

    /// forward.forward of `ids`: logits in b.logits; a prompt chunk's KDA layers committed, a window's at `commit`.
    pub fn run(f: *const Forward, ids: []const i32) !void {
        const R = ids.len;
        try slices.enter(f, ids);
        var cut: [max_pieces][2]usize = undefined;
        if (slices.cuts(f, R, &cut)) |pieces| try f.pieced(f.pieces.?, R, cut[0..pieces]) else try f.whole(R);
        try f.finish(R);
        if (f.prof) |p| p.mark(f.stream, .head);
    }

    /// The unpieced loop over the layers (and the stream mean): each site's partials whole, then the glue.
    fn whole(f: *const Forward, R: usize) !void {
        const c = &f.ranks[0].w.config;
        const fuse = hc_glue.fuses(f, R); // TF_GLM_OVN_GLUE_FUSE: each hc_post with the next site's hc_pre, one kernel
        var alt = false; // the residual in b.x_alt (each fused glue writes the other buffer)
        var li: usize = 0;
        var di: usize = 0;
        const pf = f.paced(); // TF_GLM_OVN_L2PF_PACED
        for (f.ranks[0].w.layers, 0..) |_, j| {
            prof_mod.context(f.prof, j, null, .mixer);
            try f.prefetch(j, 2); // L2PF site `o`: the mixer's output projection while it runs
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                if (!fuse or j == 0) try f.pre(r, l.attn_hc, l.in_norm, R);
                if (r.segs.len > 0) try shared.mixer(f, r, l, li, di, R, slot(r, R)) else try f.mixer(r, l, li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), slot(r, R));
            }
            if (f.prof) |p| p.mark(f.stream, if (f.ranks[0].w.layers[j].mixer == .kda) .kda else .mla);
            try f.prefetch(j, 0); // L2PF site `a`: this layer's FFN across the mixer's all-gather
            if (pf) |p| try p.fork(f.stream, j, .f); // paced site f: this layer's router or dense gate/up across gather_a
            try f.gather(R);
            if (f.prof) |p| p.mark(f.stream, .gather_a);
            prof_mod.context(f.prof, j, null, .ffn);
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                if (fuse) {
                    try hc_glue.fused(f, r, alt, l.ffn_hc, l.post_norm, R, 0);
                } else {
                    try f.post(r, r.b.x, R, 0);
                    try f.pre(r, l.ffn_hc, l.post_norm, R);
                }
                if (f.prof) |p| p.mark(f.stream, .glue); // the FFN-side HC glue and norm, apart from the router after it
                try f.ffn(r, l, R, slot(r, R));
            }
            alt = alt != fuse;
            if (f.prof) |p| p.mark(f.stream, if (f.ranks[0].w.layers[j].ffn == .dense) .dense else .shared); // MoE: its fill
            try f.prefetch(j, 1); // L2PF site `f`: the next layer's mixer across the FFN's all-gather
            if (pf) |p| try p.fork(f.stream, j, .n); // paced site n: the next layer's input projections across gather_b
            try f.gather(R);
            const next = fuse and j + 1 < f.ranks[0].w.layers.len; // fused: the next layer's attention-site hc_pre too
            for (f.ranks) |r| if (next) try hc_glue.fused(f, r, alt, r.w.layers[j + 1].attn_hc, r.w.layers[j + 1].in_norm, R, rotlayer.first(&r.w.share, j)) else try f.post(r, hc_glue.xAt(r, alt), R, rotlayer.first(&r.w.share, j));
            alt = alt != next;
            if (f.taps) |t| try t.at(f.stream, j, hc_glue.xAt(f.ranks[0], alt), 0, R);
            if (rotlayer.capture) |cap| for (f.ranks) |r| try cap.at(f.stream, r.w.share.rank, j, hc_glue.xAt(r, alt), 0, R, c.hc * c.hidden * 2);
            if (f.prof) |p| p.mark(f.stream, .gather_b);
            switch (f.ranks[0].w.layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
        if (f.l2pf) |p| try p.join(f.stream); // the side stream's forks rejoin before the window ends
        for (f.ranks) |r| try glue.streamMean(f.set(), f.stream, hc_glue.xAt(r, alt), r.b.hidden, R, c.hidden, c.hc);
        if (pf) |p| try p.join(f.stream); // every paced site rejoins before the window ends
    }

    /// TF_GLM_L2PF site `which` of layer `j` (a decode window's weight prefetch; off for prompt chunks).
    fn prefetch(f: *const Forward, j: usize, which: usize) !void {
        const p = f.l2pf orelse return;
        if (f.ranks[0].b.prefill) return;
        try p.site(f.stream, @intCast(j), which);
    }

    /// The final norm and this rank's rows of the head (a prompt buffer's last row only).
    pub fn finish(f: *const Forward, R: usize) !void {
        prof_mod.context(f.prof, null, null, .chunk);
        const stamp = prof_mod.span(f.prof, f.stream, .head);
        defer stamp.end();
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            const b = r.b;
            try glue.rmsnormBy(f.set(), f.stream, b.hidden, c.hidden, r.w.norm, b.fnormed, c.hidden, b.fxs, c.eps, R, c.hidden);
            if (b.prefill and r.segs.len > 0) {
                try shared.heads(f, r);
                continue;
            }
            const head_rows: usize = if (b.prefill) 1 else R;
            const first = if (b.prefill) R - 1 else 0;
            try projection.matmul(f.set(), f.stream, r.w.head, b.fnormed + first * c.hidden * 2, c.hidden, b.logits, b.kda.mm_part, head_rows, r.w.share.vocab.len(), c.hidden, false);
        }
    }

    /// mtp.mtp_forward: caches from mtp_len, drafts in b.logits; a prompt chunk past its caches runs its last row.
    pub fn mtp(f: *const Forward, ids: []const i32, hidden: []const u64) !void {
        const n = ids.len;
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        // every rank shares this buffer and this row count, so every rank takes the same path and the gathers match
        const last_row = f.ranks[0].b.prefill and n > 1;
        const tail = if (last_row) (n - 1) * D * 2 else 0; // b.mx's last row, or its front
        const rows: usize = if (last_row) 1 else n; // the pass past the caches works on that row alone
        for (f.ranks, hidden) |r, h| {
            if (r.w.mtp == null) return error.NoMtpHead;
            if (n == 0 or n > r.b.rows) return error.BadWindow;
            try r.st.readyMtp(f.stream, n);
            try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
            if (h != r.b.hin) try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(r.b.hin, h, n * D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.embedB16(f.set(), f.stream, b.ids, r.w.embed, b.me, n, D, 1);
            // the head reads each row's next token: an image's rows one place on
            if (f.images) |im| try im.v.splice(f.stream, im.slot, r.st.mtp_len + 1, r.st.mtp_len + 1 + n, b.me, 1);
            if (r.st.mtp_len == 0) try f.d.check(f.d.api.cuMemsetD8Async(b.me, 0, D * 2, f.stream.handle), "cuMemsetD8Async"); // position 0's embedding is zeroed
            try glue.rmsnorm(f.set(), f.stream, b.me, D, m.enorm, b.mcat, 2 * D, null, c.eps, n, D);
            try glue.rmsnorm(f.set(), f.stream, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, c.eps, n, D);
            try b16.matmul(f.set(), f.stream, b.mcat, 2 * D, m.eh, b.mx, b.kda.mm_part, n, D, 2 * D, false);
            try glue.rmsnormBy(f.set(), f.stream, b.mx, D, m.layer.in_norm, b.normed, D, b.xs, c.eps, n, D);
            const nch = @min(latent.chunksFor(r.st.mtp_len + n), b.nch);
            const k: dsa_block.Kernels = .{ .set = f.set(), .ops = f.m.ops(), .shard = dsa_block.index_shard.of(f.m.shard_from, b.prefill, f.exchange, r.w.share.rank, r.w.share.world) };
            if (last_row) { // all n rows' caches, one row's attention, its partial in the slot's row 0
                try dsa_block.blockLastRow(k, f.stream, m.layer.mixer.mla, r.st.mtp_caches, b.dsa, b.normed, D, n, r.st.mtp_pos_dev, r.st.mtp_len, nch, slot(r, 1));
            } else {
                try dsa_block.block(k, f.stream, m.layer.mixer.mla, r.st.mtp_caches, b.dsa, b.normed, D, n, r.st.mtp_pos_dev, r.st.mtp_len, nch, b.part_rows, slot(r, n));
            }
        }
        try f.gather(rows);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx + tail, b.mx + tail, b.gathered, rows, D, r.w.share.world);
            try glue.rmsnormBy(f.set(), f.stream, b.mx + tail, D, m.layer.post_norm, b.normed, D, b.xs, c.eps, rows, D);
            try f.ffn(r, m.layer, rows, slot(r, rows));
        }
        try f.gather(rows);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx + tail, b.mx + tail, b.gathered, rows, D, r.w.share.world);
            try glue.rmsnorm(f.set(), f.stream, b.mx + (n - 1) * D * 2, D, m.norm, b.fnormed, D, b.fxs, c.eps, 1, D);
            if (!b.prefill) try q4.matmul(&f.m.q4, f.stream, b.fnormed, D, b.fxs, m.draft, b.logits, 1);
        }
    }

    /// forward.commit: keep the first `keep` rows; window KDA layers take new states (replayed if fewer), conv shifted.
    pub fn commit(f: *const Forward, keep: usize) !void {
        for (f.ranks) |r| {
            const st = r.st;
            if (keep == 0 or keep > st.pending or (st.prompt and keep != st.pending)) return error.BadCommit;
            if (!st.prompt and st.kda_layers > 0) {
                const cur = st.cur[0];
                for (st.cur) |x| if (x != cur) return error.KdaLayersOutOfStep;
                if (keep < st.pending) {
                    const h = st.heads;
                    var a: cuda.Args = .{};
                    a.add(@as(c_int, @intCast(h)));
                    a.add(st.recAt(cur, 0));
                    a.add(@as(usize, st.rec_layer / 4));
                    for ([_]u64{ st.save_k, st.save_v, st.save_g, st.save_b }) |x| a.add(x);
                    a.add(@as(usize, window_rows * h * kda.dk));
                    a.add(@as(usize, window_rows * h));
                    a.add(@as(c_int, @intCast(keep)));
                    a.add(st.recAt(1 - cur, 0));
                    try cuda.launch.launch(f.m.kda.replay_layers, .{ .grid = .{ .x = @intCast(st.kda_layers * h) }, .block = .{ .x = 1024 } }, f.stream, &a);
                }
                for (st.cur) |*x| x.* = 1 - x.*;
                if (st.kda_layers > 0) try glue.convShift(f.set(), f.stream, st.conv, st.proj, keep, st.proj_layer, st.proj_row, st.kda_layers, st.conv_c);
            }
            st.pending = 0;
            try st.setPos(f.stream, st.pos + keep);
        }
    }
};
