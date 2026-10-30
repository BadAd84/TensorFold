//! The hyper-connection glue between sites: a prompt piece's (glueRows) and TF_GLM_OVN_GLUE_FUSE's decode glue.
const glue = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
const prof = @import("cuda_prof.zig");
const fw = @import("cuda_forward.zig");
const rotlayer = @import("cuda_rotlayer.zig");
const Forward = fw.Forward;
const Rank = fw.Rank;

/// TF_GLM_OVN_GLUE_FUSE=N: a decode window of at most N rows (hc_partial8's, under 64) glues with hcGlue.
pub fn fuses(f: *const Forward, rows: usize) bool {
    const b = f.ranks[0].b;
    return !b.prefill and b.x_alt != 0 and rows <= b.fuse_rows and rows < 64;
}

/// The residual's buffer: b.x, or b.x_alt after an odd number of fused glues.
pub fn xAt(r: Rank, alt: bool) u64 {
    return if (alt) r.b.x_alt else r.b.x;
}

/// hc_post of the gathered partials from xAt(alt) into the other buffer, then the next site's hc_pre: one kernel.
/// `first`: the rank whose partial the sum takes first (rotlayer.first: a rotated MoE layer's owner, else 0).
pub fn fused(f: *const Forward, r: Rank, alt: bool, hc: weights.Hc, norm: glue.Norm, rows: usize, first: usize) !void {
    const stamp = prof.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    if (b.x_alt == 0) return error.NoGlueBuffer; // every rank's buffers take the same knob
    try glue.hcGlue(f.set(), f.stream, xAt(r, alt), xAt(r, !alt), b.gathered, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, b.hc_count, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc, rotlayer.word(r.w.share.world, first));
}

/// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
pub fn glueRows(f: *const Forward, r: Rank, j: usize, site: Forward.Site, lo: usize, hi: usize, g: u64) !void {
    return glueRowsAt(f, r, j, site, lo, hi, g, (hi - lo) * r.w.config.hidden);
}

/// glueRows with rank k's partial rows at g + k * rs floats (TF_GLM_OVN_PRX_ZC: the pinned region's blocks).
pub fn glueRowsAt(f: *const Forward, r: Rank, j: usize, site: Forward.Site, lo: usize, hi: usize, g: u64, rs: usize) !void {
    const stamp = prof.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const layers = r.w.layers;
    if (site == .mixer) return postPre(f, r, layers[j].ffn_hc, layers[j].post_norm, lo, hi, g, rs, 0);
    const first = rotlayer.first(&r.w.share, j); // a rotated MoE layer's partials from its first share's rank
    if (j + 1 < layers.len) try postPre(f, r, layers[j + 1].attn_hc, layers[j + 1].in_norm, lo, hi, g, rs, first) else try postRows(f, r, lo, hi, g, rs, first);
    if (rotlayer.capture) |cap| try cap.at(f.stream, r.w.share.rank, j, r.b.x, lo, hi, c.hc * c.hidden * 2);
    if (j + 1 == layers.len) try glue.streamMean(f.set(), f.stream, r.b.x + lo * c.hc * c.hidden * 2, r.b.hidden + lo * c.hidden * 2, hi - lo, c.hidden, c.hc);
}

/// hc_post then hc_pre of rows lo..hi; TF_GLM_OVN_GLUE_ROW=N: one kernel (hcRow) for N rows or more.
fn postPre(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, lo: usize, hi: usize, g: u64, rs: usize, first: usize) !void {
    if (f.m.glue_row == 0 or hi - lo < f.m.glue_row) {
        try postRows(f, r, lo, hi, g, rs, first);
        return preRows(f, r, hc, norm, lo, hi);
    }
    const c = &r.w.config;
    const b = r.b;
    const S: usize = c.hc;
    const D: usize = c.hidden;
    try glue.hcRowAt(f.set(), f.stream, b.x + lo * S * D * 2, g, rs, hc.fn_w, hc.base, hc.scale, norm, b.normed + lo * D * 2, b.xs + lo * (D / 64) * 4, b.post + lo * S * 4, b.comb + lo * S * S * 4, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S, rotlayer.word(r.w.share.world, first));
}

/// hc_post of rows lo..hi from partials `g`, rank k's at g + k * rs floats (a gathered room or a split one's own rows).
fn postRows(f: *const Forward, r: Rank, lo: usize, hi: usize, g: u64, rs: usize, first: usize) !void {
    const c = &r.w.config;
    const b = r.b;
    const S: usize = c.hc;
    const D: usize = c.hidden;
    const x = b.x + lo * S * D * 2;
    try glue.hcPostAt(f.set(), f.stream, x, x, g, rs, b.post + lo * S * 4, b.comb + lo * S * S * 4, hi - lo, D, S, rotlayer.word(r.w.share.world, first));
}

/// forward._pre of rows lo..hi.
fn preRows(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, lo: usize, hi: usize) !void {
    const c = &r.w.config;
    const b = r.b;
    const S: usize = c.hc;
    const D: usize = c.hidden;
    try glue.hcPre(f.set(), f.stream, b.x + lo * S * D * 2, hc.fn_w, hc.base, hc.scale, norm, b.normed + lo * D * 2, b.xs + lo * (D / 64) * 4, b.post + lo * S * 4, b.comb + lo * S * S * 4, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S);
}
