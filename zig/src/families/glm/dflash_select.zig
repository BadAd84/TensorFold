//! DFlash2's path selector on rank 0: each draft among a row's 16 likeliest by the previous draft's codebook row.
const ovn = @import("dflash_ovn.zig");
const df = @import("dflash.zig");
const CW = df.CW;
const RANK = df.RANK;
const TOPK = df.TOPK;

const Cand = struct { v: f32, id: u32 };

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Stream `i`'s 7 drafts: each row's best of every rank's 16 by logit plus (EDGE-weighted) codebook score, chance kept.
pub fn select(dr: *const df.Drafter, i: usize, M: usize, q: df.Req) void {
    const out = q.out.?;
    const k = &dr.knobs;
    const keyed = if (k.noise > 0) q.sampling else null; // NOISE: a sampled stream picks with the target's keyed draws
    const inv_t: f32 = if (keyed) |s| @floatCast(1.0 / @max(s.temperature, 1e-6)) else 1; // cuda_sampler.rule's 1 / T
    var prev: usize = q.anchor;
    for (1..df.BLOCK) |t| {
        const r = i * df.BLOCK + t;
        var c: [TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..dr.world) |rk| {
            const blk = dr.cand[rk * M * CW + r * CW ..][0..CW];
            for (0..TOPK) |j| {
                if (blk[j] < 0) continue;
                const cand: Cand = .{ .v = @bitCast(blk[TOPK + j]), .id = @intCast(blk[j]) };
                if (n < TOPK) {
                    c[n] = cand;
                    n += 1;
                    continue;
                }
                var lo: usize = 0;
                for (1..TOPK) |m| if (c[m].v < c[lo].v) {
                    lo = m;
                };
                if (cand.v > c[lo].v) c[lo] = cand;
            }
        }
        if (n == 0) { // no candidate (cannot happen with a whole vocabulary): the anchor repeated, chance 0
            for (t - 1..df.BLOCK - 1) |slot| {
                out[slot] = @intCast(prev);
                if (q.probs) |p| p[slot] = 0;
            }
            return;
        }
        const g = dr.gh[r * RANK ..][0..RANK];
        const P = dr.pred[prev * RANK ..][0..RANK];
        var pg: [RANK]f32 = undefined;
        for (&pg, P, g) |*o, p, x| o.* = bf(p) * x;
        var best: usize = 0; // the pick: the largest key
        var sc: [TOPK]f32 = undefined; // the chance is softmax(sc / temp) at the pick
        var key: [TOPK]f32 = undefined;
        for (0..n) |m| {
            const Sc = dr.succ[@as(usize, c[m].id) * RANK ..][0..RANK];
            var dot: f32 = 0;
            for (pg, Sc) |a, b| dot += a * bf(b);
            sc[m] = c[m].v + k.edge * dot; // EDGE unset: 1.0, today's bits
            key[m] = sc[m];
            if (keyed) |s| { // pick score / T + noise g; chance softmax((score / T + g) / beta) with BETA, else of score
                const g_m = ovn.gumbel(s.seed, q.pos + @as(i64, @intCast(t)), c[m].id); // the draft's position, its id
                sc[m] *= inv_t;
                key[m] = sc[m] + k.noise * g_m;
                if (k.noisy) sc[m] += g_m;
            }
            if (key[m] > key[best]) best = m;
        }
        var z: f64 = 0;
        for (sc[0..n]) |x| z += @exp(@as(f64, x - sc[best]) / dr.temp);
        out[t - 1] = c[best].id;
        if (q.probs) |p| p[t - 1] = 1.0 / z;
        prev = c[best].id;
    }
}
