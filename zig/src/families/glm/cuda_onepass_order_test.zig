//! TF_GLM_OVN_ONEPASS_EXACT accumulation-order reference (host, no GPU): three scalar fp32 models of one attention row.

const std = @import("std");

const lw = 8; // latent width in the model (the real kernel's 512; the order properties do not depend on it)
const chunk = 4; // keys a chunk (the real kernel's CHUNK = 512)
const tile = 2; // keys a tile within a chunk (the real kernel's KT = 32)
const max_keys = 16; // a small synthetic row

const inf = std.math.inf(f32);

// Round an f32 to bf16 and back, round to nearest even, as __float2bfloat16_rn does for finite values.
fn bf16(x: f32) f32 {
    const b: u32 = @bitCast(x);
    if (b & 0x7fff_ffff > 0x7f80_0000) return x; // a NaN stays a NaN (the models never feed one)
    const bias: u32 = 0x7fff + ((b >> 16) & 1);
    return @bitCast((b +% bias) & 0xffff_0000);
}

// tl.exp's value (the model uses one exp for every order, so it cancels between the chunk and exact orders).
fn expf(x: f32) f32 {
    return @exp(x);
}

const Row = struct {
    n: usize,
    q: [lw]f32, // the query, rounded to bf16 by the models
    value: [max_keys][lw]f32, // each key's unscaled fp32 value (the e4m3 decode before the scale)
    scale: [max_keys]f32, // each key's fp32 dequant scale
};

const Partial = struct {
    m: f32 = -inf,
    l: f32 = 0,
    o: [lw]f32 = @splat(0),
};

// chunk / exact: key i's value with the scale folded in (bf16, as chunks_wide's FP8 store writes kvs), and its score.
fn chunkKey(row: Row, qk: f32, i: usize, vs: *[lw]f32) f32 {
    var dot: f32 = 0;
    for (0..lw) |d| {
        vs[d] = bf16(bf16(row.value[i][d]) * row.scale[i]);
        dot = @mulAdd(f32, bf16(row.q[d]), vs[d], dot);
    }
    return dot * qk;
}

// One 512-chunk's online softmax over its tiles (chunks_wide's tile loop); keys past n are masked and add nothing.
fn chunkPartial(row: Row, qk: f32, start: usize) Partial {
    var p: Partial = .{};
    var t: usize = 0;
    while (t < chunk) : (t += tile) {
        var sc: [tile]f32 = undefined;
        var vs: [tile][lw]f32 = undefined;
        var tmax: f32 = -inf;
        for (0..tile) |k| {
            const idx = start + t + k;
            if (idx < row.n) {
                sc[k] = chunkKey(row, qk, idx, &vs[k]);
                tmax = @max(tmax, sc[k]);
            } else sc[k] = -inf;
        }
        const active = tmax != -inf;
        const nm = if (active) @max(p.m, tmax) else p.m;
        const alpha: f32 = if (active) (if (p.m == -inf) 0 else expf(p.m - nm)) else 1;
        for (0..lw) |d| p.o[d] = p.o[d] * alpha;
        var lt: f32 = 0;
        for (0..tile) |k| {
            const idx = start + t + k;
            const pk: f32 = if (active and idx < row.n) expf(sc[k] - nm) else 0;
            const pb = bf16(pk);
            for (0..lw) |d| p.o[d] = @mulAdd(f32, pb, vs[k][d], p.o[d]);
            lt += pk;
        }
        p.l = @mulAdd(f32, p.l, alpha, lt);
        p.m = nm;
    }
    return p;
}

// tf_glm_merge's fold of chunk `p` into the running (m, l, o): the same fma and fmul merge_body uses, `first` its c=0.
fn foldMerge(run: *Partial, p: Partial, first: bool) void {
    const cm = p.m;
    const cl = p.l;
    const active = cl > 0;
    const nm = if (active) @max(run.m, cm) else run.m;
    const b: f32 = if (active) expf(cm - nm) else 0;
    if (first) {
        for (0..lw) |d| run.o[d] = @mulAdd(f32, b, p.o[d], 0);
        run.l = @mulAdd(f32, b, cl, 0);
    } else {
        const a: f32 = if (active) (if (run.m == -inf) 0 else expf(run.m - nm)) else 1;
        for (0..lw) |d| run.o[d] = @mulAdd(f32, run.o[d], a, b * p.o[d]);
        run.l = @mulAdd(f32, run.l, a, b * cl);
    }
    run.m = nm;
}

fn nchunks(n: usize) usize {
    return (n + chunk - 1) / chunk;
}

// The pipe reuses the PV o across chunks; the first tile's alpha = exp(m_reset - nm) must zero the carry when reset.
fn chunkPartialReset(row: Row, qk: f32, start: usize, o_carry: [lw]f32, m_reset: f32) Partial {
    var p: Partial = .{ .m = m_reset, .l = 0, .o = o_carry };
    var t: usize = 0;
    while (t < chunk) : (t += tile) {
        var sc: [tile]f32 = undefined;
        var vs: [tile][lw]f32 = undefined;
        var tmax: f32 = -inf;
        for (0..tile) |k| {
            const idx = start + t + k;
            if (idx < row.n) {
                sc[k] = chunkKey(row, qk, idx, &vs[k]);
                tmax = @max(tmax, sc[k]);
            } else sc[k] = -inf;
        }
        const active = tmax != -inf;
        const nm = if (active) @max(p.m, tmax) else p.m;
        const alpha: f32 = if (active) (if (p.m == -inf) 0 else expf(p.m - nm)) else 1;
        for (0..lw) |d| p.o[d] = p.o[d] * alpha;
        var lt: f32 = 0;
        for (0..tile) |k| {
            const idx = start + t + k;
            const pk: f32 = if (active and idx < row.n) expf(sc[k] - nm) else 0;
            const pb = bf16(pk);
            for (0..lw) |d| p.o[d] = @mulAdd(f32, pb, vs[k][d], p.o[d]);
            lt += pk;
        }
        p.l = @mulAdd(f32, p.l, alpha, lt);
        p.m = nm;
    }
    return p;
}

// Today's default: every chunk's partial materialised, then one merge pass over them (chunks then tf_glm_merge).
fn chunkOrder(row: Row) [lw]f32 {
    const nch = nchunks(row.n);
    var parts: [max_keys]Partial = undefined;
    for (0..nch) |c| parts[c] = chunkPartial(row, qk_scale, c * chunk);
    var run: Partial = .{};
    for (0..nch) |c| foldMerge(&run, parts[c], c == 0);
    var out: [lw]f32 = undefined;
    for (0..lw) |d| out[d] = bf16(run.o[d] / run.l);
    return out;
}

// TF_GLM_OVN_ONEPASS_EXACT: one pass a row, each chunk's partial folded into the running merge as it is produced.
fn exactOrder(row: Row) [lw]f32 {
    const nch = nchunks(row.n);
    var run: Partial = .{};
    for (0..nch) |c| foldMerge(&run, chunkPartial(row, qk_scale, c * chunk), c == 0);
    var out: [lw]f32 = undefined;
    for (0..lw) |d| out[d] = bf16(run.o[d] / run.l);
    return out;
}

// TF_GLM_OVN_SPARSE_ONEPASS: one global online softmax over all keys, scale on the score and folded into P, no reset.
fn onepassGlobal(row: Row) [lw]f32 {
    var m: f32 = -inf;
    var l: f32 = 0;
    var o: [lw]f32 = @splat(0);
    var t: usize = 0;
    while (t < row.n) : (t += tile) {
        var sc: [tile]f32 = undefined;
        var vr: [tile][lw]f32 = undefined;
        var tmax: f32 = -inf;
        for (0..tile) |k| {
            const idx = t + k;
            if (idx < row.n) {
                var dot: f32 = 0;
                for (0..lw) |d| {
                    vr[k][d] = bf16(row.value[idx][d]);
                    dot = @mulAdd(f32, bf16(row.q[d]), vr[k][d], dot);
                }
                sc[k] = dot * row.scale[idx] * qk_scale;
                tmax = @max(tmax, sc[k]);
            } else sc[k] = -inf;
        }
        const active = tmax != -inf;
        const nm = if (active) @max(m, tmax) else m;
        const alpha: f32 = if (active) (if (m == -inf) 0 else expf(m - nm)) else 1;
        for (0..lw) |d| o[d] = o[d] * alpha;
        var lt: f32 = 0;
        for (0..tile) |k| {
            const idx = t + k;
            const pk: f32 = if (active and idx < row.n) expf(sc[k] - nm) else 0;
            const pst = bf16(pk * row.scale[idx]);
            for (0..lw) |d| o[d] = @mulAdd(f32, pst, vr[k][d], o[d]);
            lt += pk;
        }
        l = @mulAdd(f32, l, alpha, lt);
        m = nm;
    }
    var out: [lw]f32 = undefined;
    for (0..lw) |d| out[d] = bf16(o[d] / l);
    return out;
}

const qk_scale: f32 = 0.125;

// A deterministic synthetic row of `n` keys whose per-chunk score ranges differ, so the chunk grouping matters.
fn makeRow(seed: u64, n: usize) Row {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var row: Row = .{ .n = n, .q = undefined, .value = undefined, .scale = undefined };
    for (&row.q) |*x| x.* = rand.floatNorm(f32) * 0.5;
    for (0..n) |i| {
        const mag: f32 = @floatFromInt(1 + (i / chunk) * 3); // a chunk's keys share a magnitude, chunks differ
        for (&row.value[i]) |*x| x.* = rand.floatNorm(f32) * mag;
        row.scale[i] = std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(rand.uintLessThan(u32, 4))));
    }
    return row;
}

fn same(a: [lw]f32, b: [lw]f32) bool {
    for (0..lw) |d| if (@as(u32, @bitCast(a[d])) != @as(u32, @bitCast(b[d]))) return false;
    return true;
}

test "onepass exact order: equals the chunk programs' order bit for bit and differs from the existing one-pass" {
    // a single key: softmax weight 1, so the output is just its scaled value rounded to bf16 (a known value)
    const one = makeRow(1, 1);
    const got = exactOrder(one);
    for (0..lw) |d| try std.testing.expectEqual(bf16(bf16(bf16(one.value[0][d]) * one.scale[0])), got[d]);

    var differed = false;
    for ([_]usize{ 5, 7, 10, 12, 16 }, 0..) |n, s| {
        const row = makeRow(0x51A + s, n);
        const ch = chunkOrder(row);
        const ex = exactOrder(row);
        const op = onepassGlobal(row);
        try std.testing.expect(same(ex, ch)); // the exact one-pass order equals the chunk programs' order, bit for bit
        if (nchunks(n) > 1 and !same(ex, op)) differed = true; // and a multi-chunk row separates it from the global one
    }
    try std.testing.expect(differed); // at least one multi-chunk row separates the exact order from the global one
}

test "onepass pipe reset: a reset chunk zeroes the carried PV accumulator (every chunk's first tile is active)" {
    const reset_m = -inf; // the pipe resets the QK max to -inf at a chunk boundary; the first tile's alpha is then 0
    for ([_]usize{ 5, 8, 12, 13, 16 }, 0..) |n, s| { // n <= max_keys; chunk = 4, so these are 2 to 4 chunks
        const row = makeRow(0xC0 + s, n);
        var carry: [lw]f32 = undefined;
        for (&carry, 0..) |*c, d| c.* = 3.5 + @as(f32, @floatFromInt(d)); // a nonzero leftover from a previous chunk
        var cstart: usize = 0;
        while (cstart < n) : (cstart += chunk) { // every live chunk's first tile has a key, so alpha = 0 zeroes carry
            const fresh = chunkPartial(row, qk_scale, cstart);
            const reused = chunkPartialReset(row, qk_scale, cstart, carry, reset_m);
            try std.testing.expect(same(fresh.o, reused.o)); // the carry must not survive a correct reset
            try std.testing.expectEqual(fresh.l, reused.l);
            try std.testing.expectEqual(fresh.m, reused.m);
        }
    }
}
