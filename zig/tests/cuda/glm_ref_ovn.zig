//! TF_GLM_OVN_GLUE_FUSE, _GLUE_ROW and _PDL_EARLY: each knob's kernels give the bytes of the launches they replace.
const std = @import("std");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const glue = glm.glue;

const d = 4096;
const s = 4;
const wide = s * d;
const world = 3;
const iters = 20;
const eps: f32 = 1e-6;

/// A site's inputs: residual streams, the ranks' gathered partials, the last hc_pre's post and comb.
const Host = struct { x: []const u16, g: []const f32, post: []const f32, comb: []const f32 };

fn host(r: *Rig, rows: usize) !Host {
    const g = try r.a.alloc(f32, world * rows * d);
    for (g) |*v| v.* = (r.rng.float(f32) * 2 - 1) * 0.5;
    const post = try r.a.alloc(f32, rows * s);
    for (post) |*v| v.* = r.rng.float(f32) * 2;
    const comb = try r.a.alloc(f32, rows * s * s);
    for (comb) |*v| v.* = r.rng.float(f32);
    return .{ .x = try r.bfs(rows * wide, 1.0), .g = g, .post = post, .comb = comb };
}

/// One path's device copy of a site and its outputs (filled with 0xff: a launch that writes nothing fails).
const Dev = struct { x: u64, g: u64, post: u64, comb: u64, out: u64, xs: u64, part: u64 };

fn dev(r: *Rig, h: Host, rows: usize) !Dev {
    return .{
        .x = try r.dev(std.mem.sliceAsBytes(h.x)),
        .g = try r.dev(std.mem.sliceAsBytes(h.g)),
        .post = try r.dev(std.mem.sliceAsBytes(h.post)),
        .comb = try r.dev(std.mem.sliceAsBytes(h.comb)),
        .out = try r.zeros(rows * d * 2),
        .xs = try r.zeros(rows * (d / 64) * 4),
        .part = try r.zeros(glue.hcPartBytes(rows)),
    };
}

/// hcGlue's count: a zero u32 a row.
fn counts(r: *Rig, rows: usize) !u64 {
    const z = try r.a.alloc(u32, rows);
    @memset(z, 0);
    return r.dev(std.mem.sliceAsBytes(z));
}

/// The next site's mixing weights (fn, base, scale) and norm: bf16, or float32 holding bf16 values (FP4-Local's).
const Site = struct { w: [3]u64, norm: glue.Norm };

fn site(r: *Rig, f32w: bool) !Site {
    const base = try r.a.alloc(f32, 24);
    for (base) |*b| b.* = (r.rng.float(f32) * 2 - 1) * 0.5;
    const scale = [3]f32{ 0.7, 0.9, 1.1 };
    const w = [3]u64{ try r.dev(std.mem.sliceAsBytes(try r.bfs(24 * wide, 0.02))), try r.dev(std.mem.sliceAsBytes(base)), try r.dev(std.mem.asBytes(&scale)) };
    const nw = try r.bfs(d, 1.0);
    if (!f32w) return .{ .w = w, .norm = .{ .bf16 = try r.dev(std.mem.sliceAsBytes(nw)) } };
    const g = try r.a.alloc(f32, d);
    for (g, nw) |*o, v| o.* = @floatCast(bf(v));
    return .{ .w = w, .norm = .{ .f32 = try r.dev(std.mem.sliceAsBytes(g)) } };
}

/// Path b's residual (at bx) and outputs equal path a's (at ax); the partials too when b writes them.
fn same(r: *Rig, what: []const u8, rows: usize, a: Dev, ax: u64, b: Dev, bx: u64, parts: bool) !void {
    var name: [160]u8 = undefined;
    const outs = [_]struct { []const u8, u64, u64, usize }{
        .{ "residual", ax, bx, rows * wide * 2 },
        .{ "normed rows", a.out, b.out, rows * d * 2 },
        .{ "group sums", a.xs, b.xs, rows * (d / 64) * 4 },
        .{ "post", a.post, b.post, rows * s * 4 },
        .{ "comb", a.comb, b.comb, rows * s * s * 4 },
        .{ "partials", a.part, b.part, glue.hcPartBytes(rows) },
    };
    for (outs[0..if (parts) outs.len else outs.len - 1]) |o|
        try ref.sameRows(r, try std.fmt.bufPrint(&name, "{s}, {d} rows: {s}", .{ what, rows, o[0] }), o[1], o[2], o[3]);
}

/// The knobs-off glue at a site: hcPost in place, then hcPre.
fn split(r: *Rig, set: glue.Set, v: Dev, g: u64, w: Site, rows: usize) !void {
    try glue.hcPost(set, r.s, v.x, v.x, g, v.post, v.comb, rows, d, s, world);
    try glue.hcPre(set, r.s, v.x, w.w[0], w.w[1], w.w[2], w.norm, v.out, v.xs, v.post, v.comb, v.part, eps, eps, iters, rows, d, s);
}

fn fused(r: *Rig, set: glue.Set, v: Dev, x: u64, xout: u64, g: u64, count: u64, w: Site, rows: usize) !void {
    try glue.hcGlue(set, r.s, x, xout, g, w.w[0], w.w[1], w.w[2], w.norm, v.out, v.xs, v.post, v.comb, v.part, count, eps, eps, iters, rows, d, s, world);
}

/// GLUE_FUSE: two sites by hcGlue (x into x_alt and back) equal hcPost then hcPre twice; the counts end at zero.
fn fuse(r: *Rig, set: glue.Set) !void {
    for ([_]bool{ false, true }) |f32w| for ([_]usize{ 1, 2, 5, 8, 32, 33, 48, 63 }) |rows| { // to TF_GLM_OVN_SHARED_ROWS
        const w = try site(r, f32w);
        const h = try host(r, rows);
        const g2 = std.mem.sliceAsBytes((try host(r, rows)).g);
        const a = try dev(r, h, rows);
        const b = try dev(r, h, rows);
        const alt = try r.zeros(rows * wide * 2);
        const count = try counts(r, rows);
        try split(r, set, a, a.g, w, rows);
        try fused(r, set, b, b.x, alt, b.g, count, w, rows);
        try same(r, "fused glue, first site", rows, a, a.x, b, alt, true);
        try split(r, set, a, try r.dev(g2), w, rows);
        try fused(r, set, b, alt, b.x, try r.dev(g2), count, w, rows);
        try same(r, "fused glue, second site", rows, a, a.x, b, b.x, true);
        for (std.mem.bytesAsSlice(u32, try r.back(count, rows * 4))) |n| try check.expect(n == 0, "fused glue: the row counts return to zero", .{});
    };
    // 64 rows: hcGlue refuses them, so the forward glues them unfused; each row equals the fused glue's at 63
    for ([_]bool{ false, true }) |f32w| {
        const w = try site(r, f32w);
        const h = try host(r, 64);
        const g = try r.a.alloc(f32, world * 63 * d); // [world][rows][d]: each rank's first 63 rows
        for (0..world) |q| @memcpy(g[q * 63 * d ..][0 .. 63 * d], h.g[q * 64 * d ..][0 .. 63 * d]);
        const h63: Host = .{ .x = h.x[0 .. 63 * wide], .g = g, .post = h.post[0 .. 63 * s], .comb = h.comb[0 .. 63 * s * s] };
        const a = try dev(r, h, 64);
        const b = try dev(r, h63, 63);
        const alt = try r.zeros(63 * wide * 2);
        const refused = if (fused(r, set, a, a.x, alt, a.g, try counts(r, 64), w, 64)) false else |err| err == error.HcGlueWindow;
        try check.expect(refused, "hcGlue refuses 64 rows (the forward's unfused path takes them)", .{});
        try split(r, set, a, a.g, w, 64);
        try fused(r, set, b, b.x, alt, b.g, try counts(r, 63), w, 63);
        try same(r, "unfused glue at 64 rows against the fused glue at 63", 63, a, a.x, b, alt, true);
    }
    check.pass("TF_GLM_OVN_GLUE_FUSE: hcGlue's residual, normed rows, sums, post, comb and partials equal hcPost + hcPre (1 to 63 rows), and the unfused 64-row glue's each row", .{});
}

/// GLUE_ROW: hcRow equals hcPost in place then hcPre, prompt rows and a decode window's.
fn row(r: *Rig, set: glue.Set) !void {
    for ([_]bool{ false, true }) |f32w| for ([_]usize{ 1, 3, 70, 130 }) |rows| {
        const w = try site(r, f32w);
        const h = try host(r, rows);
        const a = try dev(r, h, rows);
        const b = try dev(r, h, rows);
        try split(r, set, a, a.g, w, rows);
        try glue.hcRow(set, r.s, b.x, b.g, w.w[0], w.w[1], w.w[2], w.norm, b.out, b.xs, b.post, b.comb, eps, eps, iters, rows, d, s, world);
        try same(r, "row glue", rows, a, a.x, b, b.x, false);
    };
    check.pass("TF_GLM_OVN_GLUE_ROW: hcRow's residual, normed rows, sums, post and comb equal hcPost + hcPre", .{});
}

/// HC_ROWS with GLUE_ROW: hcRow on `forms` (prompt sets, 16 and 32 rows a block) equals hcPost then knobs-off hcPre.
pub fn rowForms(r: *Rig, off: glue.Set, forms: []const glue.Set, sizes: []const usize) !void {
    for ([_]bool{ false, true }) |f32w| for (sizes) |rows| {
        const w = try site(r, f32w);
        const h = try host(r, rows);
        const a = try dev(r, h, rows);
        try split(r, off, a, a.g, w, rows);
        for (forms) |set| {
            const b = try dev(r, h, rows);
            try glue.hcRow(set, r.s, b.x, b.g, w.w[0], w.w[1], w.w[2], w.norm, b.out, b.xs, b.post, b.comb, eps, eps, iters, rows, d, s, world);
            try same(r, if (set.glue.hc_row32 != null) "row glue, 32 rows a block" else "row glue, 16 rows a block", rows, a, a.x, b, b.x, false);
        }
    };
    check.pass("TF_GLM_OVN_HC_ROWS with GLUE_ROW: hcRow's 16- and 32-row blocks equal hcPost + hcPre at every row count", .{});
}

/// PDL_EARLY: the glue chain on the _pdl kernels (dependent launches included) gives the plain chain's bytes.
fn early(r: *Rig, set: glue.Set) !void {
    var on = set;
    on.pdl_early = true;
    const ne = 256;
    const router_w = try r.dev(std.mem.sliceAsBytes(try r.bfs(ne * d, 0.02)));
    const nws = [3]u64{ try r.dev(std.mem.sliceAsBytes(try r.bfs(d, 1.0))), try r.dev(std.mem.sliceAsBytes(try r.bfs(1536, 1.0))), try r.dev(std.mem.sliceAsBytes(try r.bfs(512, 1.0))) };
    const plain = try site(r, false);
    const kept = try site(r, true);
    var name: [160]u8 = undefined;
    for ([_]usize{ 2, 70 }) |rows| {
        const h = try host(r, rows);
        const gu = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * 2 * 1024, 4.0)));
        const sizes = [_]usize{ rows * ne * 4, rows * 704 * 2, rows * 11 * 4, rows * 1024 * 2, rows * 16 * 4, rows * 1536 * 2, rows * 512 * 2, rows * d * 2, rows * 64 * 4, rows * wide * 2, rows * d * 2 };
        var got: [2][sizes.len]u64 = undefined;
        var sites: [2]Dev = undefined;
        for ([2]glue.Set{ set, on }, &got, &sites) |st, *o, *v| {
            for (o, sizes) |*p, n| p.* = try r.zeros(n);
            v.* = try dev(r, h, rows);
            try split(r, st, v.*, v.g, plain, rows); // hc_post_pdl, then hc_partial8_pdl (or hc_partial), hc_finish_pdl
            try glue.router(st, r.s, v.out, d, router_w, o[0], try r.zeros(glue.router_ks * rows * ne * 4), rows, d, ne);
            try glue.swiglu(st, r.s, gu, o[1], o[2], 7.0, rows, 704);
            try glue.swiglu(st, r.s, gu, o[3], o[4], 7.0, rows, 1024);
            try glue.rmsnorm(st, r.s, v.out, d, nws[1], o[5], 1536, null, eps, rows, 1536);
            try glue.rmsnorm(st, r.s, v.out, d, nws[2], o[6], 512, null, eps, rows, 512);
            try glue.rmsnormBy(st, r.s, v.out, d, kept.norm, o[7], d, o[8], eps, rows, d);
            try glue.rmsnorm(st, r.s, o[7], d, nws[0], o[10], d, null, eps, rows, d);
            if (rows < 64) try fused(r, st, v.*, v.x, o[9], v.g, try counts(r, rows), kept, rows);
        }
        try same(r, "early launches, the hc chain", rows, sites[0], sites[0].x, sites[1], sites[1].x, true);
        for (sizes, 0..) |n, i| if (i != 9 or rows < 64)
            try ref.sameRows(r, try std.fmt.bufPrint(&name, "early launches, {d} rows: output {d} of the router, swiglu and norms", .{ rows, i }), got[0][i], got[1][i], n);
    }
    check.pass("TF_GLM_OVN_PDL_EARLY: hc_post, hc_pre, router, swiglu, rmsnorm and hcGlue's _pdl kernels give the same bytes", .{});
}

/// glm-ref's checks of the three knobs (after gammas: the draws of every check before keep their order).
pub fn run(r: *Rig, set: glue.Set) !void {
    try fuse(r, set);
    try row(r, set);
    try early(r, set);
}
