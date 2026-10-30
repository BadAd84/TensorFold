//! TF_GLM_OVN_X3_ROTEXPERT (a numerics trade): three simulated TP=3 ranks of a synthetic GLM EXL3 layer, plain and
//! rotated. Every expert share's slot row is the plain rank's bytes, the rank sums agree within fp32 reassociation,
//! and a rotated row's bits are the row's alone; decode through every X3 variant, prompt through the switches.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const x3 = glm.exl3;
const Rot = cuda.exl3_experts.Rot;

const E = 12; // routed experts (the shared one is id E, skipped)
const D = 512; // dims
const slots = x3.top_k + 1;
const limit: f32 = 7.0;
const world = 3;

/// The checkpoint's layer: per projection (gate, up, down) and expert the i16 trellis bytes and fp16 scales.
const Full = struct { inter: usize, one: bool, tr: [3][]const []const u8, suh: [3][]const []const u8, svh: [3][]const []const u8 };

fn signs(r: *Rig, n: usize, scale: f16) ![]const u8 {
    const out = try r.a.alloc(f16, n);
    for (out) |*v| v.* = if (r.rng.boolean()) scale else -scale;
    return std.mem.sliceAsBytes(out);
}

/// `one`: every gate and up input shares one rotation (the loader's shx).
fn full(r: *Rig, inter: usize, one: bool) !Full {
    var f: Full = .{ .inter = inter, .one = one, .tr = undefined, .suh = undefined, .svh = undefined };
    const shared = try signs(r, D, 1.0);
    for (0..3) |m| {
        const k: usize = if (m < 2) D else inter;
        const n: usize = if (m < 2) inter else D;
        const tr = try r.a.alloc([]const u8, E);
        const su = try r.a.alloc([]const u8, E);
        const sv = try r.a.alloc([]const u8, E);
        for (0..E) |e| {
            const words = try r.a.alloc(i16, k / 16 * (n / 16) * 64);
            for (words) |*v| v.* = r.rng.int(i16);
            tr[e] = std.mem.sliceAsBytes(words);
            su[e] = if (one and m < 2) shared else try signs(r, k, 1.0);
            sv[e] = try signs(r, n, 0.02);
        }
        f.tr[m] = tr;
        f.suh[m] = su;
        f.svh[m] = sv;
    }
    return f;
}

/// Rank `rk`'s layer laid out with cuda_exl3.load's own pieces (expertCut, trellisShare, shareBytes, scaleShare).
fn rank(r: *Rig, f: Full, rk: usize, rot: Rot) !x3.Experts {
    const pad = if (rot.on()) rot.wn + rot.wx else (try x3.expertCut(f.inter, rk, world, 0, .{})).len();
    var w: x3.Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = f.one, .rot = rot };
    const k2: [E]i32 = @splat(8);
    w.k2 = try r.dev(std.mem.sliceAsBytes(&k2));
    var host: std.ArrayList(u8) = .empty;
    var got: [3][4]u64 = undefined; // per projection: trellis base, pointer table, suh, svh
    for (0..3) |m| {
        const by_cols = m < 2;
        const k: usize = if (by_cols) D else f.inter;
        const n: usize = if (by_cols) f.inter else D;
        host.clearRetainingCapacity();
        for (0..E) |e| try x3.trellisShare(r.a, &host, f.tr[m][e], k, n, by_cols, try x3.expertCut(f.inter, rk, world, e, rot));
        got[m][0] = try r.dev(host.items);
        var ptrs: [E]u64 = undefined;
        var at: u64 = got[m][0];
        for (&ptrs, 0..) |*p, e| {
            p.* = at;
            at += x3.shareBytes(k, n, by_cols, try x3.expertCut(f.inter, rk, world, e, rot));
        }
        got[m][1] = try r.dev(std.mem.sliceAsBytes(&ptrs));
        // suh (down: the share, padded), then svh (gate, up: the share, padded); the other projections' whole vectors
        for ([_][]const []const u8{ f.suh[m], f.svh[m] }, [_]bool{ !by_cols, by_cols }, 2..) |vs, padded, slot| {
            host.clearRetainingCapacity();
            for (0..E) |e| {
                if (padded) try x3.scaleShare(r.a, &host, vs[e], try x3.expertCut(f.inter, rk, world, e, rot), pad) else try host.appendSlice(r.a, vs[e]);
            }
            got[m][slot] = try r.dev(host.items);
        }
    }
    w.gb = got[0][0];
    w.gt = got[0][1];
    w.suh_g = got[0][2];
    w.svh_g = got[0][3];
    w.ub = got[1][0];
    w.ut = got[1][1];
    w.suh_u = got[1][2];
    w.svh_u = got[1][3];
    w.db = got[2][0];
    w.dt = got[2][1];
    w.suh_d = got[2][2];
    w.svh_d = got[2][3];
    return w;
}

/// Each row's 8 distinct routed picks, then the shared expert's id.
fn picks(r: *Rig, rows: usize) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [E]bool = @splat(false);
        for (0..slots - 1) |j| {
            var e = r.rng.uintLessThan(usize, E);
            while (seen[e]) e = r.rng.uintLessThan(usize, E);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + slots - 1] = E;
    }
    return out;
}

/// cuda_exl3.run of `w` at `width`: ey [rows][slots][D] bytes (`eb` a value), the shared slot left as the canary.
fn run(r: *Rig, k: *const x3.Kernels, ek: *const glm.experts.Kernels, w: x3.Experts, width: usize, x: []const u16, pk: []const i32, rows: usize, prompt: bool, eb: usize) ![]u8 {
    const P = rows * x3.top_k;
    const item_tile: usize = if (prompt) glm.experts.staged_tile else glm.experts.tile;
    const wide = P > cuda.grouped.small;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E, item_tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) P else 1) * 4),
        .hist = try r.zeros((if (wide) (P + 1023) / 1024 * E else 1) * 4),
    };
    const sz = x3.scratchBytes(rows, D, width, prompt, w.shx, k.prompt.ovn);
    const sc: x3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
    const ey = try r.zeros(rows * slots * D * eb);
    try x3.run(k, ek, r.s, w, sc, plan, try r.dev(std.mem.sliceAsBytes(x)), D, try r.dev(std.mem.sliceAsBytes(pk)), slots, rows, E, D, width, limit, prompt, ey, 0);
    return r.back(ey, rows * slots * D * eb);
}

/// Comparisons made and failed; a failure prints one FAIL line and the run goes on (the count fails it at the end).
const Tally = struct {
    cases: usize = 0,
    failed: usize = 0,
    fn note(t: *Tally, ok: bool, comptime fmt: []const u8, args: anytype) void {
        t.cases += 1;
        if (ok) return;
        t.failed += 1;
        std.debug.print("FAIL " ++ fmt ++ "\n", args);
    }
};

/// Rotated rank rk's slot row of expert e is plain rank (rk - e mod 3) mod 3's, byte for byte (that rank's share).
/// One line a rank: how many slot rows differ, the first, and its first value got against want.
fn shares(t: *Tally, name: []const u8, got: [world][]u8, want: [world][]u8, pk: []const i32, rows: usize, eb: usize) void {
    const n = D * eb;
    for (0..world) |rk| {
        var bad: usize = 0;
        var first: [4]usize = undefined; // row, slot, source rank, column
        for (0..rows) |row| for (0..slots) |j| {
            const e: usize = @intCast(pk[row * slots + j]);
            const src = if (j == slots - 1) rk else (rk + world - e % world) % world; // the shared slot: the canary
            const at = (row * slots + j) * n;
            const g = got[rk][at..][0..n];
            const w = want[src][at..][0..n];
            if (std.mem.eql(u8, g, w)) continue;
            if (bad == 0) first = .{ row, j, src, (std.mem.indexOfDiff(u8, g, w).?) / eb };
            bad += 1;
        };
        if (bad == 0) {
            t.note(true, "", .{});
            continue;
        }
        const at = ((first[0] * slots + first[1]) * D + first[3]) * eb;
        const gb = got[rk][at..][0..eb];
        const wb = want[first[2]][at..][0..eb];
        const gv: f64 = if (eb == 4) @as(f32, @bitCast(std.mem.readInt(u32, gb[0..4], .little))) else @as(f64, @floatFromInt(std.mem.readInt(u16, gb[0..2], .little)));
        const wv: f64 = if (eb == 4) @as(f32, @bitCast(std.mem.readInt(u32, wb[0..4], .little))) else @as(f64, @floatFromInt(std.mem.readInt(u16, wb[0..2], .little)));
        t.note(false, "{s}: rank {d}: {d} of {d} slot rows differ from the plain ranks' shares; first row {d} slot {d} (plain rank {d}) column {d}: got {e}, want {e} ({s})", .{ name, rk, bad, rows * slots, first[0], first[1], first[2], first[3], gv, wv, if (eb == 4) "fp32" else "bf16 bits" });
    }
}

/// The slot rows summed over the ranks in rank order: rotated within 2^-22 of the plain partials' summed magnitudes
/// (two fp32 roundings a sum), and the same bits where the expert's owner is rank 0 (the plain share order).
/// A NaN in either sum is a failure of its own (@max would drop it from the bound).
fn sums(t: *Tally, name: []const u8, got: [world][]u8, want: [world][]u8, pk: []const i32, rows: usize) f64 {
    var most: f64 = 0;
    var nan: usize = 0;
    var owned: usize = 0;
    for (0..rows) |row| for (0..slots - 1) |j| for (0..D) |c| {
        const at = ((row * slots + j) * D + c) * 4;
        var p: f32 = 0;
        var q: f32 = 0;
        var mag: f64 = 0;
        for (0..world) |rk| {
            const a: f32 = @bitCast(std.mem.readInt(u32, want[rk][at..][0..4], .little));
            p += a;
            q += @as(f32, @bitCast(std.mem.readInt(u32, got[rk][at..][0..4], .little)));
            mag += @abs(a);
        }
        if (std.math.isNan(p) or std.math.isNan(q)) {
            nan += 1;
            continue;
        }
        const e: usize = @intCast(pk[row * slots + j]);
        if (e % world == 0 and @as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) owned += 1;
        if (p == q) continue;
        most = @max(most, @abs(@as(f64, p) - q) / (mag * 0x1.00001p-22));
    };
    std.debug.print("RESULT {s}: rank sums within {d:.3} of 2^-22 of their magnitudes\n", .{ name, most });
    t.note(nan == 0, "{s}: {d} rank sums are NaN", .{ name, nan });
    t.note(owned == 0, "{s}: {d} sums of experts owned by rank 0 lost their bits", .{ name, owned });
    t.note(most <= 1, "{s}: rank sums within fp32 reassociation (largest {d:.3})", .{ name, most });
    return most;
}

const variants = [_]cuda.exl3_experts.Ovn{
    .{},
    .{ .loads = .nc1 },
    .{ .loads = .nc2 },
    .{ .loads = .word },
    .{ .fuse_down = true },
    .{ .xrow = true },
    .{ .loads = .nc2, .fuse_down = true, .xrow = true },
    .{ .loads = .word, .fuse_down = true, .xrow = true },
    .{ .loads = .nc1, .fuse_down = true },
};

const prompt_cases = [_]x3.Ovn{ .{}, .{ .k64 = 1 }, .{ .k64 = 2, .order = true, .ey_direct = true }, .{ .k64 = 1, .ybf16 = true } };

pub fn run_(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(31);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }

    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try x3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ek.unload();
    var runs: usize = 0;
    var most: f64 = 0;
    var t: Tally = .{};
    for ([_]usize{ 512, 896 }) |inter| for ([_]bool{ true, false }) |one| { // 4 and 7 blocks: 256/128/128, 384/256/256
        const f = try full(&r, inter, one);
        var plain: [world]x3.Experts = undefined;
        var rotd: [world]x3.Experts = undefined;
        var wp: [world]usize = undefined;
        var wr: usize = 0;
        for (0..world) |rk| {
            const rot = try x3.rotationOf(inter, rk, world);
            try check.expect(rot.on(), "glm-exl3-rotx: {d} columns over {d} ranks rotate", .{ inter, world });
            plain[rk] = try rank(&r, f, rk, .{});
            rotd[rk] = try rank(&r, f, rk, rot);
            wp[rk] = (try x3.expertCut(inter, rk, world, 0, .{})).len();
            wr = rot.wn + rot.wx;
        }
        for ([_]usize{ 1, 2, 3, 17 }) |rows| { // decode windows; 17 rows take two member tiles
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            var want: [world][]u8 = undefined;
            for (0..world) |rk| want[rk] = try run(&r, &k, &ek, plain[rk], wp[rk], x, pk, rows, false, 4);
            for (variants) |o| {
                var got: [world][]u8 = undefined;
                for (0..world) |rk| {
                    var w = rotd[rk];
                    w.ovn = o;
                    got[rk] = try run(&r, &k, &ek, w, wr, x, pk, rows, false, 4);
                }
                const name = try std.fmt.allocPrint(r.a, "glm-exl3-rotx decode {d} columns, one rotation {any}, {d} rows, loads {t}, fuse {any}, xrow {any}", .{ inter, one, rows, o.loads, o.fuse_down, o.xrow });
                shares(&t, name, got, want, pk, rows, 4);
                most = @max(most, sums(&t, name, got, want, pk, rows));
                runs += 1;
            }
            if (rows > 1) for (0..world) |rk| { // concurrent == solo: each row's bits as the row alone
                const all = try run(&r, &k, &ek, rotd[rk], wr, x, pk, rows, false, 4);
                for (0..rows) |row| {
                    const alone = try run(&r, &k, &ek, rotd[rk], wr, x[row * D ..][0..D], pk[row * slots ..][0..slots], 1, false, 4);
                    t.note(std.mem.eql(u8, alone, all[row * slots * D * 4 ..][0 .. slots * D * 4]), "glm-exl3-rotx decode: rank {d} row {d} of {d} equals the row alone", .{ rk, row, rows });
                }
            };
        }
        for ([_]usize{ 40, 200 }) |rows| { // prompt chunks
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            for (prompt_cases, 0..) |o, c| {
                var kp = k;
                kp.prompt.ovn = o;
                try kp.prompt.prepare(kp.mod);
                var kr = k;
                kr.prompt.ovn = o;
                kr.prompt.ovn.rot = true;
                try kr.prompt.prepare(kr.mod);
                const eb: usize = if (o.ybf16) 2 else 4;
                var want: [world][]u8 = undefined;
                var got: [world][]u8 = undefined;
                for (0..world) |rk| {
                    want[rk] = try run(&r, &kp, &ek, plain[rk], wp[rk], x, pk, rows, true, eb);
                    got[rk] = try run(&r, &kr, &ek, rotd[rk], wr, x, pk, rows, true, eb);
                }
                const name = try std.fmt.allocPrint(r.a, "glm-exl3-rotx prompt {d} columns, one rotation {any}, {d} rows, switches case {d}", .{ inter, one, rows, c });
                shares(&t, name, got, want, pk, rows, eb);
                if (eb == 4) most = @max(most, sums(&t, name, got, want, pk, rows));
                runs += 1;
            }
        }
    };
    std.debug.print("RESULT glm-exl3-rotx: {d} of {d} comparisons failed\n", .{ t.failed, t.cases });
    if (t.failed != 0) {
        std.debug.print("FAIL glm-exl3-rotx: {d} of {d} comparisons failed\n", .{ t.failed, t.cases });
        return error.TestFailed;
    }
    check.pass("glm-exl3-rotx: {d} rotated runs give the plain shares' bytes, rank sums within {d:.3} of 2^-22, rows as alone", .{ runs, most });
}
