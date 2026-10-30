//! TF_GLM_OVN_HC_ROWS and _HC_DEC under a recording driver: which mixing kernel each call launches, its grid and args.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const t = std.testing;

/// One launch: its kernel, grid, block, dynamic shared bytes, whether launched by cuLaunchKernelEx, and its arguments.
const Launch = struct { f: usize, grid: [3]c_uint, block: c_uint, shared: c_uint, ex: bool, args: [24]u64, count: usize };

const Rec = struct {
    var log: [8]Launch = undefined;
    var n: usize = 0;

    fn note(f: cuda.abi.Function, grid: [3]c_uint, block: c_uint, shared: c_uint, by_ex: bool, params: ?[*]?*anyopaque) void {
        const ptrs: *const [cuda.Args.max_args]?*anyopaque = @ptrCast(params.?);
        const a: *const cuda.Args = @alignCast(@fieldParentPtr("ptrs", ptrs));
        var l: Launch = .{ .f = @intFromPtr(f), .grid = grid, .block = block, .shared = shared, .ex = by_ex, .args = @splat(0), .count = a.count };
        for (0..a.count) |i| l.args[i] = a.integer(i) orelse 0;
        log[n] = l;
        n += 1;
    }

    fn plain(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f, .{ gx, gy, gz }, bx, shared, false, params);
        return cuda.abi.success;
    }

    fn ex(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f, .{ c.grid_x, c.grid_y, c.grid_z }, c.block_x, c.shared_bytes, true, params);
        return cuda.abi.success;
    }
};

/// A kernel's handle in `kernels`: its field's index plus one.
fn id(comptime name: []const u8) usize {
    return std.meta.fieldIndex(glue.Kernels, name).? + 1;
}

/// Every kernel at its own handle; HC_ROWS' kernels for `rows` (16, 32; 0 none) and HC_DEC's when `dec`, else null.
fn kernels(d: *cuda.Driver, rows: u32, dec: bool) glue.Kernels {
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 0..) |name, i| @field(ks, name) = .{ .d = d, .handle = @ptrFromInt(i + 1) };
    inline for (.{ "hc_partial16", "hc_row16", "hc_row16_f32w" }) |n| if (rows != 16) {
        @field(ks, n) = null;
    };
    inline for (.{ "hc_partial32", "hc_row32", "hc_row32_f32w" }) |n| if (rows != 32) {
        @field(ks, n) = null;
    };
    if (!dec) ks.hc_partial48 = null;
    if (!dec) ks.hc_partial48_pdl = null;
    return ks;
}

const x: u64 = 0x1000_0000;
const fnw: u64 = 0x2000_0000;
const part: u64 = 0x3000_0000;
const nw: glue.Norm = .{ .bf16 = 0x4000_0000 };
const counts = [_]usize{ 1, 2, 4, 5, 8, 16, 32, 63, 64, 65, 70, 130, 683, 1366 };

/// hcPre's launches (the partials, then hc_finish).
fn pre(set: glue.Set, s: cuda.Stream, rows: usize) ![]const Launch {
    Rec.n = 0;
    try glue.hcPre(set, s, x, fnw, 0x5000, 0x6000, nw, 0x7000, 0x8000, 0x9000, 0xa000, part, 1e-6, 1e-6, 20, rows, 4096, 4);
    return Rec.log[0..Rec.n];
}

/// hcRow's one launch.
fn row(set: glue.Set, s: cuda.Stream, norm: glue.Norm, rows: usize) !Launch {
    Rec.n = 0;
    try glue.hcRow(set, s, x, 0x5000, fnw, 0x6000, 0x7000, norm, 0x8000, 0x9000, 0xa000, 0xb000, 1e-6, 1e-6, 20, rows, 4096, 4, 3);
    try t.expectEqual(@as(usize, 1), Rec.n);
    return Rec.log[0];
}

fn cdiv(a: usize, b: usize) c_uint {
    return @intCast((a + b - 1) / b);
}

/// Today's partial for `rows`: hc_partial8 (16 blocks a row) under 64 rows, else hc_partial's 4-row blocks.
fn today(l: Launch, rows: usize, pdl: bool) !void {
    if (rows < 64) {
        try t.expectEqual(if (pdl) id("hc_partial8_pdl") else id("hc_partial8"), l.f);
        try t.expectEqual([3]c_uint{ @intCast(rows), 16, 1 }, l.grid);
        try t.expectEqual(@as(c_uint, 1024), l.block);
        try t.expectEqual(pdl, l.ex);
    } else {
        try t.expectEqual(id("hc_partial"), l.f);
        try t.expectEqual([3]c_uint{ cdiv(rows, 4), 16, 1 }, l.grid);
        try t.expectEqual(@as(c_uint, 128), l.block);
    }
}

test "TF_GLM_OVN_HC_ROWS and _HC_DEC off: hcPre and hcRow launch today's kernels, prompt rows or not" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.plain;
    d.api.cuLaunchKernelEx = Rec.ex;
    const ks = kernels(&d, 0, false);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |prompt| for (counts) |rows| {
        const l = try pre(.{ .glue = &ks, .prompt = prompt }, s, rows);
        try t.expectEqual(@as(usize, 2), l.len);
        try today(l[0], rows, false);
        try t.expectEqual(id("hc_finish"), l[1].f);
        const r = try row(.{ .glue = &ks, .prompt = prompt }, s, nw, rows);
        try t.expectEqual(id("hc_row"), r.f);
        try t.expectEqual([3]c_uint{ cdiv(rows, 4), 1, 1 }, r.grid);
        try t.expectEqual(@as(c_uint, 0), r.shared);
    };
}

test "TF_GLM_OVN_HC_ROWS: prompt rows on R-row blocks at every row count, decode rows on today's kernels" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.plain;
    d.api.cuLaunchKernelEx = Rec.ex;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const f32w: glue.Norm = .{ .f32 = 0x4100_0000 };
    inline for (.{ 16, 32 }) |n| {
        const p = std.fmt.comptimePrint("{d}", .{n});
        const ks = kernels(&d, n, false);
        for (counts) |rows| {
            const l = try pre(.{ .glue = &ks, .prompt = true, .pdl_early = true }, s, rows); // a prompt partial never waits
            try t.expectEqual(id("hc_partial" ++ p), l[0].f);
            try t.expectEqual([3]c_uint{ cdiv(rows, n), 16, 1 }, l[0].grid);
            try t.expectEqual(@as(c_uint, 128), l[0].block);
            try t.expectEqual(false, l[0].ex);
            try t.expectEqualSlices(u64, &.{ x, fnw, part, 16384, 16, rows }, l[0].args[0..l[0].count]);
            try t.expectEqual(id("hc_finish_pdl"), l[1].f);
            try today((try pre(.{ .glue = &ks }, s, rows))[0], rows, false);
            try today((try pre(.{ .glue = &ks, .pdl_early = true }, s, rows))[0], rows, true);
            for ([_]glue.Norm{ nw, f32w }, [_]usize{ id("hc_row" ++ p), id("hc_row" ++ p ++ "_f32w") }) |norm, want| {
                const r = try row(.{ .glue = &ks, .prompt = true }, s, norm, rows);
                try t.expectEqual(want, r.f);
                try t.expectEqual([3]c_uint{ cdiv(rows, n), 1, 1 }, r.grid);
                try t.expectEqual(@as(c_uint, 256), r.block);
                try t.expectEqual(@as(c_uint, n * 16 * 32 * 4), r.shared);
                try t.expectEqual(rows, r.args[15]);
                const w = try row(.{ .glue = &ks }, s, norm, rows); // a decode window's glue: rows4's kernel
                try t.expectEqual(if (norm == .bf16) id("hc_row") else id("hc_row_f32w"), w.f);
                try t.expectEqual([3]c_uint{ cdiv(rows, 4), 1, 1 }, w.grid);
                try t.expectEqual(@as(c_uint, 0), w.shared);
            }
        }
    }
}

test "TF_GLM_OVN_HC_DEC: rows under 64 on hc.cu's grid, launched dependent under PDL_EARLY; 64 rows as today" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.plain;
    d.api.cuLaunchKernelEx = Rec.ex;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const ks = kernels(&d, 0, true);
    for (counts) |rows| for ([_]bool{ false, true }) |pdl| for ([_]bool{ false, true }) |prompt| {
        const l = (try pre(.{ .glue = &ks, .pdl_early = pdl, .prompt = prompt }, s, rows))[0];
        if (rows >= 64) {
            try today(l, rows, pdl);
            continue;
        }
        try t.expectEqual(if (pdl) id("hc_partial48_pdl") else id("hc_partial48"), l.f);
        try t.expectEqual([3]c_uint{ 16, 3, cdiv(rows, 4) }, l.grid);
        try t.expectEqual(@as(c_uint, 288), l.block);
        try t.expectEqual(pdl, l.ex);
        try t.expectEqualSlices(u64, &.{ x, fnw, part, rows }, l.args[0..l.count]);
    };
    const both = kernels(&d, 16, true); // with HC_ROWS too: prompt rows take its blocks, decode windows HC_DEC's
    try t.expectEqual(id("hc_partial16"), (try pre(.{ .glue = &both, .prompt = true }, s, 5))[0].f);
    try t.expectEqual(id("hc_partial48"), (try pre(.{ .glue = &both }, s, 5))[0].f);
    var half = ks;
    half.hc_partial48_pdl = null;
    try t.expectError(error.HcDecUnloaded, pre(.{ .glue = &half, .pdl_early = true }, s, 5));
}

/// A module whose every symbol resolves, its handle the order of lookup; dynamic shared memory opt-ins recorded.
const Mod = struct {
    var names: [256][]const u8 = undefined;
    var smem: [256]c_int = undefined;
    var n: usize = 0;

    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        names[n] = std.mem.span(name);
        smem[n] = 0;
        n += 1;
        out.* = @ptrFromInt(n);
        return cuda.abi.success;
    }

    fn attribute(f: cuda.abi.Function, a: cuda.abi.FunctionAttribute, v: c_int) callconv(.c) cuda.abi.Result {
        if (a == .max_dynamic_shared_size_bytes) smem[@intFromPtr(f) - 1] = v;
        return cuda.abi.success;
    }

    /// `f`'s symbol and opt-in, or null for a kernel not looked up.
    fn of(f: ?cuda.Function) ?struct { []const u8, c_int } {
        const g = f orelse return null;
        return .{ names[@intFromPtr(g.handle) - 1], smem[@intFromPtr(g.handle) - 1] };
    }
};

fn expectLoaded(f: ?cuda.Function, name: []const u8, shared: c_int) !void {
    const got = Mod.of(f) orelse return error.TestExpectedLoaded;
    try t.expectEqualStrings(name, got[0]);
    try t.expectEqual(shared, got[1]);
}

test "TF_GLM_OVN_HC_ROWS and _HC_DEC: loadOvn looks up the knob's kernels only, a bad row count refused" {
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Mod.function;
    d.api.cuFuncSetAttribute = Mod.attribute;
    const m: cuda.Module = .{ .d = &d, .handle = null };
    defer cuda.ovn.fake("HC_ROWS", null);
    defer cuda.ovn.fake("HC_DEC", null);
    cuda.ovn.fake("HC_DEC", null);
    for ([_]?[]const u8{ null, "0", "16", "32" }) |v| {
        cuda.ovn.fake("HC_ROWS", v);
        Mod.n = 0;
        const k = try glue.Kernels.loadOvn(m, false);
        const n = try glue.hcRows();
        inline for (.{ 16, 32 }) |r| {
            const p = std.fmt.comptimePrint("{d}", .{r});
            const want = n == r;
            try t.expectEqual(want, @field(k, "hc_partial" ++ p) != null);
            if (want) try expectLoaded(@field(k, "hc_partial" ++ p), "tf_glm_hc_partial" ++ p, 0);
            inline for (.{ "", "_f32w" }) |w| {
                if (want) try expectLoaded(@field(k, "hc_row" ++ p ++ w), "tf_glm_hc_row" ++ p ++ w, r * 16 * 32 * 4) else try t.expectEqual(@as(?cuda.Function, null), @field(k, "hc_row" ++ p ++ w));
            }
        }
        try t.expectEqual(@as(?cuda.Function, null), k.hc_partial48);
        try t.expectEqual(@as(?cuda.Function, null), k.hc_partial48_pdl);
    }
    cuda.ovn.fake("HC_ROWS", null);
    cuda.ovn.fake("HC_DEC", "1");
    Mod.n = 0;
    const dec = try glue.Kernels.loadOvn(m, false);
    try expectLoaded(dec.hc_partial48, "tf_glm_hc_partial48", 0);
    try expectLoaded(dec.hc_partial48_pdl, "tf_glm_hc_partial48_pdl", 0);
    try t.expectEqual(@as(?cuda.Function, null), dec.hc_partial16);
    for ([_][]const u8{ "8", "64", "x" }) |v| {
        cuda.ovn.fake("HC_ROWS", v);
        Mod.n = 0;
        try t.expectError(error.BadOvnKnob, glue.Kernels.loadOvn(m, false));
    }
}
