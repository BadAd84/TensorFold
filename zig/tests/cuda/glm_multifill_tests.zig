//! glm-multifill (GPU): TF_GLM_OVN_MULTI_FILL's batched resumed prefill against solo chunk fills, byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const glm = @import("glm_cuda");
const Resume = glm.engine.Resume;
const Marks = glm.engine.Marks;
const Leader = glm.follow.Leader;

const max_prompts = 8;
const scratch = 15; // a slot clear of the prompts' (0..max_prompts) for the state-keeping pass

/// One resumed prompt: its fork depth, the kept state it resumes there, and a distinct suffix past it.
const Prompt = struct { depth: usize, id: u64, suffix: []const u32 };

/// What one arm leaves per prompt: first token, position, next window's tokens and logits, kept-state image.
const Rec = struct { image: []const u8, at: u32, first: u32, pos: usize, outs: [4]u32, logits: []const u16 };

/// A fresh pass's marks keep one state at each fork depth, in order.
const Keep = struct {
    e: *glm.engine.Engine,
    ids: [max_prompts]u64 = @splat(0),
    n: usize = 0,
    fn call(ctx: *anyopaque, _: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ctx));
        if (k.n == max_prompts) return;
        k.ids[k.n] = k.e.keepState() catch 0;
        k.n += 1;
    }
};

/// Rank 0's image of kept state `id`: the bytes the state holds, for a held-byte comparison.
fn image(e: *glm.engine.Engine, a: std.mem.Allocator, id: u64) ![]u8 {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const out = try a.alloc(u8, glm.disk_image.imageBytes(try glm.disk_ranks.shape(e), @intCast(s.pos)));
    try glm.disk_ranks.copy(e, id, 0, out, true);
    return out;
}

fn ids(a: std.mem.Allocator, base: []const u32, p: Prompt) ![]const u32 {
    return std.mem.concat(a, u32, &.{ base[0..p.depth], p.suffix });
}

/// The slot's prompt is filled (last row drawn as `first`): image the kept state, then a verify window over it.
fn record(x: Leader, a: std.mem.Allocator, slot: u32, first: u32, V: usize) !Rec {
    const e = x.e;
    try x.bind(slot);
    const pos = e.cur.st.pos;
    const id = try e.keepState();
    if (id == 0) return error.NoKeptState;
    const img = try image(e, a, id);
    const at: u32 = @intCast(e.snaps.get(id).?.pos);
    e.dropState(id);
    const win = [_]u32{ first, 101, 202, 303 };
    var outs: [4]u32 = undefined;
    try x.verify(&win, &outs, 0);
    const logits = try a.alloc(u16, win.len * V);
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(logits.ptr, e.bd.logits, logits.len * 2), "cuMemcpyDtoH");
    return .{ .image = img, .at = at, .first = first, .pos = pos, .outs = outs, .logits = logits };
}

/// The solo arm: each prompt restored on its own slot and filled a chunk a call, as the knob-off path does.
fn solo(x: Leader, a: std.mem.Allocator, base: []const u32, ps: []const Prompt, V: usize, out: []Rec) !void {
    for (ps, 0..) |p, i| {
        try x.bind(@intCast(i));
        try x.begin(null);
        const full = try ids(a, base, p);
        var start = p.depth;
        const first = while (true) {
            const c = try x.chunk(full, start, false, if (start == p.depth) Resume{ .id = p.id, .at = p.depth } else null, null, null);
            if (c.first) |f| break f;
            start += c.rows;
        };
        out[i] = try record(x, a, @intCast(i), first, V);
    }
    for (ps, 0..) |_, i| try x.release(@intCast(i));
}

/// The batched arm: every prompt's suffix in one pass (cuda_prompt_pieces), then each prompt's draw, keep and window.
fn batched(x: Leader, a: std.mem.Allocator, base: []const u32, ps: []const Prompt, V: usize, out: []Rec) !void {
    const pp = glm.prompt_pieces;
    var pieces: [max_prompts]pp.Piece = undefined;
    var marks: [max_prompts]?Marks = undefined;
    var firsts: [max_prompts]?u32 = undefined;
    for (ps, 0..) |p, i| {
        try x.bind(@intCast(i));
        try x.begin(null);
        const full = try ids(a, base, p);
        pieces[i] = .{ .slot = @intCast(i), .start = p.depth, .rows = p.suffix.len, .len = full.len, .part = full[p.depth..], .from = .{ .id = p.id, .at = p.depth }, .mark = false };
        marks[i] = null;
    }
    try x.prefillPieces(pieces[0..ps.len], marks[0..ps.len], firsts[0..ps.len]);
    for (ps, 0..) |_, i| out[i] = try record(x, a, @intCast(i), firsts[i] orelse return error.NoFirst, V);
    for (ps, 0..) |_, i| try x.release(@intCast(i));
}

/// Two records equal: same first token, position, window tokens and logits, and the same held state bytes.
fn same(sh: glm.disk_image.Shape, a: std.mem.Allocator, x: Rec, y: Rec) !bool {
    if (x.first != y.first or x.pos != y.pos or x.at != y.at) return false;
    if (!std.mem.eql(u32, &x.outs, &y.outs) or !std.mem.eql(u16, x.logits, y.logits)) return false;
    if (x.image.len != y.image.len) return false;
    return (try glm.disk_image.heldDiff(sh, x.at, x.image, y.image, a)) == null;
}

pub fn multifillCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const capacity = 1 << 16;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's solo and batched resumed prompt passes", .{rank});
        return;
    }
    e.link = &link;
    const x: Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    try x.snapshots(4 << 30);
    const V = e.share.vocab.len();

    // one shared prefix; keep a state at a dense depth, one crossing the dense limit, and a deep sparse one
    const base = try a.alloc(u32, 6400);
    for (base, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
    const mark_at = [_]u32{ 1500, 2040, 6000 };
    var keep: Keep = .{ .e = e };
    try x.bind(scratch);
    try x.begin(null);
    var start: usize = 0;
    while (true) {
        const c = try x.chunk(base, start, false, null, null, Marks{ .at = &mark_at, .ctx = &keep, .call = Keep.call });
        if (c.first != null) break;
        start += c.rows;
    }
    try x.release(scratch);
    if (keep.n != mark_at.len or keep.ids[0] == 0 or keep.ids[1] == 0 or keep.ids[2] == 0) {
        std.debug.print("FAIL multifill: kept {d} of {d} states (ids {d}/{d}/{d})\n", .{ keep.n, mark_at.len, keep.ids[0], keep.ids[1], keep.ids[2] });
        return error.TestFailed;
    }

    // suffix seeds: distinct per prompt so the forks never share a first token or a window
    const S = struct {
        fn make(al: std.mem.Allocator, seed: usize, n: usize) ![]u32 {
            const s = try al.alloc(u32, n);
            for (s, 0..) |*u, j| u.* = @intCast(200 + (j * 6007 + seed * 101) % 4000);
            return s;
        }
    };
    const deep = keep.ids[2]; // the forks cell: several short suffixes off one long prefix (depth 6000)
    const ps = [_]Prompt{
        .{ .depth = 6000, .id = deep, .suffix = try S.make(a, 1, 20) },
        .{ .depth = 6000, .id = deep, .suffix = try S.make(a, 2, 20) },
        .{ .depth = 6000, .id = deep, .suffix = try S.make(a, 3, 20) },
        .{ .depth = 6000, .id = deep, .suffix = try S.make(a, 4, 20) },
        .{ .depth = 1500, .id = keep.ids[0], .suffix = try S.make(a, 5, 20) }, // dense, a different depth
        .{ .depth = 2040, .id = keep.ids[1], .suffix = try S.make(a, 6, 40) }, // crosses the dense limit, longer
    };

    var rsolo: [ps.len]Rec = undefined;
    var rbatch: [ps.len]Rec = undefined;
    var rred: [ps.len]Rec = undefined;
    try solo(x, a, base, &ps, V, &rsolo);
    try batched(x, a, base, &ps, V, &rbatch);

    // red control: one fork's last suffix token changed; its records must differ and no other piece may move
    var red = ps;
    const rs = try a.dupe(u32, ps[0].suffix);
    rs[rs.len - 1] = if (rs[rs.len - 1] == 4321) 4322 else 4321;
    red[0].suffix = rs;
    try batched(x, a, base, &red, V, &rred);

    const sh = try glm.disk_ranks.shape(e);
    var fails: usize = 0;
    for (rsolo, rbatch, 0..) |s, b, i| {
        if (!try same(sh, a, s, b)) {
            std.debug.print("FAIL multifill: prompt {d} (depth {d}, suffix {d}) batched differs from solo\n", .{ i, ps[i].depth, ps[i].suffix.len });
            fails += 1;
        }
    }
    for (rbatch, rred, 0..) |b, r, i| {
        const eq = try same(sh, a, b, r);
        if (i == 0 and eq) {
            std.debug.print("FAIL multifill: the red control (prompt 0's suffix changed) left its first token, window and state unchanged\n", .{});
            fails += 1;
        }
        if (i != 0 and !eq) {
            std.debug.print("FAIL multifill: prompt 0's change bled into prompt {d} (pieces are not independent)\n", .{i});
            fails += 1;
        }
    }
    for ([_]u64{ keep.ids[0], keep.ids[1], keep.ids[2] }) |id| e.dropState(id);
    if (fails != 0) return error.TestFailed;
    check.pass("multifill at TP={d}: {d} resumed prompts (four forks of one depth-6000 state, a dense depth-1500, one crossing the dense limit) batched in one pass equal their solo chunk fills in first token, window tokens and logits, position and every held state byte; the red control differed only on its own piece", .{ world, ps.len });
}
