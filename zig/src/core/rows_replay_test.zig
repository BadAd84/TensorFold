//! TF_GLM_OVN_DISK_CARRY's replay on the host: a Fake family's stored states prefilled again and compared on each rank.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const carry = @import("rows_carry.zig");
const replay = @import("rows_replay.zig");
const fake = @import("rows_fake.zig");
const local = @import("rows_local.zig");
const prefetch = @import("rows_prefetch.zig");
const chain = @import("rows_chain.zig");
const lanes = @import("lanes");
const image = lanes.rows_image;
const file = lanes.rows_file;
const compare = lanes.carry_compare;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Record = carry.record.Record;
const Allocator = std.mem.Allocator;

const gpa = std.testing.allocator;
const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const old_identity: u64 = 7; // the build that wrote the states
const new_identity: u64 = 8; // a kernel change: another identity, the same carry key
const key: u64 = 0xca55;

/// Under TMPDIR, else the working directory: std's file calls (setting a file's time) read the same path there.
fn root(buf: []u8, tag: []const u8) []const u8 {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    return std.fmt.bufPrint(buf, "{s}/tf-rows-replay-{s}-{d}", .{ tmp, tag, platform.c.getpid() }) catch unreachable;
}

fn oldDir(buf: []u8, dir: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/{x:0>16}", .{ dir, carry.dirOf(old_identity) });
}

/// What a start under `identity` brings: its own directory, the key, a proof, the store's keying.
fn want(identity: u64) Record {
    return .{ .dir = carry.dirOf(identity), .key = key, .proof = 1, .block = rows.block_tokens, .lookahead = 1 };
}

/// A conversation of 1500 tokens, its words salted (`picture`: one of them a picture's row).
fn prompt(a: Allocator, salt: u32, picture: bool) ![]u32 {
    const out = try a.alloc(u32, 1500);
    for (out, 0..) |*t, i| t.* = @intCast(50_000 + i * 7 % 997 + 1000 * salt);
    if (picture) out[100] = 0x8000_0123;
    return out;
}

const Keep = struct { salt: u32, at: u32, picture: bool = false };

/// The old build's store: each conversation of `keeps` kept at its position, and the directory's record.
fn seed(a: Allocator, dir: []const u8, keeps: []const Keep) !void {
    var f: fake.Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), .{ .sync_writes = true, .root = dir, .identity = old_identity, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 });
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    for (keeps) |k| {
        const p = try prompt(a, k.salt, k.picture);
        f.prompt = p;
        const plan = try s.begin(a, p, k.at, &.{}, f.starts, null, f.spans);
        _ = f.pass(&s, plan);
        while (rows.write.step(r)) {}
    }
    try std.testing.expectEqual(keeps.len, r.states.items.len);
    try carry.record.write(r.dir, want(old_identity));
}

const StatePlan = struct { at: u32, name: chain.Name, plan: prefetch.Plan };

/// The old directory's states as the old build opens them, each with the plan of its files (a's memory).
fn plans(a: Allocator, dir: []const u8) ![]StatePlan {
    var f: fake.Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), .{ .root = dir, .identity = old_identity, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 });
    defer r.deinit();
    const out = try a.alloc(StatePlan, r.states.items.len);
    for (r.states.items, out) |*s, *o| o.* = .{ .at = s.at, .name = s.name, .plan = try prefetch.Plan.of(r, a, s) };
    return out;
}

/// A rank other than 0 (TF_GLM_OVN_DISK_LOCAL): its own disk, holding its part of every file under its rank's names.
const Rank = struct {
    rank: u32,
    root: [:0]const u8,

    /// File `f` on this rank's disk, in rank 0's directory `dir` as this rank's root holds it.
    fn path(k: *const Rank, buf: []u8, dir: []const u8, f: local.File) ![:0]const u8 {
        var eb: [24]u8 = undefined;
        var sub: [400]u8 = undefined;
        const mine = try std.fmt.bufPrint(&sub, "{s}/{s}", .{ k.root, std.fs.path.basename(dir) });
        return file.path(buf, mine, &chain.hex(f.name), try file.rankExt(&eb, if (f.kind == .block) ".blk" else ".st", k.rank), false);
    }

    /// This rank's answer for `files` of fresh state `saved` at `at`, as a follower compares its own part.
    fn compareAll(k: *const Rank, family: pc.Snapshots, dir: []const u8, saved: pc.Saved, at: u32, files: []const local.File) !compare.Compare {
        var parts: std.ArrayList(image.Part) = .empty;
        defer parts.deinit(gpa);
        try fake.Fake.layout(at, &parts, gpa);
        var g: Gather = .{ .family = family, .saved = saved };
        for (files) |f| {
            var rs: std.ArrayList(image.Range) = .empty;
            defer rs.deinit(gpa);
            if (f.kind == .block) try image.blockRanges(parts.items, rows.block_tokens, f.window, f.k, &rs, gpa) else try image.ownRanges(parts.items, at, rows.block_tokens, &rs, gpa);
            const n = image.total(rs.items);
            const stage = try file.stage(gpa, n);
            defer gpa.free(stage);
            const buf = try gpa.alloc(u8, @intCast(n));
            defer gpa.free(buf);
            var pb: [600]u8 = undefined;
            const c = try compare.same(gpa, try k.path(&pb, dir, f), f.kind, f.name, rows.block_tokens, if (f.kind == .state) at else 0, rs.items, null, stage, buf, .{ .ctx = &g, .gather = Gather.gather });
            if (c != .same) return c;
        }
        return .same;
    }
};

const Gather = struct {
    family: pc.Snapshots,
    saved: pc.Saved,

    fn gather(ctx: *anyopaque, ranges: []const image.Range, out: []u8) anyerror!void {
        const g: *Gather = @ptrCast(@alignCast(ctx));
        try g.family.vtable.rows.?.gather(g.family.ptr, g.saved, ranges, out);
    }
};

/// This build's engine as a replay sees it: the Fake family's fresh states, and the other ranks.
const Engine = struct {
    f: fake.Fake = .{ .gpa = gpa },
    asked: std.ArrayList(u32) = .empty, // each fresh state's position, in order
    ids: std.ArrayList([]u32) = .empty, // the ids each was asked with
    ranks: []const Rank = &.{},
    fail: bool = false, // its prefill fails: a replay that could not run

    fn deinit(e: *Engine) void {
        for (e.ids.items) |x| gpa.free(x);
        e.ids.deinit(gpa);
        e.asked.deinit(gpa);
    }

    fn engine(e: *Engine) replay.Engine {
        return .{ .ctx = e, .fresh = fresh, .others = if (e.ranks.len > 0) others else null };
    }

    fn fresh(ctx: *anyopaque, tokens: []const u32, at: u32) anyerror!pc.Saved {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        if (e.fail) return error.TestPrefill;
        try e.asked.append(gpa, at);
        try e.ids.append(gpa, try gpa.dupe(u32, tokens));
        e.f.prompt = tokens;
        e.f.at = at;
        return e.f.snapshots().vtable.save(&e.f, null, at);
    }

    fn others(ctx: *anyopaque, dir: []const u8, saved: pc.Saved, at: u32, files: []const local.File, out: []u64) anyerror![]const u64 {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        out[0] = 0;
        for (e.ranks, 1..) |*k, i| out[i] = compare.word(try k.compareAll(e.f.snapshots(), dir, saved, at, files));
        return out[0 .. e.ranks.len + 1];
    }

    fn run(e: *Engine, dir: []const u8, cap: u64) replay.Verdict {
        return replay.run(gpa, e.f.snapshots(), dir, carry.dirOf(old_identity), want(new_identity), cap, e.engine());
    }
};

/// The file of piece `p` in `dir` (extension `ext`) written again with payload byte `byte` changed, its checks whole.
fn rewrite(dir: []const u8, ext: []const u8, p: prefetch.Piece, byte: usize) !void {
    var pb: [600]u8 = undefined;
    var o = try file.open(try file.path(&pb, dir, &chain.hex(p.name), ext, false), p.kind, p.name);
    const stage = try file.stage(gpa, p.len);
    defer gpa.free(stage);
    const got = file.payload(&o, stage) catch |err| {
        o.close();
        return err;
    };
    const t = o.trailer;
    o.close();
    got[byte] +%= 1;
    var w = try file.Writer.open(dir, &chain.hex(p.name), ext);
    try w.append(got);
    _ = try w.finish(.{ .kind = t.kind, .payload = 0, .payload_hash = 0, .name = t.name, .at = t.at, .block = t.block }, &.{}, true);
}

/// Rank `k`'s own copy of every file of `pl` (rank 0's bytes under the rank's names), in its own root.
fn copyTo(k: *const Rank, dir: []const u8, pl: *const prefetch.Plan) !void {
    var sub: [400]u8 = undefined;
    const mine = try std.fmt.bufPrintSentinel(&sub, "{s}/{s}", .{ k.root, std.fs.path.basename(dir) }, 0);
    _ = platform.c.mkdir(k.root, 0o700);
    _ = platform.c.mkdir(mine, 0o700);
    for (pl.pieces) |p| {
        var from: [600]u8 = undefined;
        var to: [600]u8 = undefined;
        const f: local.File = .{ .name = p.name, .kind = p.kind, .window = p.window, .k = p.k };
        try copyFile(try file.path(&from, dir, &chain.hex(p.name), prefetch.ext(p.kind), false), try k.path(&to, dir, f));
    }
}

fn copyFile(from: [:0]const u8, to: [:0]const u8) !void {
    const in = platform.c.open(from, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (in < 0) return error.TestCopy;
    defer _ = platform.c.close(in);
    const end = platform.c.lseek(in, 0, platform.c.SEEK.END);
    if (end < 0) return error.TestCopy;
    const bytes = try gpa.alloc(u8, @intCast(end));
    defer gpa.free(bytes);
    if (platform.c.pread(in, bytes.ptr, bytes.len, 0) != end) return error.TestCopy;
    const out = platform.c.open(to, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    if (out < 0) return error.TestCopy;
    defer _ = platform.c.close(out);
    if (platform.c.write(out, bytes.ptr, bytes.len) != end) return error.TestCopy;
}

test "rows_carry replay accepts states this build computes alike, asked by their own ids; none of them stays kept" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "same");
    defer rmTree(dir);
    try seed(a, dir, &.{ .{ .salt = 0, .at = 1400 }, .{ .salt = 3, .at = 1100 }, .{ .salt = 5, .at = 900, .picture = true } });
    var e: Engine = .{};
    defer e.deinit();
    try std.testing.expectEqual(replay.Verdict{ .same = 2 }, e.run(dir, replay.default_tokens));
    try std.testing.expectEqual(@as(usize, 0), e.f.live); // every fresh state dropped: the engine as a fresh boot leaves it
    // the two prompt-row states, each by its own file's ids; the picture's state is never asked (ids cannot prefill it)
    try std.testing.expectEqual(@as(usize, 2), e.asked.items.len);
    for (e.ids.items, e.asked.items) |ids, at| {
        const salt: u32 = if (at == 1400) 0 else 3;
        try std.testing.expect(at == 1400 or at == 1100);
        try std.testing.expectEqualSlices(u32, (try prompt(a, salt, false))[0 .. at + 1], ids);
    }
}

test "rows_carry replay refuses a state whose stored bytes differ in one block, naming the state and the byte" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "block");
    defer rmTree(dir);
    try seed(a, dir, &.{.{ .salt = 0, .at = 1400 }});
    const st = (try plans(a, dir))[0];
    var old_buf: [160]u8 = undefined;
    const blk = st.plan.pieces[1]; // a block of rows from position 0
    try std.testing.expectEqual(file.Kind.block, blk.kind);
    try rewrite(try oldDir(&old_buf, dir), ".blk", blk, 0);
    var e: Engine = .{};
    defer e.deinit();
    const at_byte = st.plan.rangesOf(blk)[0].off;
    try std.testing.expectEqual(replay.Verdict{ .differs = .{ .rank = 0, .name = st.name, .at = 1400, .off = at_byte } }, e.run(dir, replay.default_tokens));
    try std.testing.expectEqual(@as(usize, 0), e.f.live);
    // through the carry: no adoption, no alias, and the new build starts in its own directory
    var hook: Hook = .{ .e = &e, .dir = dir, .cap = replay.default_tokens };
    const c = carry.choose(dir, want(new_identity), null, hook.proof());
    try std.testing.expect(!c.adopted and c.id == carry.dirOf(new_identity) and hook.calls == 1);
    try std.testing.expect(carry.readAlias(dir, carry.dirOf(new_identity)) == null);
}

test "rows_carry replay refuses when one rank's part differs; a rank lacking its part proves nothing" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "ranks");
    defer rmTree(dir);
    try seed(a, dir, &.{.{ .salt = 0, .at = 1400 }});
    const st = (try plans(a, dir))[0];
    var old_buf: [160]u8 = undefined;
    const old_dir = try oldDir(&old_buf, dir);
    var rb: [2][160]u8 = undefined;
    const ranks = [_]Rank{
        .{ .rank = 1, .root = try std.fmt.bufPrintSentinel(&rb[0], "{s}-r1", .{dir}, 0) },
        .{ .rank = 2, .root = try std.fmt.bufPrintSentinel(&rb[1], "{s}-r2", .{dir}, 0) },
    };
    defer for (ranks) |k| rmTree(k.root);
    for (&ranks) |*k| try copyTo(k, old_dir, &st.plan);
    var e: Engine = .{ .ranks = &ranks };
    defer e.deinit();
    try std.testing.expectEqual(replay.Verdict{ .same = 1 }, e.run(dir, replay.default_tokens));
    // the last rank's part of one block holds another byte: that rank, that state, that byte
    var r2_buf: [400]u8 = undefined;
    const r2_dir = try std.fmt.bufPrint(&r2_buf, "{s}/{s}", .{ ranks[1].root, std.fs.path.basename(old_dir) });
    var eb: [24]u8 = undefined;
    const blk = st.plan.pieces[2];
    try rewrite(r2_dir, try file.rankExt(&eb, ".blk", 2), blk, 5);
    const off = st.plan.rangesOf(blk)[0].off + 5;
    try std.testing.expectEqual(replay.Verdict{ .differs = .{ .rank = 2, .name = st.name, .at = 1400, .off = off } }, e.run(dir, replay.default_tokens));
    // that rank's part whole again, and the first rank's own file gone: nothing replayed proves the directory
    try copyTo(&ranks[1], old_dir, &st.plan);
    var gone: [600]u8 = undefined;
    const own = st.plan.pieces[st.plan.pieces.len - 1];
    try std.testing.expectEqual(@as(c_int, 0), platform.c.unlink(try ranks[0].path(&gone, old_dir, .{ .name = own.name, .kind = .state })));
    try std.testing.expect(e.run(dir, replay.default_tokens) == .none);
    // every other rank whole, rank 0 lacking a block: skipped before its prefill, so no proof either
    try copyTo(&ranks[0], old_dir, &st.plan);
    var b0: [600]u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), platform.c.unlink(try file.path(&b0, old_dir, &chain.hex(st.plan.pieces[0].name), ".blk", false)));
    const asked = e.asked.items.len;
    try std.testing.expect(e.run(dir, replay.default_tokens) == .none);
    try std.testing.expectEqual(asked, e.asked.items.len);
    try std.testing.expectEqual(@as(usize, 0), e.f.live);
}

test "rows_carry replay takes the newest states first, the longest among equals, within its rows and three at most" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "cap");
    defer rmTree(dir);
    try seed(a, dir, &.{ .{ .salt = 1, .at = 500 }, .{ .salt = 2, .at = 800 }, .{ .salt = 3, .at = 1400 }, .{ .salt = 4, .at = 300 } });
    // their files' times: 500 the newest, then 800, then 1400 and 300 together
    var old_buf: [160]u8 = undefined;
    const old_dir = try oldDir(&old_buf, dir);
    for (try plans(a, dir)) |st| {
        const t: i96 = switch (st.at) {
            500 => 3,
            800 => 2,
            else => 1,
        };
        var pb: [600]u8 = undefined;
        try age(try file.path(&pb, old_dir, &chain.hex(st.name), ".st", false), 1_700_000_000 + t);
    }
    const Case = struct { cap: u64, asked: []const u32 };
    for ([_]Case{
        .{ .cap = 1 << 20, .asked = &.{ 500, 800, 1400 } }, // three at most
        .{ .cap = 1700, .asked = &.{ 500, 800, 300 } }, // 1400 would pass the cap; 300 still fits after it
        .{ .cap = 400, .asked = &.{300} },
        .{ .cap = 200, .asked = &.{} }, // nothing fits: no state replayed, no proof
    }) |case| {
        var e: Engine = .{};
        defer e.deinit();
        const v = e.run(dir, case.cap);
        try std.testing.expectEqualSlices(u32, case.asked, e.asked.items);
        if (case.asked.len == 0) try std.testing.expect(v == .none) else try std.testing.expectEqual(replay.Verdict{ .same = case.asked.len }, v);
    }
}

/// File `p`'s modify time set to `s` seconds past the epoch, through std (whose file calls read the same path).
fn age(p: []const u8, s: i96) !void {
    const fh = try std.Io.Dir.cwd().openFile(std.testing.io, p, .{ .mode = .read_write });
    defer fh.close(std.testing.io);
    try fh.setTimestamps(std.testing.io, .{ .modify_timestamp = .{ .new = .fromNanoseconds(s * std.time.ns_per_s) } });
}

/// The carry's replay hook over this file's engine, and how often it ran.
const Hook = struct {
    e: *Engine,
    dir: []const u8,
    cap: u64,
    calls: usize = 0,

    fn proof(h: *Hook) carry.Replay {
        return .{ .ctx = h, .run = run, .cap = h.cap };
    }

    fn run(ctx: *anyopaque, id: u64) carry.Proof {
        const h: *Hook = @ptrCast(@alignCast(ctx));
        h.calls += 1;
        return replay.proof(replay.run(gpa, h.e.f.snapshots(), h.dir, id, want(new_identity), h.cap, h.e.engine()));
    }
};

/// A replay standing in for rows_replay's: one answer under one cap, and how often it ran.
const Answer = struct {
    says: carry.Proof,
    cap: u64 = replay.default_tokens,
    calls: usize = 0,

    fn proof(r: *Answer) carry.Replay {
        return .{ .ctx = r, .run = run, .cap = r.cap };
    }

    fn run(ctx: *anyopaque, _: u64) carry.Proof {
        const r: *Answer = @ptrCast(@alignCast(ctx));
        r.calls += 1;
        return r.says;
    }
};

test "rows_carry replay runs only before an alias is written: never without a directory to adopt, nor again after" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "once");
    defer rmTree(dir);
    var e: Engine = .{};
    defer e.deinit();
    var hook: Hook = .{ .e = &e, .dir = dir, .cap = replay.default_tokens };
    try std.testing.expect(!carry.choose(dir, want(new_identity), null, hook.proof()).adopted); // nothing to adopt
    try std.testing.expectEqual(@as(usize, 0), hook.calls);
    try seed(a, dir, &.{.{ .salt = 0, .at = 1400 }});
    const c = carry.choose(dir, want(new_identity), null, hook.proof());
    try std.testing.expect(c.adopted and hook.calls == 1 and e.asked.items.len == 1);
    try std.testing.expect(carry.choose(dir, want(new_identity), null, hook.proof()).adopted); // its alias in place
    try std.testing.expectEqual(@as(usize, 1), hook.calls);
}

test "rows_carry remembers a refusal: the same build's next starts skip the candidate without replaying it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "refused");
    defer rmTree(dir);
    try seed(a, dir, &.{.{ .salt = 0, .at = 1400 }});
    var old_buf: [160]u8 = undefined;
    try rewrite(try oldDir(&old_buf, dir), ".blk", (try plans(a, dir))[0].plan.pieces[1], 0); // computed otherwise here
    var e: Engine = .{};
    defer e.deinit();
    var hook: Hook = .{ .e = &e, .dir = dir, .cap = replay.default_tokens };
    const own = carry.dirOf(new_identity);
    try std.testing.expect(!carry.choose(dir, want(new_identity), null, hook.proof()).adopted);
    // a crash's alias left beside the refusal changes nothing: refused from the record, never replayed again
    var alias_buf: [200]u8 = undefined;
    const al: carry.Alias = .{ .own = own, .dir = carry.dirOf(old_identity), .key = key };
    try carry.record.replace(try std.fmt.bufPrintSentinel(&alias_buf, "{s}/{x:0>16}.carry", .{ dir, own }, 0), std.mem.asBytes(&al));
    for (0..2) |_| {
        const c = carry.choose(dir, want(new_identity), null, hook.proof());
        try std.testing.expect(!c.adopted and c.id == own and carry.readAlias(dir, own) == null);
    }
    try std.testing.expectEqual(@as(usize, 1), hook.calls);
    try std.testing.expectEqual(@as(usize, 1), e.asked.items.len);
    // a replay that could not run is no refusal: another build's failing starts replay each time
    var failing: Engine = .{ .fail = true };
    defer failing.deinit();
    var tried: Hook = .{ .e = &failing, .dir = dir, .cap = replay.default_tokens };
    for (0..2) |_| try std.testing.expect(!carry.choose(dir, want(9), null, tried.proof()).adopted);
    try std.testing.expectEqual(@as(usize, 2), tried.calls);
}

test "rows_carry voids a remembered refusal when the candidate's newest state, its states, the key or the cap change" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "void");
    defer rmTree(dir);
    try seed(a, dir, &.{ .{ .salt = 1, .at = 500 }, .{ .salt = 2, .at = 800 } });
    var old_buf: [160]u8 = undefined;
    const old_dir = try oldDir(&old_buf, dir);
    var newest_buf: [600]u8 = undefined;
    var newest: []const u8 = "";
    for (try plans(a, dir)) |st| {
        var pb: [600]u8 = undefined;
        const p = try file.path(&pb, old_dir, &chain.hex(st.name), ".st", false);
        try age(p, if (st.at == 500) 1_700_000_003 else 1_700_000_002);
        if (st.at == 500) newest = try std.fmt.bufPrint(&newest_buf, "{s}", .{p});
    }
    var w = want(new_identity);
    var no: Answer = .{ .says = .refused };
    try twice(dir, w, &no, 1); // refused and remembered: its second start not replayed
    try age(newest, 1_700_000_005); // the newest state written again
    try twice(dir, w, &no, 2);
    var more: [600]u8 = undefined; // a state more, older than the newest: only the set changed
    try age(try fakeState(&more, old_dir, 0xa1), 1_700_000_001);
    try twice(dir, w, &no, 3);
    no.cap = 200_000;
    try twice(dir, w, &no, 4);
    var rec = want(old_identity); // another key on both sides, the directory still proving it
    rec.key += 1;
    w.key += 1;
    try carry.record.write(old_dir, rec);
    try twice(dir, w, &no, 5);
    // an adoption forgets the refusal: its states back as remembered, the alias still holds without a replay
    var yes: Answer = .{ .says = .same, .cap = no.cap };
    var last: [600]u8 = undefined;
    const added = try fakeState(&last, old_dir, 0xa2);
    try std.testing.expect(carry.choose(dir, w, null, yes.proof()).adopted);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.unlink(added));
    try std.testing.expect(carry.choose(dir, w, null, yes.proof()).adopted);
    try std.testing.expectEqual(@as(usize, 1), yes.calls);
}

/// Two starts of build `w` with replay `r`: neither adopts, and the replay has run `calls` times after them.
fn twice(dir: []const u8, w: Record, r: *Answer, calls: usize) !void {
    for (0..2) |_| try std.testing.expect(!carry.choose(dir, w, null, r.proof()).adopted);
    try std.testing.expectEqual(calls, r.calls);
}

/// A state file in `dir` named by `n`, its bytes no state (a refusal's record reads names and times only).
fn fakeState(buf: []u8, dir: []const u8, n: u8) ![:0]const u8 {
    const p = try std.fmt.bufPrintSentinel(buf, "{s}/{x:0>32}.st", .{ dir, n }, 0);
    const fd = platform.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.TestState;
    _ = platform.c.close(fd);
    return p;
}

test "rows_carry forgets a refusal once the build's own directory holds states, or nothing is left to adopt" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "forget");
    defer rmTree(dir);
    try seed(a, dir, &.{.{ .salt = 0, .at = 1400 }});
    var no: Answer = .{ .says = .refused };
    var ref_buf: [200]u8 = undefined;
    for ([_]u64{ new_identity, 9 }) |id| {
        try std.testing.expect(!carry.choose(dir, want(id), null, no.proof()).adopted);
        try std.testing.expect(present(try refusalOf(&ref_buf, dir, id)));
    }
    // the new build's own directory holds a state: its refusal goes with its alias
    var own_buf: [200]u8 = undefined;
    const own_dir = try std.fmt.bufPrintSentinel(&own_buf, "{s}/{x:0>16}", .{ dir, carry.dirOf(new_identity) }, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(own_dir, 0o700));
    var st_buf: [600]u8 = undefined;
    _ = try fakeState(&st_buf, own_dir, 0xb1);
    try std.testing.expect(!carry.choose(dir, want(new_identity), null, no.proof()).adopted);
    try std.testing.expect(!present(try refusalOf(&ref_buf, dir, new_identity)));
    // the older directory gone: nothing to adopt, and the other build's refusal goes too
    var old_buf: [160]u8 = undefined;
    rmTree(try oldDir(&old_buf, dir));
    try std.testing.expect(!carry.choose(dir, want(9), null, no.proof()).adopted);
    try std.testing.expect(!present(try refusalOf(&ref_buf, dir, 9)));
    try std.testing.expectEqual(@as(usize, 2), no.calls);
}

/// Build `identity`'s remembered refusal under `dir`.
fn refusalOf(buf: []u8, dir: []const u8, identity: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.refused", .{ dir, carry.dirOf(identity) }, 0);
}

fn present(p: [:0]const u8) bool {
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return false;
    _ = platform.c.close(fd);
    return true;
}
