//! TF_GLM_OVN_DISK_CARRY's replay: a directory's stored states prefilled again from their ids, compared byte for byte.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const chain = @import("rows_chain.zig");
const file = @import("lanes").rows_file;
const compare = @import("lanes").carry_compare;
const rows_store = @import("rows_store.zig");
const prefetch = @import("rows_prefetch.zig");
const local = @import("rows_local.zig");
const scan = @import("rows_scan.zig");
const Rows = rows_store.Rows;
const State = rows_store.State;
const Record = @import("lanes").carry_record.Record;
const Proof = @import("rows_carry.zig").Proof;
const Allocator = std.mem.Allocator;

/// States a replay prefills at most, as production's gate samples them (SpillCarryDriver: three).
pub const most = 3;

/// TF_GLM_OVN_DISK_CARRY_REPLAY_TOKENS unset: rows prefilled at most, every replayed state's together.
pub const default_tokens: u64 = 300_000;

/// A word with this bit is a picture's row in a cache key (server/image_key.zig): ids alone cannot prefill it.
const picture: u32 = 0x8000_0000;

/// The engine side: a fresh state of stored ids, and (TF_GLM_OVN_DISK_LOCAL) every other rank's own comparison.
pub const Engine = struct {
    ctx: *anyopaque,
    /// A new kept state of `tokens` at `at`, prefilled by this build (the family's drop frees it).
    fresh: *const fn (ctx: *anyopaque, tokens: []const u32, at: u32) anyerror!pc.Saved,
    /// Every rank's carry_compare word (rank 0's 0) for its own part of `files` in `dir`, against `saved` on that rank.
    others: ?*const fn (ctx: *anyopaque, dir: []const u8, saved: pc.Saved, at: u32, files: []const local.File, out: []u64) anyerror![]const u64 = null,
};

/// A replay's outcome: how many states were the same on every rank, none to replay, or the first difference.
pub const Verdict = union(enum) {
    same: usize,
    none,
    differs: struct { rank: usize, name: chain.Name, at: u32, off: u64 }, // the state and its first differing byte
    failed: anyerror,
};

/// The carry's answer: proved; refused and remembered (a difference, nothing to prove it); an error is tried again.
pub fn proof(v: Verdict) Proof {
    return switch (v) {
        .same => .same,
        .none, .differs => .refused,
        .failed => .failed,
    };
}

/// One state's outcome: the same everywhere, skipped (a rank could not read its files), or differing on `rank`.
const Result = union(enum) { same, skip, differs: struct { rank: usize, off: u64 } };

/// A stored state the replay may take, and its file's time.
const Candidate = struct { state: State, mtime: i128 };

/// Directory `id`'s states (keyed as `keying` says): the newest first, the longest among equals, `most` and `cap` rows.
pub fn run(gpa: Allocator, family: pc.Snapshots, root: []const u8, id: u64, keying: Record, cap: u64, e: Engine) Verdict {
    const view = Rows.init(gpa, family, .{ .root = root, .identity = 0, .adopted = id, .cap = 0, .floor = 0, .daily = 0, .lookahead = keying.lookahead, .planned = keying.planned != 0 }) catch |err| return .{ .failed = err };
    defer view.deinit();
    var list: std.ArrayList(Candidate) = .empty;
    defer {
        for (list.items) |*c| Rows.freeState(gpa, &c.state);
        list.deinit(gpa);
    }
    candidates(view, &list) catch |err| return .{ .failed = err };
    std.mem.sort(Candidate, list.items, {}, newer);
    var spent: u64 = 0;
    var same: usize = 0;
    for (list.items) |*c| {
        if (same == most) break;
        if (spent + c.state.at > cap) continue;
        const got = one(view, &c.state, e) catch |err| return .{ .failed = err };
        if (got.prefilled) spent += c.state.at;
        switch (got.result) {
            .same => same += 1,
            .skip => {},
            .differs => |d| return .{ .differs = .{ .rank = d.rank, .name = c.state.name, .at = c.state.at, .off = d.off } },
        }
    }
    return if (same == 0) .none else .{ .same = same };
}

/// Newest first by the state file's time, the longest first among equals, then by name: the same order every start.
fn newer(_: void, x: Candidate, y: Candidate) bool {
    if (x.mtime != y.mtime) return x.mtime > y.mtime;
    if (x.state.at != y.state.at) return x.state.at > y.state.at;
    return std.mem.lessThan(u8, &x.state.name, &y.state.name);
}

/// The view's states a prompt pass of their ids alone reproduces: no decoded span and no picture's row.
fn candidates(view: *const Rows, out: *std.ArrayList(Candidate)) !void {
    const d = platform.c.opendir(view.dir) orelse return error.RowsDir;
    defer _ = platform.c.closedir(d);
    while (platform.c.readdir(d)) |ent| {
        const name = scan.stateName(ent) orelse continue;
        var buf: [1200]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ view.dir, name }, 0) catch continue;
        var s = scan.stateOf(view, p) catch continue;
        const replayable = s.spans.len == 0 and for (s.tokens) |t| {
            if (t & picture != 0) break false;
        } else true;
        if (!replayable) {
            Rows.freeState(view.gpa, &s);
            continue;
        }
        out.append(view.gpa, .{ .state = s, .mtime = scan.mtimeOf(p) }) catch |err| {
            Rows.freeState(view.gpa, &s);
            return err;
        };
    }
}

/// State `s`: its files on rank 0 present, then prefilled fresh, compared here and on every other rank, dropped again.
fn one(view: *Rows, s: *const State, e: Engine) !struct { prefilled: bool, result: Result } {
    var plan = try prefetch.Plan.of(view, view.gpa, s);
    defer plan.deinit(view.gpa);
    for (plan.pieces) |p| {
        var buf: [1200]u8 = undefined;
        var o = compare.opens(try path(&buf, view, p), p.kind, p.name, view.keying.block, p.at, p.len) orelse return .{ .prefilled = false, .result = .skip };
        o.close();
    }
    const saved = try e.fresh(e.ctx, s.tokens, s.at);
    defer view.family.vtable.drop(view.family.ptr, saved);
    const here = try mine(view, &plan, saved);
    if (here != .same) return .{ .prefilled = true, .result = result(0, here) };
    const others = e.others orelse return .{ .prefilled = true, .result = .same };
    const files = try view.gpa.alloc(local.File, plan.pieces.len);
    defer view.gpa.free(files);
    for (plan.pieces, files) |p, *f| f.* = .{ .name = p.name, .kind = p.kind, .window = p.window, .k = p.k };
    var words: [64]u64 = undefined;
    var skip = false;
    for (try others(e.ctx, view.dir, saved, s.at, files, &words), 0..) |w, rank| {
        const r = result(rank, compare.answer(w));
        switch (r) {
            .same => {},
            .skip => skip = true,
            .differs => return .{ .prefilled = true, .result = r },
        }
    }
    return .{ .prefilled = true, .result = if (skip) .skip else .same };
}

fn result(rank: usize, c: compare.Compare) Result {
    return switch (c) {
        .same => .same,
        .unreadable => .skip,
        .differs => |off| .{ .differs = .{ .rank = rank, .off = off } },
    };
}

/// Rank 0's files of the plan against the fresh state's image, as the store gathers it, one file after another.
fn mine(view: *Rows, plan: *const prefetch.Plan, saved: pc.Saved) !compare.Compare {
    var most_bytes: u64 = 0;
    for (plan.pieces) |p| most_bytes = @max(most_bytes, p.len);
    const stage = try view.staged(most_bytes);
    const buf = try view.gpa.alloc(u8, @intCast(most_bytes));
    defer view.gpa.free(buf);
    var g: Gather = .{ .view = view, .saved = saved };
    const fresh: compare.Fresh = .{ .ctx = &g, .gather = Gather.gather };
    for (plan.pieces) |p| {
        var pb: [1200]u8 = undefined;
        const c = try compare.same(try path(&pb, view, p), p.kind, p.name, view.keying.block, p.at, plan.rangesOf(p), stage, buf, fresh);
        if (c != .same) return c;
    }
    return .same;
}

const Gather = struct {
    view: *Rows,
    saved: pc.Saved,

    fn gather(ctx: *anyopaque, ranges: []const @import("lanes").rows_image.Range, out: []u8) anyerror!void {
        const g: *Gather = @ptrCast(@alignCast(ctx));
        try g.view.img.gather(g.view.family.ptr, g.saved, ranges, out);
    }
};

fn path(buf: []u8, view: *const Rows, p: prefetch.Piece) ![:0]const u8 {
    return file.path(buf, view.dir, &chain.hex(p.name), prefetch.ext(p.kind), false);
}

test {
    _ = @import("rows_replay_test.zig");
}

test "rows_carry replay order: the newest file first, the longest among equals, then by name" {
    const of = struct {
        fn c(at: u32, n: u8, mtime: i128) Candidate {
            return .{ .state = .{ .name = @splat(n), .at = at, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = 0, .rows = 0, .used = 0, .shared = false }, .mtime = mtime };
        }
    }.c;
    var list = [_]Candidate{ of(300, 1, 1), of(1400, 9, 1), of(500, 5, 3), of(1400, 2, 1), of(800, 0, 2) };
    std.mem.sort(Candidate, &list, {}, newer);
    const want = [_][2]u32{ .{ 500, 5 }, .{ 800, 0 }, .{ 1400, 2 }, .{ 1400, 9 }, .{ 300, 1 } };
    for (list, want) |got, w| try std.testing.expect(got.state.at == w[0] and got.state.name[0] == w[1]);
}
