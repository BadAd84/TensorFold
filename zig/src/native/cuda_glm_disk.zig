//! GLM-5.3-Flash's CUDA kept states on disk (--learn-conversations): every rank's part in rank 0's files; with
//! TF_GLM_OVN_DISK_LOCAL each rank's part in its own files on its own disk (cuda_glm_disk_local).
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const cuda_disk = @import("cuda_disk.zig");
const carry = @import("cuda_glm_disk_carry.zig"); // TF_GLM_OVN_DISK_CARRY
const img = glm.disk_image;
const ranks = glm.disk_ranks;
const rows = api.prompt_cache.rows;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Allocator = std.mem.Allocator;

pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole, .canon = canon, .canon_head = img.Words.bytes };

/// Every rank's prefix bytes, read at attach (one GLM engine a process; rank 0 alone serves).
var prefix_buf: [8]u64 = undefined;
var prefixes: []const u64 = &.{};
/// TF_GLM_OVN_DISK_BCAST, read at attach: a restore's shared bytes go to the other ranks by broadcast from rank 0.
var bcast = false;
/// TF_GLM_OVN_DISK_LOCAL (TP only), from attach: rank 0's store holds rank 0's image alone, copied without the link.
pub var local = false;
/// The prefixes the store's image carries: every rank's, or with TF_GLM_OVN_DISK_LOCAL rank 0's alone.
var store_pre: []const u64 = &.{};

fn engineOf(ptr: *anyopaque) *Engine {
    return @ptrCast(@alignCast(ptr));
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(a);
    try img.storeParts(try ranks.shape(engineOf(ptr)), store_pre, at, &list, a);
    for (list.items) |p| try out.append(a, .{ .bytes = p.bytes, .row = p.row, .per = p.per });
}

/// TF_GLM_OVN_DISK_CARRY's replay: rank 0's image by its words, then each other rank's prefix as rank 0's if as long.
fn canon(ptr: *anyopaque, at: u32, stored: []const u8, fresh: []const u8, out: *std.ArrayList(rows.image.Match), a: Allocator) anyerror!void {
    const sh = try ranks.shape(engineOf(ptr));
    const s = try img.Words.decode(wordsIn(stored), at);
    const f = try img.Words.decode(wordsIn(fresh), at);
    try img.canon(sh, at, s, f, out, a);
    var off = img.imageBytes(sh, at);
    for (store_pre[1..]) |p| { // a prefix of another length lays its fields out otherwise: every byte, in place
        if (p == sh.prefix) try img.prefix(sh, off, s.cur, f.cur, out, a) else try out.append(a, .{ .off = off, .len = p, .fresh = off });
        off += p;
    }
}

fn wordsIn(head: []const u8) [6]u64 {
    var x: [6]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&x), head[0..img.Words.bytes]);
    return x;
}

/// The other ranks asleep before a disk call go back to sleep after it (their lines, not NCCL, wait meanwhile).
pub const Rest = struct {
    e: *Engine,
    asleep: bool,

    pub fn of(e: *Engine) Rest {
        return .{ .e = e, .asleep = ranks.asleep(e) };
    }

    pub fn done(r: Rest) void {
        if (r.asleep) ranks.rest(r.e) catch |err| std.log.warn("glm: the ranks did not sleep after a disk step ({s})", .{@errorName(err)});
    }
};

fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    if (local) return mine(e, k, ranges, out, true);
    var pieces: std.ArrayList(img.Piece) = .empty;
    defer pieces.deinit(e.gpa);
    const sh = try ranks.shape(e);
    var o: usize = 0;
    for (ranges) |g| {
        pieces.clearRetainingCapacity();
        try img.route(sh, prefixes, k.at, g.off, g.len, &pieces, e.gpa);
        for (pieces.items) |p| {
            const n: usize = @intCast(p.len);
            try ranks.take(e, p.rank, k.id, p.off, out[o..][0..n]);
            o += n;
        }
    }
}

/// TF_GLM_OVN_DISK_LOCAL: `ranges` of rank 0's own image of `k` out to `host` or in, through one copier.
fn mine(e: *Engine, k: *const Kept, ranges: []const rows.image.Range, host: []u8, out: bool) !void {
    var end: u64 = 0;
    for (ranges) |g| end = @max(end, g.off + g.len);
    var segs: std.ArrayList(img.Seg) = .empty;
    defer segs.deinit(e.gpa);
    var w: [6]u64 = undefined;
    const c = try ranks.copier(e, k.id, end, &segs, &w);
    var o: usize = 0;
    for (ranges) |g| {
        const n: usize = @intCast(g.len);
        defer o += n;
        // in: the words went with the seal (`scatter`); the rest of the range follows them
        const skip: usize = if (out or g.off >= img.Words.bytes) 0 else @intCast(@min(g.len, img.Words.bytes - g.off));
        if (skip < n) try c.run(g.off + skip, host[o + skip ..][0 .. n - skip], out);
    }
}

/// A new id leased at `at` on every rank, filled by `scatter` and sealed by the words in its last one.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const e = engineOf(ptr);
    const rest = Rest.of(e);
    defer rest.done();
    const k = try e.gpa.create(Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try ranks.blank(e, at), .at = at };
    return @ptrCast(k);
}

/// The last scatter brings the state's own file, words first: every rank seals it then (the words place the head).
fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    if (ranges.len > 0 and ranges[0].off == 0) {
        if (ranges[0].len < img.Words.bytes) return error.DiskWords;
        var x: [6]u64 = undefined;
        @memcpy(std.mem.sliceAsBytes(&x), in[0..img.Words.bytes]);
        const w = try img.Words.decode(x, k.at);
        // TF_GLM_OVN_DISK_LOCAL: rank 0 seals its own; each other rank seals from its own file (cuda_disk_part)
        if (local) try e.sealState(k.id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur) else try ranks.seal(e, k.id, w);
    }
    if (local) return mine(e, k, ranges, @constCast(in), false);
    var pieces: std.ArrayList(img.Piece) = .empty;
    defer pieces.deinit(e.gpa);
    const sh = try ranks.shape(e);
    var o: usize = 0;
    for (ranges) |g| {
        pieces.clearRetainingCapacity();
        try img.route(sh, prefixes, k.at, g.off, g.len, &pieces, e.gpa);
        for (pieces.items) |p| {
            const n: usize = @intCast(p.len);
            const bytes = in[o..][0..n];
            o += n;
            if (p.rank == 0 and p.off < img.Words.bytes) continue; // the words went with the seal
            if (p.shared and bcast) {
                try ranks.share(e, k.id, p.off, bytes, prefixes);
                continue;
            }
            try ranks.put(e, p.rank, k.id, p.off, bytes);
            if (p.shared) for (prefixes[1..], 1..) |q, r| try ranks.put(e, @intCast(r), k.id, p.off - prefixes[0] + q, bytes);
        }
    }
}

fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const k: *Kept = @ptrCast(@alignCast(saved));
    const s = engineOf(ptr).snaps.get(k.id) orelse return false;
    return s.len != 0;
}

/// A fixed prompt's store image hashed, and each other rank's shared bytes checked equal to rank 0's (taken as theirs).
fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, e: *Engine) !struct { hash: u64, rate: f64 } {
    const mark: u32 = glm.native.snap_min_rows; // the smallest kept state; the pass's steps cut at it (Engine.chunkEnd)
    var toks: [glm.native.snap_min_rows + 40]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        e: *Engine,
        saved: ?*anyopaque = null,
        failed: ?anyerror = null,
        fn at(p: *anyopaque, _: *lanes.Stream, m: u32) void {
            const h: *@This() = @ptrCast(@alignCast(p));
            h.saved = glm.native.snap_save(h.e, null, m) catch |err| {
                h.failed = err;
                return;
            };
        }
    };
    var hook: Hook = .{ .e = e };
    const marks = [_]u32{mark};
    var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = e.mtp, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    backend.vtable.prefill(backend.ptr, &s) catch |err| {
        backend.vtable.release(backend.ptr, &s);
        return err;
    };
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    backend.vtable.release(backend.ptr, &s);
    const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
    defer glm.native.snap_drop(e, saved);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const sh = try ranks.shape(e);
    const own = img.imageBytes(sh, mark);
    var total = own;
    for (prefixes[1..]) |p| total += p;
    const bytes = try a.alloc(u8, @intCast(total));
    try gather(e, saved, &.{.{ .off = 0, .len = total }}, bytes);
    const shared = bytes[@intCast(img.Words.bytes + sh.prefix)..@intCast(own)];
    const theirs = try a.alloc(u8, shared.len);
    for (prefixes[1..], 1..) |q, r| {
        try ranks.take(e, @intCast(r), k.id, img.Words.bytes + q, theirs);
        if (!std.mem.eql(u8, shared, theirs)) return error.RanksDiffer;
    }
    try ranks.rest(e);
    return .{ .hash = std.hash.Wyhash.hash(0x676c6d, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// --learn-conversations: every rank's prefix read, the identity probed on this thread, the store attached.
pub fn attach(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, store: *api.prompt_cache.Store) !void {
    const e: *Engine = loaded.target;
    if (e.share.world > prefix_buf.len) return error.TooManyRanks;
    try ranks.prefixes(e, prefix_buf[0..e.share.world]);
    prefixes = prefix_buf[0..e.share.world];
    store_pre = prefixes;
    bcast = cuda.ovn.on("DISK_BCAST");
    const keep_local = cuda.ovn.on("DISK_LOCAL") and e.share.world > 1;
    const sh = try ranks.shape(e);
    var facts: std.ArrayList(u64) = .empty;
    try facts.appendSlice(a, &.{ e.share.world, sh.capacity, sh.ring_rows, sh.tail, @intFromBool(sh.head), @intFromBool(sh.scales), @intFromBool(e.drafter != null) });
    try facts.appendSlice(a, prefixes);
    if (glm.sparse.pk8) try facts.append(a, sh.pooled_row); // TF_GLM_OVN_PK8: its own store, the default's id unchanged
    if (keep_local) try facts.append(a, 0x6c6f63616c); // TF_GLM_OVN_DISK_LOCAL: its own store, rank 0's image alone
    const p = try probe(a, io, loaded.backend, e); // every rank's prefix still: it checks the ranks' shared bytes agree
    if (keep_local) store_pre = prefixes[0..1];
    const id = try cuda_disk.identityOf(a, io, ctx, o.dir, kernels, facts.items, p.hash);
    // TF_GLM_OVN_DISK_CARRY: this start's carry key and proof; an older directory proving them is read instead
    const carried = if (cuda.ovn.on("DISK_CARRY")) try carry.start(a, io, ctx, o, root, loaded.backend, e, facts.items, p.hash, id, store, keep_local) else null;
    const adopted: ?u64 = if (carried) |c| (if (c.choice.adopted) c.choice.id else null) else null;
    const gibs = struct {
        fn bytes(g: f64) u64 {
            return @intFromFloat(g * (1 << 30));
        }
    }.bytes;
    // spill stays off: an eviction would gather on rank 0 mid-pass while the other ranks wait in the gate's all-reduce
    const r = try rows.Rows.open(store.gpa, store.family, .{ .root = root, .identity = id, .cap = gibs(o.learn_gib), .floor = gibs(o.learn_min_free_gib), .daily = gibs(o.learn_daily_gib), .lookahead = store.rules.lookahead, .planned = store.rules.planned, .grid = store.rules.grid, .adopted = adopted });
    r.prefill_rate = p.rate;
    store.disk = r;
    if (keep_local) try @import("cuda_glm_disk_local.zig").open(e, r, gibs(o.learn_min_free_gib));
    if (carried) |c| try carry.records(e, r, c, keep_local);
    std.log.info("glm: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB; {d} ranks' parts{s})", .{ r.dir, r.states.items.len, r.held() >> 20, r.cap >> 20, prefixes.len, if (local) ", each on its own disk" else "" });
}

test {
    std.testing.refAllDecls(@This());
}
