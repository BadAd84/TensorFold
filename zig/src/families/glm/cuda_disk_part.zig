//! TF_GLM_OVN_DISK_LOCAL, a rank other than 0: its own part of each kept state in files on its own disk, under the
//! names rank 0's store gives them (`.r<rank>` after the kind), written behind on one thread in rank 0's order.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const file = lanes.rows_file;
const rimg = lanes.rows_image;
const readers = lanes.disk_readers;
const image = @import("cuda_disk_image.zig");
const ranks = @import("cuda_disk_ranks.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Allocator = std.mem.Allocator;

pub const Stem = [16]u8;

/// A file as every rank keys it: its name's stem and its kind.
pub const Key = struct { stem: Stem, kind: file.Kind };

/// A file rank 0 names: its name, kind and (a block) its group and index.
pub const File = struct { name: file.Name, kind: file.Kind, window: u32 = 0, k: u32 = 0 };

/// One step of the writer, in rank 0's order: a file written, a file removed, another identity's directory removed.
const Job = struct { what: enum { write, unlink, other }, key: Key, name: file.Name = @splat(0), at: u32 = 0, payload: []u8 = &.{}, id: u64 = 0 };

const pages = std.heap.page_allocator;

fn doze() void {
    var t: platform.c.timespec = .{ .sec = 0, .nsec = 200 * std.time.ns_per_us };
    _ = platform.c.nanosleep(&t, null);
}

/// CUDA pinning of read slots: one host-to-device hop from them into the snapshot pages.
pub fn pinOf(e: *Engine) readers.Pin {
    const P = struct {
        fn pin(ctx: *anyopaque, buf: []u8) bool {
            const x: *Engine = @ptrCast(@alignCast(ctx));
            return x.d.api.cuMemHostRegister_v2(buf.ptr, buf.len, 0) == 0;
        }
        fn unpin(ctx: *anyopaque, buf: []u8) void {
            const x: *Engine = @ptrCast(@alignCast(ctx));
            _ = x.d.api.cuMemHostUnregister(buf.ptr);
        }
    };
    return .{ .ctx = e, .pin = P.pin, .unpin = P.unpin };
}

/// This rank's image layout at `at` as the rows store lays a state out (its parts, rows by position).
pub fn layout(e: *const Engine, at: u32, a: Allocator) ![]rimg.Part {
    var list: std.ArrayList(image.Part) = .empty;
    defer list.deinit(a);
    try image.parts(try ranks.shape(e), at, &list, a);
    const out = try a.alloc(rimg.Part, list.items.len);
    for (list.items, out) |p, *x| x.* = .{ .bytes = p.bytes, .row = p.row, .per = p.per };
    return out;
}

/// The image ranges of file `f` of a state at `at` in this rank's layout `parts`, appended to `out`.
pub fn ranges(parts: []const rimg.Part, at: u32, block: u32, f: File, out: *std.ArrayList(rimg.Range), a: Allocator) !void {
    if (f.kind == .block) return rimg.blockRanges(parts, block, f.window, f.k, out, a);
    try rimg.ownRanges(parts, at, block, out, a);
    // the words lead the own file: the state is sealed from them before its head and tail are copied in
    if (out.items.len == 0 or out.items[0].off != 0 or out.items[0].len != image.Words.bytes) return error.SnapshotShape;
}

/// TF_IMPRINT_VERIFY=1 on this rank: whether kept states `a` and `b` at `at` hold the same image (16 MiB a pass).
pub fn same(e: *Engine, a: u64, b: u64, at: u32) !bool {
    const total = image.imageBytes(try ranks.shape(e), at);
    var sa: std.ArrayList(image.Seg) = .empty;
    defer sa.deinit(e.gpa);
    var sb: std.ArrayList(image.Seg) = .empty;
    defer sb.deinit(e.gpa);
    var wa: [6]u64 = undefined;
    var wb: [6]u64 = undefined;
    const ca = try ranks.copier(e, a, total, &sa, &wa);
    const cb = try ranks.copier(e, b, total, &sb, &wb);
    const piece: u64 = 16 << 20;
    const x = try pages.alloc(u8, piece);
    defer pages.free(x);
    const y = try pages.alloc(u8, piece);
    defer pages.free(y);
    var off: u64 = 0;
    while (off < total) : (off += piece) {
        const n: usize = @intCast(@min(piece, total - off));
        try ca.run(off, x[0..n], true);
        try cb.run(off, y[0..n], true);
        if (!std.mem.eql(u8, x[0..n], y[0..n])) return false;
    }
    return true;
}

pub const Part = struct {
    gpa: Allocator,
    dir: [:0]u8,
    rank: u32,
    floor: u64, // --learn-min-free-gib: below it this rank writes nothing (its parts missing, rank 0 forgets the state)
    block: u32,
    threads: u32,
    seen: std.AutoHashMapUnmanaged(Key, void) = .empty, // startup: the files rank 0 asked about
    ok: bool = false, // the last read's outcome, until rank 0 asks for it (none read: no)
    stage: readers.Staging = .{},
    lock: std.atomic.Mutex = .unlocked,
    jobs: std.ArrayList(Job) = .empty, // under `lock`, taken from `head`
    head: usize = 0,
    busy: bool = false, // under `lock`: the writer runs a job it took
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    skipped: std.atomic.Value(u64) = .init(0),

    /// This rank's store under `dir` (made 0700 when missing), its writer started.
    pub fn open(gpa: Allocator, dir: []const u8, rank: u32, floor: u64, block: u32, threads: u32) !*Part {
        const p = try gpa.create(Part);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .dir = try gpa.dupeSentinel(u8, dir, 0), .rank = rank, .floor = floor, .block = block, .threads = threads };
        errdefer gpa.free(p.dir);
        try makePath(p.dir);
        p.thread = try std.Thread.spawn(.{}, run, .{p});
        return p;
    }

    /// Every queued write finished, then the writer stopped and everything freed.
    pub fn close(p: *Part) void {
        p.drain();
        p.stop.store(true, .release);
        if (p.thread) |t| t.join();
        for (p.jobs.items[p.head..]) |j| pages.free(j.payload);
        p.jobs.deinit(p.gpa);
        p.seen.deinit(p.gpa);
        p.stage.deinit();
        p.gpa.free(p.dir);
        p.gpa.destroy(p);
    }

    fn take(p: *Part) void {
        while (!p.lock.tryLock()) std.Thread.yield() catch {};
    }

    fn push(p: *Part, j: Job) !void {
        p.take();
        defer p.lock.unlock();
        try p.jobs.append(p.gpa, j);
    }

    fn pending(p: *Part) bool {
        p.take();
        defer p.lock.unlock();
        return p.head < p.jobs.items.len or p.busy;
    }

    /// Until every job queued before now is done (a read or a check sees every file rank 0 asked for before it).
    pub fn drain(p: *Part) void {
        while (p.pending()) doze();
    }

    fn run(p: *Part) void {
        while (true) {
            p.take();
            if (p.head < p.jobs.items.len) {
                const j = p.jobs.items[p.head];
                p.head += 1;
                if (p.head == p.jobs.items.len) {
                    p.jobs.clearRetainingCapacity();
                    p.head = 0;
                }
                p.busy = true;
                p.lock.unlock();
                p.exec(j);
                p.take();
                p.busy = false;
                p.lock.unlock();
                continue;
            }
            p.lock.unlock();
            if (p.stop.load(.acquire)) return;
            doze();
        }
    }

    /// Rank `rank`'s path of a file of this store.
    pub fn path(p: *const Part, buf: []u8, k: Key, part: bool) ![:0]const u8 {
        var eb: [24]u8 = undefined;
        const ext = try file.rankExt(&eb, if (k.kind == .block) ".blk" else ".st", p.rank);
        return file.path(buf, p.dir, &std.fmt.bytesToHex(k.stem, .lower), ext, part);
    }

    fn exec(p: *Part, j: Job) void {
        defer pages.free(j.payload);
        var buf: [1200]u8 = undefined;
        switch (j.what) {
            .write => p.write(j) catch |err| std.log.warn("glm rank {d}: writing its part of a kept state failed ({s})", .{ p.rank, @errorName(err) }),
            .unlink => _ = platform.c.unlink(p.path(&buf, j.key, false) catch return),
            .other => lanes.learned_dirs.removeOther(lanes.learned_dirs.rootOf(p.dir), p.dir, j.id) catch {},
        }
    }

    fn write(p: *Part, j: Job) !void {
        const free = lanes.learned_disk.available(p.dir) orelse 0;
        if (free < p.floor +| j.payload.len +| @sizeOf(file.Trailer)) {
            if (p.skipped.fetchAdd(1, .acq_rel) == 0) std.log.warn("glm rank {d}: under the free-disk floor; its parts are not written", .{p.rank});
            return;
        }
        var eb: [24]u8 = undefined;
        const ext = try file.rankExt(&eb, if (j.key.kind == .block) ".blk" else ".st", p.rank);
        var w = try file.Writer.open(p.dir, &std.fmt.bytesToHex(j.key.stem, .lower), ext);
        errdefer w.abort();
        try w.append(j.payload);
        _ = try w.finish(.{ .kind = @backingInt(j.key.kind), .payload = 0, .payload_hash = 0, .name = j.name, .at = j.at, .block = p.block }, &.{}, true);
    }

    /// Rank 0 staged file `f` of kept state `id` at `at`: this rank's part of it copied out now, written behind.
    pub fn save(p: *Part, e: *Engine, id: u64, at: u32, f: File) !void {
        const parts = try layout(e, at, p.gpa);
        defer p.gpa.free(parts);
        var rs: std.ArrayList(rimg.Range) = .empty;
        defer rs.deinit(p.gpa);
        try ranges(parts, at, p.block, f, &rs, p.gpa);
        const payload = try pages.alloc(u8, @intCast(rimg.total(rs.items)));
        errdefer pages.free(payload);
        var end: u64 = 0;
        for (rs.items) |r| end = @max(end, r.off + r.len);
        var segs: std.ArrayList(image.Seg) = .empty;
        defer segs.deinit(p.gpa);
        var w: [6]u64 = undefined;
        const c = try ranks.copier(e, id, end, &segs, &w);
        var o: usize = 0;
        for (rs.items) |r| {
            try c.run(r.off, payload[o..][0..@intCast(r.len)], true);
            o += @intCast(r.len);
        }
        try p.push(.{ .what = .write, .key = .{ .stem = f.name[0..16].*, .kind = f.kind }, .name = f.name, .at = if (f.kind == .state) at else 0, .payload = payload });
    }

    pub fn unlink(p: *Part, k: Key) !void {
        try p.push(.{ .what = .unlink, .key = k });
    }

    pub fn other(p: *Part, id: u64) !void {
        try p.push(.{ .what = .other, .key = undefined, .id = id });
    }

    /// Whether this rank holds a checked trailer of `k` (startup; nothing is queued yet).
    pub fn has(p: *const Part, k: Key) bool {
        var buf: [1200]u8 = undefined;
        var o = file.open(p.path(&buf, k, false) catch return false, k.kind, null) catch return false;
        defer o.close();
        return std.mem.eql(u8, o.trailer.name[0..16], &k.stem) and o.trailer.block == p.block;
    }

    /// Startup, after rank 0's questions: every file of this rank's no question named goes, and every partial one.
    pub fn sweep(p: *Part) void {
        p.drain();
        defer p.seen.clearAndFree(p.gpa);
        var tails: [2][24]u8 = undefined;
        const blk = file.rankExt(&tails[0], ".blk", p.rank) catch return;
        const st = file.rankExt(&tails[1], ".st", p.rank) catch return;
        const d = platform.c.opendir(p.dir) orelse return;
        defer _ = platform.c.closedir(d);
        while (platform.c.readdir(d)) |ent| {
            if (ent.type != platform.c.DT.REG) continue;
            const name = std.mem.sliceTo(&ent.name, 0);
            const whole = if (std.mem.endsWith(u8, name, ".part")) name[0 .. name.len - 5] else name;
            const kind: file.Kind = if (std.mem.endsWith(u8, whole, blk)) .block else if (std.mem.endsWith(u8, whole, st)) .state else continue;
            var stem: Stem = undefined;
            const named = whole.len >= 32 and (std.fmt.hexToBytes(&stem, whole[0..32]) catch null) != null;
            // a partial write, a name that is not one, or a file no question named: none of them rank 0 holds
            if (whole.len == name.len and named and p.seen.contains(.{ .stem = stem, .kind = kind })) continue;
            var buf: [1400]u8 = undefined;
            _ = platform.c.unlink(std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ p.dir, name }, 0) catch continue);
        }
    }

    /// Rank 0's restore of `files` (blocks, then the own file) into blank `id` at `at`: this rank's parts read on its
    /// reader threads and copied in here; `ok` records the outcome and a file that fails its check goes.
    pub fn read(p: *Part, e: *Engine, id: u64, at: u32, files: []const File) void {
        p.ok = false;
        p.readAll(e, id, at, files) catch |err| std.log.warn("glm rank {d}: reading its part of {d} disk tokens failed ({s})", .{ p.rank, at, @errorName(err) });
    }

    fn readAll(p: *Part, e: *Engine, id: u64, at: u32, files: []const File) !void {
        if (files.len == 0 or files[files.len - 1].kind != .state) return error.BadStep;
        p.drain();
        const parts = try layout(e, at, p.gpa);
        defer p.gpa.free(parts);
        var rs: std.ArrayList(rimg.Range) = .empty;
        defer rs.deinit(p.gpa);
        const firsts = try p.gpa.alloc(usize, files.len + 1);
        defer p.gpa.free(firsts);
        var slot: u64 = 0;
        for (files, 0..) |f, i| {
            firsts[i] = rs.items.len;
            try ranges(parts, at, p.block, f, &rs, p.gpa);
            if (f.kind == .block) slot = @max(slot, rimg.total(rs.items[firsts[i]..]));
        }
        firsts[files.len] = rs.items.len;
        if (p.stage.pin == null) p.stage.pin = pinOf(e);
        try p.stage.ensure(slot, 2 * @as(usize, p.threads), rimg.total(rs.items[firsts[files.len - 1]..]));
        const sh = try ranks.shape(e);
        var segs: std.ArrayList(image.Seg) = .empty;
        defer segs.deinit(p.gpa);
        var w: [6]u64 = undefined;
        var c: Reader = .{ .p = p, .e = e, .id = id, .at = at, .files = files, .rs = rs.items, .firsts = firsts, .rows = try ranks.copier(e, id, image.Words.bytes + image.front(sh, at), &segs, &w) };
        var failed: ?usize = null;
        readers.pipe(p.threads, files.len, files.len - 1, &p.stage, &c, &failed) catch |err| {
            if (failed) |k| p.unlink(.{ .stem = files[k].name[0..16].*, .kind = files[k].kind }) catch {};
            return err;
        };
        p.ok = true;
    }
};

/// One restore's files as the reader pool takes them: read and checked on reader threads, copied in on this one.
const Reader = struct {
    p: *Part,
    e: *Engine,
    id: u64,
    at: u32,
    files: []const File,
    rs: []const rimg.Range,
    firsts: []const usize,
    rows: image.Copy, // into the blank's rows before its head (every block's ranges sit there)

    fn of(c: *const Reader, i: usize) []const rimg.Range {
        return c.rs[c.firsts[i]..c.firsts[i + 1]];
    }

    pub fn span(c: *Reader, i: usize) u64 {
        return std.mem.alignForward(u64, rimg.total(c.of(i)), file.alignment);
    }

    pub fn read(c: *Reader, i: usize, into: file.Stage) ![]const u8 {
        const f = c.files[i];
        var buf: [1200]u8 = undefined;
        var o = try file.open(try c.p.path(&buf, .{ .stem = f.name[0..16].*, .kind = f.kind }, false), f.kind, f.name);
        defer o.close();
        const at: u32 = if (f.kind == .state) c.at else 0;
        if (o.trailer.payload != rimg.total(c.of(i)) or o.trailer.block != c.p.block or o.trailer.at != at) return error.RowsRead;
        return file.payload(&o, into);
    }

    pub fn consume(c: *Reader, i: usize, payload: []const u8) !void {
        var rs = c.of(i);
        var o: usize = 0;
        var copy = c.rows;
        var segs: std.ArrayList(image.Seg) = .empty;
        defer segs.deinit(c.p.gpa);
        var w: [6]u64 = undefined;
        if (c.files[i].kind == .state) { // sealed from its words first: the head and tail follow it
            var x: [6]u64 = undefined;
            @memcpy(std.mem.sliceAsBytes(&x), payload[0..image.Words.bytes]);
            const words = try image.Words.decode(x, c.at);
            try c.e.sealState(c.id, words.mtp_len, .{ words.base, words.mtp_base }, words.cur);
            var end: u64 = 0;
            for (rs) |r| end = @max(end, r.off + r.len);
            copy = try ranks.copier(c.e, c.id, end, &segs, &w);
            rs = rs[1..];
            o = image.Words.bytes;
        }
        for (rs) |r| {
            try copy.run(r.off, @constCast(payload[o..][0..@intCast(r.len)]), false);
            o += @intCast(r.len);
        }
    }
};

/// `dir` and its parents, 0700.
fn makePath(dir: [:0]u8) !void {
    for (dir[1..], 1..) |ch, i| if (ch == '/') {
        dir[i] = 0;
        _ = platform.c.mkdir(dir.ptr, 0o700);
        dir[i] = '/';
    };
    const rc = platform.c.mkdir(dir.ptr, 0o700);
    if (rc != 0 and platform.c.errno(rc) != .EXIST) return error.RowsDir;
}

test "glm disk part: a rank's files carry its rank's names, read back checked, and go unless rank 0 asked for them" {
    const a = std.testing.allocator;
    const tmp = if (platform.c.getenv("TMPDIR")) |t| std.mem.span(t) else ".";
    var rb: [256]u8 = undefined;
    const root = try std.fmt.bufPrintSentinel(&rb, "{s}/tf-glm-part-{d}", .{ tmp, platform.c.getpid() }, 0);
    var db: [300]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&db, "{s}/0123456789abcdef", .{root}, 0);
    const p = try Part.open(a, dir, 2, 0, 256, 2);
    defer { // nothing asked: every file of this rank goes, then its directories
        p.seen.clearRetainingCapacity();
        p.sweep();
        p.close();
        _ = platform.c.rmdir(dir);
        _ = platform.c.rmdir(root);
    }
    const kept: file.Name = @splat(0x5a);
    const lost: file.Name = @splat(0x3c);
    for ([_]file.Name{ kept, lost }) |n| {
        const payload = try pages.alloc(u8, 5000);
        for (payload, 0..) |*b, i| b.* = @truncate(i *% 13 +% n[0]);
        try p.push(.{ .what = .write, .key = .{ .stem = n[0..16].*, .kind = .block }, .name = n, .payload = payload });
    }
    p.drain();
    const k: Key = .{ .stem = kept[0..16].*, .kind = .block };
    const l: Key = .{ .stem = lost[0..16].*, .kind = .block };
    try std.testing.expect(p.has(k) and p.has(l));
    try std.testing.expect(!p.has(.{ .stem = k.stem, .kind = .state }));
    var buf: [400]u8 = undefined;
    // rank 2's file is rank 0's name with `.r2` after it; nothing under rank 0's own name
    try std.testing.expect(std.mem.endsWith(u8, try p.path(&buf, k, false), ".blk.r2"));
    var r0: [400]u8 = undefined;
    try std.testing.expect(platform.c.access(try file.path(&r0, dir, &std.fmt.bytesToHex(k.stem, .lower), ".blk", false), 0) != 0);
    // a leftover partial write of this rank goes in the sweep, as does a file rank 0 never asked about
    const part = try p.path(&buf, .{ .stem = @splat(0x11), .kind = .state }, true);
    const fd = platform.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(platform.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    _ = platform.c.close(fd);
    try p.seen.put(a, k, {});
    p.sweep();
    try std.testing.expect(p.has(k));
    try std.testing.expect(!p.has(l));
    try std.testing.expect(platform.c.access(part, 0) != 0);
    try p.unlink(k);
    p.drain();
    try std.testing.expect(!p.has(k));
}
