//! Conversation states on disk (--learn-conversations): rows in blocks states share, each state's own parts in a file.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
pub const image = @import("lanes").rows_image;
pub const chain = @import("rows_chain.zig");
pub const file = @import("lanes").rows_file;
pub const write = @import("rows_write.zig");
pub const read = @import("rows_read.zig");
pub const local = @import("rows_local.zig");
pub const carry = @import("rows_carry.zig"); // TF_GLM_OVN_DISK_CARRY: an older identity's directory, read on proof
pub const replay = @import("rows_replay.zig"); // TF_GLM_OVN_DISK_CARRY: stored states prefilled again and compared
const dirs = @import("lanes").learned_dirs;
const readers = @import("lanes").disk_readers;
const Admission = @import("lanes").learned_disk.Admission;
const Allocator = std.mem.Allocator;
const Name = chain.Name;
const Span = modes.Span;

/// Tokens a block holds: 256 rows of every row part (the 27B's keys and values: 16 MiB).
pub const block_tokens = 256;

/// What a family gives the store: its kept states' image as parts, and copies out of and into kept states.
pub const Image = struct {
    /// A state's image at `at`, its parts in order, appended to `out`.
    parts: *const fn (ptr: *anyopaque, at: u32, out: *std.ArrayList(image.Part), a: Allocator) anyerror!void,
    /// Copy into owned `out`; all GPU producers and copies must complete before returning.
    gather: *const fn (ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, out: []u8) anyerror!void,
    /// A new kept state of `at` tokens, its image filled by `scatter` and then restored as any other.
    blank: *const fn (ptr: *anyopaque, at: u32) anyerror!pc.Saved,
    /// Copy `in`, back to back, into `ranges` of `saved`'s image.
    scatter: *const fn (ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, in: []const u8) anyerror!void,
    /// Whether `saved`'s image can still be read whole (null: always).
    whole: ?*const fn (ptr: *anyopaque, saved: pc.Saved) bool = null,
};

/// A state on disk: its name and position, what it read, its file's bytes and its blocks' (each counted whole).
pub const State = struct {
    name: Name,
    at: u32,
    tokens: []u32, // position plus lookahead
    spans: []Span, // decoded spans through `at`
    starts: []u32, // a planned family's chunk starts below `at`
    own: u64,
    rows: u64,
    used: u64,
    shared: bool,
};

/// A block file's bytes, and how many states name it.
pub const Block = struct { bytes: u64, refs: u32 = 0 };

pub const Counts = struct { written: u64 = 0, read: u64 = 0, reads: u64 = 0, writes: u64 = 0, skipped: u64 = 0, failed: u64 = 0, evicted: u64 = 0, verified: u64 = 0, differed: u64 = 0 };

/// --spill-gib: off (every keep written behind), evicted (evictions and close), finish (both: evictions finish).
pub const Spill = enum { off, evicted, finish };

pub const Options = struct {
    root: []const u8,
    identity: u64, // the family's identity (weights, arithmetic, build); the store's directory is its own hash of it
    cap: u64, // --learn-gib: bytes under the root, every identity together
    floor: u64, // --learn-min-free-gib
    daily: u64, // --learn-daily-gib: bytes written a day under the root
    lookahead: u32 = 0,
    planned: bool = false,
    grid: u32 = 0, // a planned family's zero-anchored chunk rows: its starts where a request names none
    spill: Spill = .off,
    sync_writes: bool = false,
    staging_bytes: u64 = 1 << 30,
    slots: ?usize = null, // --checkpoint-slots: the RAM store's states at most, beside this one (null: the family's)
    adopted: ?u64 = null, // TF_GLM_OVN_DISK_CARRY: the older directory read and written instead (rows_carry.choose)
};

/// The store under `o`, the RAM store's disk: `o.slots` bounds what it keeps in RAM.
pub fn attach(gpa: Allocator, store: *pc.Store, o: Options) !*Rows {
    const r = try Rows.open(gpa, store.family, o);
    store.disk = r;
    if (o.slots) |n| store.most = n;
    return r;
}

/// Where --spill-gib's states live: $HOME/.cache/tensorfold/spilled, beside the learned states' root.
pub fn spillRoot(a: Allocator) ![]const u8 {
    const home = platform.c.getenv("HOME") orelse return error.RowsDir;
    return std.fmt.allocPrint(a, "{s}/.cache/tensorfold/spilled", .{std.mem.span(home)});
}

pub const Rows = struct {
    gpa: Allocator,
    family: pc.Snapshots, // the kept states' family (its drop frees a state read back that failed)
    img: *const Image,
    root: []u8,
    dir: [:0]u8,
    keying: chain.Keying,
    grid: u32 = 0,
    cap: u64,
    daily: u64,
    admission: Admission = .{},
    states: std.ArrayList(State) = .empty,
    blocks: std.AutoHashMapUnmanaged(chain.Stem, Block) = .empty,
    queue: std.ArrayList(write.Queued) = .empty, // kept states waiting to be written, newest last (written first)
    job: ?*write.Job = null, // the synchronous writer's borrowed entry
    stage: file.Stage = &.{},
    clock: u64 = 0,
    read_rate: f64 = 2e9, // bytes a second, measured as states are read
    prefill_rate: f64 = 0, // the family's prompt tokens a second (0: none measured yet)
    counts: Counts = .{},
    turn: u64 = 0, // bytes written since the last report line
    verify: bool = false, // TF_IMPRINT_VERIFY=1: each state written is read back and compared with the RAM state
    spill: Spill = .off,
    file_io: file.IO = .{},
    worker: ?*write.io.Worker = null,
    staging_bytes: u64 = 1 << 30,
    sync_writes: bool = false,
    stopped: bool = false,
    flush_clock: ?u64 = null, // a close protects states touched after this clock from older queued saves
    next_job: u64 = 0,
    waiting: ?*pc.Entry = null, // the queued save the jobs in flight left no room for (a reap lets it try again)
    failures: write.Failures = .{},
    fetch: ?*@import("rows_prefetch.zig").Prefetch = null, // states read ahead for queued requests (a lane host's)
    pump: ?*@import("rows_pump.zig").Pump = null, // TF_GLM_OVN_DISK_PUMP: the job whose payloads stage a few a round
    pump_blocks: u32 = 0, // TF_GLM_OVN_DISK_PUMP: pieces staged a round (0: a state staged in one step)
    pump_auto: bool = false, // DISK_LOCAL's default pump: only a save wider than the staging permit goes by it
    mirror: ?*const @import("rows_local.zig").Mirror = null, // TF_GLM_OVN_DISK_LOCAL: the other ranks' own parts
    readers: u32 = 1, // TF_GLM_OVN_DISK_READERS: threads a restore reads its files on (1: one file after another)
    readers_stage: readers.Staging = .{}, // their slots, made by the first read that uses them

    /// The store under `o.root` for `o.identity` (made 0700 when missing); its files are read, unreferenced blocks go.
    pub fn open(gpa: Allocator, family: pc.Snapshots, o: Options) !*Rows {
        const r = try init(gpa, family, o);
        errdefer r.deinit();
        try makePath(gpa, r.dir);
        try @import("rows_scan.zig").scan(r);
        @import("rows_scan.zig").settle(r);
        r.stamp();
        r.sync_writes = o.sync_writes or (if (platform.c.getenv("TENSORFOLD_ROWS_SYNC")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false);
        r.staging_bytes = stagingBytes(o.staging_bytes, if (platform.c.getenv("TENSORFOLD_ROWS_STAGING_GIB")) |v| std.mem.span(v) else null);
        r.next_job = @truncate(@as(u128, @intCast(write.nowNs())));
        r.verify = if (platform.c.getenv("TF_IMPRINT_VERIFY")) |v| v[0] == '1' else false;
        r.pump_blocks = try @import("rows_pump.zig").knob();
        r.readers = try readers.knob(1);
        return r;
    }

    /// A store holding only its names, nothing read or made (rows_replay views a directory so); `deinit` alone owns it.
    pub fn init(gpa: Allocator, family: pc.Snapshots, o: Options) !*Rows {
        const img = family.vtable.rows orelse return error.RowsUnsupported;
        const r = try gpa.create(Rows);
        errdefer gpa.destroy(r);
        const id = o.adopted orelse carry.dirOf(o.identity); // an adopted directory's names stay keyed by its own id
        const root = try gpa.dupe(u8, o.root);
        errdefer gpa.free(root);
        const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ o.root, id }, 0);
        r.* = .{ .gpa = gpa, .family = family, .img = img, .root = root, .dir = dir, .keying = .{ .identity = id, .block = block_tokens, .lookahead = o.lookahead, .planned = o.planned }, .grid = o.grid, .cap = o.cap, .daily = o.daily, .admission = .{ .floor = o.floor }, .spill = o.spill };
        return r;
    }

    pub fn deinit(r: *Rows) void {
        write.stop(r, write.nowNs());
        if (r.fetch) |f| f.destroy();
        if (r.job) |j| write.abort(r, j);
        for (r.queue.items) |q| r.gpa.free(q.starts);
        for (r.states.items) |*s| freeState(r.gpa, s);
        r.states.deinit(r.gpa);
        r.blocks.deinit(r.gpa);
        r.queue.deinit(r.gpa);
        if (r.stage.len > 0) r.gpa.free(r.stage);
        r.readers_stage.deinit();
        r.gpa.free(r.dir);
        r.gpa.free(r.root);
        r.gpa.destroy(r);
    }

    pub fn freeState(gpa: Allocator, s: *State) void {
        gpa.free(s.tokens);
        gpa.free(s.spans);
        gpa.free(s.starts);
    }

    /// A staging buffer of at least `bytes` (kept between steps).
    pub fn staged(r: *Rows, bytes: u64) !file.Stage {
        if (r.stage.len >= bytes) return r.stage;
        if (r.stage.len > 0) r.gpa.free(r.stage);
        r.stage = &.{};
        r.stage = try file.stage(r.gpa, bytes);
        return r.stage;
    }

    /// A prompt pass of `tokens` in `ns` moves the rate a disk read is weighed against (under 1,024 tokens: none).
    pub fn prefilled(r: *Rows, tokens: usize, ns: i96) void {
        if (tokens < 1024 or ns <= 0) return;
        const rate = @as(f64, @floatFromInt(tokens)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        r.prefill_rate = if (r.prefill_rate > 0) 0.5 * r.prefill_rate + 0.5 * rate else rate;
    }

    /// Every byte the store holds: its states' files and its blocks.
    pub fn held(r: *const Rows) u64 {
        var n: u64 = 0;
        for (r.states.items) |s| n += s.own;
        var it = r.blocks.valueIterator();
        while (it.next()) |b| n += b.bytes;
        return n;
    }

    /// A state's image parts at `at`, from its family (a's memory).
    pub fn parts(r: *const Rows, a: Allocator, at: u32) ![]image.Part {
        var list: std.ArrayList(image.Part) = .empty;
        errdefer list.deinit(a);
        try r.img.parts(r.family.ptr, at, &list, a);
        if (!image.valid(list.items, at, r.keying.block)) return error.RowsLayout;
        return list.toOwnedSlice(a);
    }

    /// A state's whole blocks (names, groups, bytes), in `out`.
    pub fn wantsOf(r: *const Rows, a: Allocator, s: *const State, out: *std.ArrayList(write.Want)) !void {
        const ps = try r.parts(a, s.at);
        defer a.free(ps);
        try r.blockNames(a, ps, s.tokens, s.spans, s.starts, s.at, out);
    }

    pub fn find(r: *const Rows, name: Name) ?usize {
        for (r.states.items, 0..) |s, i| if (std.mem.eql(u8, &s.name, &name)) return i;
        return null;
    }

    /// The names of a state's whole blocks, every group, in `out` (a's memory).
    pub fn blockNames(r: *const Rows, a: Allocator, layout: []const image.Part, tokens: []const u32, spans: []const Span, starts: []const u32, at: u32, out: *std.ArrayList(write.Want)) !void {
        var gbuf: [8]u32 = undefined;
        for (try image.groups(layout, &gbuf)) |g| {
            const w = image.wholeBlocks(at, r.keying.block, g);
            if (w.end == 0) continue;
            const names = try a.alloc(Name, w.end);
            defer a.free(names);
            const planned_starts = try withAt(a, starts, at, r.keying.planned);
            defer a.free(planned_starts);
            chain.blocks(r.keying, tokens, spans, planned_starts, at, g, w.end, names);
            for (names[w.first..w.end], w.first..) |n, k| try out.append(a, .{ .name = n, .window = g, .k = @intCast(k), .bytes = image.blockBytes(layout, r.keying.block, g) + @sizeOf(file.Trailer) });
        }
    }

    /// A state's block names named again, each block's refs moved by `delta`.
    pub fn ref(r: *Rows, wants: []const write.Want, delta: i2) void {
        for (wants) |w| if (r.blocks.getPtr(chain.stem(w.name))) |b| {
            if (delta > 0) b.refs += 1 else b.refs -|= 1;
        };
    }

    /// State `i` forgotten: its file, and every block no other state names.
    pub fn remove(r: *Rows, i: usize) void {
        write.fence(r);
        var s = r.states.orderedRemove(i);
        defer freeState(r.gpa, &s);
        r.unlink(s.name, .state);
        var wants: std.ArrayList(write.Want) = .empty;
        defer wants.deinit(r.gpa);
        r.wantsOf(r.gpa, &s, &wants) catch {};
        r.ref(wants.items, -1);
        r.collect();
    }

    /// Blocks no state names are deleted (a write that lost its entry, or the last state that named them).
    pub fn collect(r: *Rows) void {
        var gone: std.ArrayList(chain.Stem) = .empty;
        defer gone.deinit(r.gpa);
        var it = r.blocks.iterator();
        while (it.next()) |kv| if (kv.value_ptr.refs == 0 and !write.wanted(r, kv.key_ptr.*)) gone.append(r.gpa, kv.key_ptr.*) catch {};
        if (gone.items.len > 0) write.fence(r);
        for (gone.items) |n| {
            r.unlink(n, .block);
            _ = r.blocks.remove(n);
        }
    }

    /// A file of the store removed (`n` a Name or a Stem), and with TF_GLM_OVN_DISK_LOCAL every other rank's part.
    pub fn unlink(r: *Rows, n: anytype, kind: file.Kind) void {
        unlinkName(r.dir, n, if (kind == .block) ".blk" else ".st");
        if (r.mirror) |m| m.unlink(m.ctx, .{ .stem = n[0..16].*, .kind = kind });
    }

    /// The disk state reaching furthest into `prompt` past `longer_than`: its tokens a prefix, spans and starts equal.
    pub fn best(r: *const Rows, prompt: []const u32, starts: []const u32, spans: []const Span, longer_than: u32) ?usize {
        var out: ?usize = null;
        for (r.states.items, 0..) |*s, i| {
            if (s.at <= longer_than) continue;
            if (out != null and s.at <= r.states.items[out.?].at) continue;
            if (!r.matches(s, prompt, starts, spans)) continue;
            out = i;
        }
        return out;
    }

    pub fn matches(r: *const Rows, s: *const State, prompt: []const u32, starts: []const u32, spans: []const Span) bool {
        if (s.tokens.len > prompt.len or s.at >= prompt.len) return false;
        if (!std.mem.eql(u32, prompt[0..s.tokens.len], s.tokens) or !modes.equal(s.spans, spans, s.at)) return false;
        if (r.keying.planned and starts.len == 0 and r.grid > 0) return s.at % r.grid == 0 and onGrid(s.starts, s.at, r.grid);
        return !r.keying.planned or sameBelow(starts, s.starts, s.at);
    }

    /// State `i` was used now: the newest for eviction.
    pub fn touch(r: *Rows, i: usize) void {
        r.clock += 1;
        r.states.items[i].used = r.clock;
    }

    /// Room for `bytes` more under the cap: other identities first (least recently opened), then this one's by value.
    pub fn room(r: *Rows, bytes: u64) bool {
        const others = dirs.totalOthers(r.root, r.dir) catch return false;
        var over = (others + r.held() + bytes) -| r.cap;
        if (over == 0) return true;
        const Old = struct { id: u64, used: u64 };
        var olds: std.ArrayList(Old) = .empty;
        defer olds.deinit(r.gpa);
        var cursor: ?u64 = null;
        while (dirs.next(r.root, r.dir, cursor) catch null) |id| {
            olds.append(r.gpa, .{ .id = id, .used = dirs.used(r.root, id) catch 0 }) catch return false;
            cursor = id;
        }
        std.mem.sort(Old, olds.items, {}, struct {
            fn older(_: void, x: Old, y: Old) bool {
                return x.used < y.used;
            }
        }.older);
        for (olds.items) |o| {
            if (over == 0) break;
            const b = dirs.otherBytes(r.root, r.dir, o.id) catch continue;
            write.fence(r);
            dirs.removeOther(r.root, r.dir, o.id) catch continue;
            if (r.mirror) |m| m.other(m.ctx, o.id);
            over -|= b;
        }
        while (over > 0) {
            const i = r.victim() orelse return false;
            const before = r.held();
            r.remove(i);
            over -|= before - r.held();
            r.counts.evicted += 1;
        }
        return true;
    }

    /// The state worth least: tokens saved per byte only it holds (its file, blocks no other names), over 1 + its age.
    fn victim(r: *const Rows) ?usize {
        var out: ?usize = null;
        if (r.spill == .evicted) { // --spill-gib alone: the least recently used first
            for (r.states.items, 0..) |s, i| {
                if (r.flush_clock) |cut| if (s.used > cut) continue;
                if (out == null or s.used < r.states.items[out.?].used) out = i;
            }
            return out;
        }
        var worst: f64 = std.math.inf(f64);
        var wants: std.ArrayList(write.Want) = .empty;
        defer wants.deinit(r.gpa);
        for (r.states.items, 0..) |*s, i| {
            if (r.flush_clock) |cut| if (s.used > cut) continue;
            wants.clearRetainingCapacity();
            r.wantsOf(r.gpa, s, &wants) catch {};
            var only = s.own;
            for (wants.items) |w| if (r.blocks.get(chain.stem(w.name))) |b| {
                if (b.refs <= 1) only += b.bytes;
            };
            const age: f64 = @floatFromInt(r.clock -| s.used);
            const v = @as(f64, @floatFromInt(s.at)) / @as(f64, @floatFromInt(@max(only, 1))) / (1 + age);
            if (v < worst) {
                worst = v;
                out = i;
            }
        }
        return out;
    }

    /// Whether `bytes` more fit today's write budget (root/written: the day and its bytes, every identity together).
    pub fn budget(r: *const Rows, bytes: u64) bool {
        const w = r.written();
        return w.bytes + bytes <= r.daily;
    }

    const Day = struct { day: u64, bytes: u64 };

    fn written(r: *const Rows) Day {
        const today = dayNow();
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/written", .{r.root}, 0) catch return .{ .day = today, .bytes = std.math.maxInt(u64) };
        const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
        if (fd < 0) return .{ .day = today, .bytes = 0 };
        defer _ = platform.c.close(fd);
        var d: Day = undefined;
        if (platform.c.pread(fd, std.mem.asBytes(&d), @sizeOf(Day), 0) != @sizeOf(Day) or d.day != today) return .{ .day = today, .bytes = 0 };
        return d;
    }

    /// `bytes` written now count against today's budget.
    pub fn spent(r: *Rows, bytes: u64) void {
        r.counts.written += bytes;
        r.turn += bytes;
        var d = r.written();
        d.bytes +|= bytes;
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/written", .{r.root}, 0) catch return;
        const fd = platform.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
        if (fd < 0) return;
        defer _ = platform.c.close(fd);
        _ = platform.c.pwrite(fd, std.mem.asBytes(&d), @sizeOf(Day), 0);
    }

    /// Every state and block of this identity forgotten (DELETE /v1/imprint); the queue too. The bytes freed.
    pub fn forgetAll(r: *Rows) u64 {
        write.fence(r);
        if (r.job) |j| write.abort(r, j);
        for (r.queue.items) |q| r.gpa.free(q.starts);
        r.queue.clearRetainingCapacity();
        const before = r.held();
        for (r.states.items) |*s| {
            r.unlink(s.name, .state);
            freeState(r.gpa, s);
        }
        r.states.clearRetainingCapacity();
        var it = r.blocks.keyIterator();
        while (it.next()) |n| r.unlink(n.*, .block);
        r.blocks.clearRetainingCapacity();
        return before;
    }

    /// This identity was opened (and wrote): the newest under the root for every identity's sweep.
    pub fn stamp(r: *const Rows) void {
        var path_buf: [1100]u8 = undefined;
        const clock_path = std.fmt.bufPrintSentinel(&path_buf, "{s}/clock", .{r.root}, 0) catch return;
        const now = readWord(clock_path) +| 1;
        writeWord(clock_path, now);
        var used: [1100]u8 = undefined;
        writeWord(std.fmt.bufPrintSentinel(&used, "{s}/used", .{r.dir}, 0) catch return, now);
    }
};

/// Whether a prompt's chunk starts below `at` are the state's, and `at` is one of them.
fn sameBelow(starts: []const u32, kept: []const u32, at: u32) bool {
    var n: usize = 0;
    while (n < starts.len and starts[n] < at) n += 1;
    return std.mem.eql(u32, starts[0..n], kept) and std.mem.indexOfScalar(u32, starts, at) != null;
}

/// A planned request's chunk starts below `at`: its own, else its family's grid's (from `grid`) when it names none.
pub fn startsBelow(r: *const Rows, a: Allocator, starts: []const u32, at: u32) ![]u32 {
    if (!r.keying.planned) return a.alloc(u32, 0);
    if (starts.len == 0 and r.grid > 0) {
        const out = try a.alloc(u32, (at -| 1) / r.grid);
        for (out, 1..) |*x, i| x.* = @intCast(i * r.grid);
        return out;
    }
    var n: usize = 0;
    while (n < starts.len and starts[n] < at) n += 1;
    return a.dupe(u32, starts[0..n]);
}

/// Whether `kept` are the grid's starts below `at`.
fn onGrid(kept: []const u32, at: u32, grid: u32) bool {
    if (kept.len != (at -| 1) / grid) return false;
    for (kept, 1..) |k, i| if (k != i * grid) return false;
    return true;
}

/// A planned state's starts with its own position last: its blocks' last chunk ends by it.
pub fn withAt(a: Allocator, starts: []const u32, at: u32, planned: bool) ![]u32 {
    if (!planned) return a.alloc(u32, 0);
    var out = try a.alloc(u32, starts.len + 1);
    @memcpy(out[0..starts.len], starts);
    out[starts.len] = at;
    return out;
}

fn dayNow() u64 {
    var t: platform.c.timespec = undefined;
    if (platform.c.clock_gettime(platform.c.CLOCK.REALTIME, &t) != 0 or t.sec < 0) return 0;
    return @as(u64, @intCast(t.sec)) / 86400;
}

/// The file of name `n` (a Name or a Stem) with extension `ext` removed (a missing one is fine).
pub fn unlinkName(dir: []const u8, n: anytype, ext: []const u8) void {
    var buf: [1200]u8 = undefined;
    const p = file.path(&buf, dir, &chain.hex(n), ext, false) catch return;
    _ = platform.c.unlink(p);
}

fn readWord(p: [:0]const u8) u64 {
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = platform.c.close(fd);
    var w: u64 = 0;
    return if (platform.c.pread(fd, std.mem.asBytes(&w), 8, 0) == 8) w else 0;
}

fn writeWord(p: [:0]const u8, w: u64) void {
    var buf: [1200]u8 = undefined;
    const part = std.fmt.bufPrintSentinel(&buf, "{s}.part", .{p}, 0) catch return;
    const fd = platform.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return;
    defer _ = platform.c.close(fd);
    if (platform.c.write(fd, std.mem.asBytes(&w), 8) != 8) {
        _ = platform.c.unlink(part);
        return;
    }
    _ = platform.c.rename(part, p);
}

/// `dir` and its parents, 0700.
fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = platform.c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = platform.c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and platform.c.errno(rc) != .EXIST) return error.RowsDir;
}

test {
    _ = image;
    _ = chain;
    _ = file;
    _ = write.io;
    _ = @import("rows_test.zig");
    _ = @import("rows_spill_test.zig");
    _ = @import("rows_async_test.zig");
    _ = @import("rows_close_test.zig");
    _ = @import("rows_stage_test.zig");
    _ = @import("rows_select_test.zig");
    _ = @import("rows_recall_select_test.zig");
    _ = @import("rows_pool_select_test.zig");
    _ = @import("rows_prefetch.zig");
    _ = @import("rows_prefetch_stack_test.zig");
    _ = @import("rows_pump_test.zig");
    _ = @import("rows_local_test.zig");
    _ = @import("rows_carry_test.zig");
    _ = @import("rows_replay.zig");
}

/// TENSORFOLD_ROWS_STAGING_GIB: the host bytes the write worker stages at once (a save larger than that is skipped, so a
/// long conversation's state needs it raised); unset: `default`, the caller's option.
fn stagingBytes(default: u64, text: ?[]const u8) u64 {
    const t = text orelse return default;
    const g = std.fmt.parseFloat(f64, t) catch 0;
    if (!(g > 0 and g <= 1024)) {
        std.log.warn("TENSORFOLD_ROWS_STAGING_GIB={s}: give GiB above 0 and at most 1024; staging stays {d} MiB", .{ t, default >> 20 });
        return default;
    }
    return @intFromFloat(g * (1 << 30));
}

test "TENSORFOLD_ROWS_STAGING_GIB raises the staging permit; unset or unreadable keeps the caller's" {
    try std.testing.expectEqual(@as(u64, 1 << 30), stagingBytes(1 << 30, null));
    try std.testing.expectEqual(@as(u64, 8 << 30), stagingBytes(1 << 30, "8"));
    try std.testing.expectEqual(@as(u64, 3 << 29), stagingBytes(1 << 30, "1.5"));
    for ([_][]const u8{ "0", "-2", "x", "2000" }) |bad| try std.testing.expectEqual(@as(u64, 1 << 30), stagingBytes(1 << 30, bad));
}
