//! TF_GLM_OVN_DISK_CARRY: a store with an empty own directory reads an older one that proves its key; none is changed.
const std = @import("std");
const platform = @import("sys");
const dirs = @import("lanes").learned_dirs;
pub const record = @import("lanes").carry_record;
const scan = @import("rows_scan.zig");
const Record = record.Record;

/// A store directory's id: its identity hashed (`Rows.init` names the directory and keys every file name by it).
pub fn dirOf(identity: u64) u64 {
    var h = std.hash.Wyhash.init(0x726f7773);
    h.update(std.mem.asBytes(&identity));
    return h.final();
}

/// An identity's adoption, `<root>/<own id>.carry`: the directory it reads, its key (never a name the sweeps take).
pub const Alias = extern struct { magic: u64 = alias_magic, own: u64, dir: u64, key: u64 };

const alias_magic: u64 = 0x317361696c614654; // "TFalias1"

/// A refusal identity `own` remembers (`<root>/<own id>.refused`): candidate, key, states and the replay's rows.
pub const Refusal = extern struct { magic: u64 = refusal_magic, own: u64, dir: u64, key: u64, states: u64, cap: u64, why: u64 = 0 };

const refusal_magic: u64 = 0x3173756665724654; // "TFrefus1"

/// What refused a remembered candidate (Refusal.why).
const Why = enum(u64) { ranks = 1, replay = 2 };

/// Every other rank's own disk (TF_GLM_OVN_DISK_LOCAL), asked about a directory before rank 0's store adopts it.
pub const Ranks = struct {
    ctx: *anyopaque,
    /// Whether every other rank holds directory `dir` (id `id`) recording carry key `key` and its own proof now.
    check: *const fn (ctx: *anyopaque, dir: []const u8, id: u64, key: u64) anyerror!bool,
};

/// A replay's answer: its states proved the directory, refused it (remembered), or could not run (tried again).
pub const Proof = enum { same, refused, failed };

/// TF_GLM_OVN_DISK_CARRY's replay, the last proof before an alias is written: directory `id`'s states hold here.
pub const Replay = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, id: u64) Proof,
    cap: u64, // the rows it prefills at most: a refusal remembered under another cap is void
};

/// The directory a store opens: `id` (its own, or the one it adopted), and why, for the log.
pub const Choice = struct { id: u64, adopted: bool = false, why: []const u8 };

/// The directory a store opens: its own, else its alias's or the newest proving `want`, if ranks and `replay` agree.
pub fn choose(root: []const u8, want: Record, ranks: ?Ranks, replay: ?Replay) Choice {
    const own = want.dir;
    var own_buf: [1200]u8 = undefined;
    const own_dir = path(&own_buf, root, own, "") catch return .{ .id = own, .why = "the root's path is too long" };
    if (holdsStates(own_dir)) {
        drop(root, own); // the identity writes its own directory: an alias of an earlier start is stale
        forget(root, own);
        return .{ .id = own, .why = "its own directory holds states" };
    }
    const pinned = readAlias(root, own);
    const from = blk: {
        if (pinned) |al| if (proves(root, al.dir, want) == null) break :blk al.dir;
        break :blk newest(root, want) orelse {
            drop(root, own); // an alias whose directory proves nothing now, or a crash's leftover of one
            forget(root, own);
            return .{ .id = own, .why = "no other directory records its carry key and proof" };
        };
    };
    var from_buf: [1200]u8 = undefined;
    const from_dir = path(&from_buf, root, from, "") catch return .{ .id = own, .why = "the root's path is too long" };
    const seen: Refusal = .{ .own = own, .dir = from, .key = want.key, .states = statesOf(from_dir), .cap = if (replay) |rp| rp.cap else 0 };
    if (remembered(root, seen)) |why| { // refused before, and the candidate holds the same states: not proved again
        drop(root, own);
        return .{ .id = own, .why = why };
    }
    if (ranks) |q| {
        const agreed = q.check(q.ctx, from_dir, from, want.key) catch null; // unanswered: refused, not remembered
        if (!(agreed orelse false)) {
            drop(root, own);
            if (agreed != null) remember(root, seen, .ranks);
            return .{ .id = own, .why = "a rank's own directory does not prove it" };
        }
    }
    if (pinned == null or pinned.?.dir != from or pinned.?.key != want.key) {
        if (replay) |rp| { // only an adoption about to be written replays
            const p = rp.run(rp.ctx, from);
            if (p != .same) {
                drop(root, own);
                if (p == .refused) remember(root, seen, .replay); // one that could not run is tried again
                return .{ .id = own, .why = "its replayed states do not prove it" };
            }
        }
        writeAlias(root, .{ .own = own, .dir = from, .key = want.key }) catch
            return .{ .id = own, .why = "its alias could not be written" };
    }
    forget(root, own); // a refusal remembered of states since changed
    return .{ .id = from, .adopted = true, .why = "adopted" };
}

/// Directory `dir`'s states as a refusal remembers them: every state file's name, and the newest one's time.
fn statesOf(dir: [:0]const u8) u64 {
    const d = platform.c.opendir(dir) orelse return 0;
    defer _ = platform.c.closedir(d);
    var set: u64 = 0;
    var latest: i128 = 0;
    while (platform.c.readdir(d)) |ent| {
        const name = scan.stateName(ent) orelse continue;
        set ^= std.hash.Wyhash.hash(0x7374, name);
        var buf: [1200]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch continue;
        latest = @max(latest, scan.mtimeOf(p));
    }
    var h = std.hash.Wyhash.init(0x72656675);
    h.update(std.mem.asBytes(&set));
    h.update(std.mem.asBytes(&latest));
    return h.final();
}

/// The refusal `r.own` remembers for exactly `r`'s candidate, key, states and cap, as its log line (null: none).
fn remembered(root: []const u8, r: Refusal) ?[]const u8 {
    var buf: [1200]u8 = undefined;
    const p = path(&buf, root, r.own, ".refused") catch return null;
    var got: Refusal = undefined;
    if (!record.readExact(p, std.mem.asBytes(&got)) or got.magic != refusal_magic) return null;
    if (got.own != r.own or got.dir != r.dir or got.key != r.key or got.states != r.states or got.cap != r.cap) return null;
    return switch (got.why) {
        @backingInt(Why.ranks) => "a rank's own directory did not prove it, and it holds the same states",
        @backingInt(Why.replay) => "its replayed states did not prove it, and it holds the same states",
        else => null,
    };
}

/// Refusal `r` remembered for identity `r.own`, `why` refusing it; one not written only proves it again next start.
fn remember(root: []const u8, r: Refusal, why: Why) void {
    var buf: [1200]u8 = undefined;
    var w = r;
    w.why = @backingInt(why);
    record.replace(path(&buf, root, r.own, ".refused") catch return, std.mem.asBytes(&w)) catch {};
}

/// Why directory `id` cannot be adopted for `want` (null: it can): its record proves `want`, and it holds a state.
fn proves(root: []const u8, id: u64, want: Record) ?[]const u8 {
    var buf: [1200]u8 = undefined;
    const d = path(&buf, root, id, "") catch return "its path is too long";
    var asked = want;
    asked.dir = id; // its record names itself: its names are keyed by its own id
    if (record.refusal(record.read(d), asked)) |why| return why;
    if (!holdsStates(d)) return "it holds no state";
    return null;
}

/// The most recently opened directory under `root` that `want` may adopt (`proves`), or null.
fn newest(root: []const u8, want: Record) ?u64 {
    var buf: [1200]u8 = undefined;
    const r = std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return null;
    const d = platform.c.opendir(r) orelse return null;
    defer _ = platform.c.closedir(d);
    var best: ?struct { id: u64, used: u64 } = null;
    while (platform.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (ent.type != platform.c.DT.DIR or name.len != 16) continue;
        const id = std.fmt.parseInt(u64, name, 16) catch continue;
        if (proves(root, id, want) != null) continue;
        const used = dirs.used(root, id) catch 0;
        if (best) |b| if (used < b.used or (used == b.used and id < b.id)) continue;
        best = .{ .id = id, .used = used };
    }
    return if (best) |b| b.id else null;
}

/// Whether directory `dir` holds a state file of rank 0's store (`<stem>.st`, as its scan reads them).
fn holdsStates(dir: [:0]const u8) bool {
    const d = platform.c.opendir(dir) orelse return false;
    defer _ = platform.c.closedir(d);
    while (platform.c.readdir(d)) |ent| if (scan.stateName(ent) != null) return true;
    return false;
}

/// `<root>/<id>` and `ext`: a directory (`ext` empty), an identity's alias (".carry") or its refusal (".refused").
fn path(buf: []u8, root: []const u8, id: u64, ext: []const u8) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}{s}", .{ root, id, ext }, 0);
}

/// Identity `own`'s alias under `root`, or null (none, torn or another format); what it names is proved before use.
pub fn readAlias(root: []const u8, own: u64) ?Alias {
    var buf: [1200]u8 = undefined;
    const p = path(&buf, root, own, ".carry") catch return null;
    var al: Alias = undefined;
    if (!record.readExact(p, std.mem.asBytes(&al)) or al.magic != alias_magic) return null;
    return al;
}

fn writeAlias(root: []const u8, al: Alias) !void {
    var buf: [1200]u8 = undefined;
    try record.replace(try path(&buf, root, al.own, ".carry"), std.mem.asBytes(&al));
}

/// Identity `own`'s alias removed (and a crash's leftover of one): it reads its own directory again.
fn drop(root: []const u8, own: u64) void {
    remove(root, own, ".carry");
}

/// Identity `own`'s remembered refusal removed (and a crash's leftover of one): its candidate is proved again.
fn forget(root: []const u8, own: u64) void {
    remove(root, own, ".refused");
}

fn remove(root: []const u8, own: u64, ext: []const u8) void {
    var buf: [1200]u8 = undefined;
    const p = path(&buf, root, own, ext) catch return;
    _ = platform.c.unlink(p);
    var part: [1300]u8 = undefined;
    _ = platform.c.unlink(std.fmt.bufPrintSentinel(&part, "{s}.part", .{p}, 0) catch return);
}
