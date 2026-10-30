//! TF_GLM_OVN_DISK_CARRY on the host: states written under one identity, read under another whose carry key matches.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const carry = @import("rows_carry.zig");
const fake = @import("rows_fake.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Record = carry.record.Record;
const Allocator = std.mem.Allocator;

const gpa = std.testing.allocator;
const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const old_identity: u64 = 7; // the build that wrote the states
const new_identity: u64 = 8; // a kernel change: another identity, the same carry key
const key: u64 = 0xca55;
const proof: u64 = 0x9f00d;

fn root(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-rows-carry-{s}-{d}", .{ tag, platform.c.getpid() }) catch unreachable;
}

fn options(dir: []const u8, identity: u64, adopted: ?u64, lookahead: u32) rows.Options {
    return .{ .sync_writes = true, .root = dir, .identity = identity, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = lookahead, .adopted = adopted };
}

/// What a start under `identity` brings: its own directory, the carry key `k`, this rank's proof, the store's keying.
fn want(identity: u64, k: u64) Record {
    return .{ .dir = carry.dirOf(identity), .key = k, .proof = proof, .block = rows.block_tokens, .lookahead = 1 };
}

/// The conversation: 1500 tokens, kept at 1400.
fn prompt(a: Allocator, salt: u32) ![]u32 {
    const out = try a.alloc(u32, 1500);
    for (out, 0..) |*t, i| t.* = @intCast(50_000 + i * 7 % 997 + salt);
    return out;
}

/// A start of build `identity`: the conversation kept at 1400 tokens on disk, its directory's record written as `rec`.
fn seed(a: Allocator, dir: []const u8, identity: u64, salt: u32, rec: Record) !void {
    var f: fake.Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir, identity, null, 1));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const p = try prompt(a, salt);
    f.prompt = p;
    const plan = try s.begin(a, p, 1400, &.{}, f.starts, null, f.spans);
    _ = f.pass(&s, plan);
    while (rows.write.step(r)) {}
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try carry.record.write(r.dir, rec);
}

/// The states as `identity` opens `adopted` (or its own): how many, each read back as the image a fresh pass leaves.
fn view(dir: []const u8, identity: u64, adopted: ?u64) !usize {
    var f: fake.Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir, identity, adopted, 1));
    defer r.deinit();
    for (0..r.states.items.len) |i| {
        const s = r.states.items[i];
        f.prompt = s.tokens;
        const img = try gpa.alloc(u8, @intCast(f.snapshots().vtable.bytes(&f, s.at)));
        defer gpa.free(img);
        f.build(s.at, img);
        const saved = try rows.read.read(r, i);
        defer f.snapshots().vtable.drop(&f, saved);
        const k: *fake.Kept = @ptrCast(@alignCast(saved));
        try std.testing.expectEqualSlices(u8, img, k.img);
    }
    return r.states.items.len;
}

/// The conversation's next turn on `identity` in `adopted` (or its own): where it resumed, and whether it was fresh.
fn nextTurn(a: Allocator, dir: []const u8, identity: u64, adopted: ?u64) !struct { from: u32, fresh: bool } {
    var f: fake.Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir, identity, adopted, 1));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const p = try prompt(a, 0);
    const turn2 = try std.mem.concat(a, u32, &.{ p[0..1401], &[_]u32{ 9, 8, 7, 6 } });
    f.prompt = turn2;
    const plan = try s.begin(a, turn2, 1404, &.{}, f.starts, null, f.spans);
    const sum = f.pass(&s, plan);
    while (rows.write.step(r)) {}
    return .{ .from = plan.from, .fresh = f.restored_ok and sum == fake.fresh(turn2) };
}

fn exists(dir: []const u8, id: u64, ext: []const u8) bool {
    var buf: [200]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}{s}", .{ dir, id, ext }, 0) catch return false;
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return false;
    _ = platform.c.close(fd);
    return true;
}

test "rows_carry adopts: a new identity reads the states of an older directory with its key, byte for byte" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "adopt");
    defer rmTree(dir);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    const old = carry.dirOf(old_identity);
    const c = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(c.adopted);
    try std.testing.expectEqual(old, c.id);
    try std.testing.expectEqual(old, carry.readAlias(dir, carry.dirOf(new_identity)).?.dir);
    try std.testing.expectEqual(@as(usize, 1), try view(dir, new_identity, c.id));
    const got = try nextTurn(a, dir, new_identity, c.id);
    try std.testing.expectEqual(@as(u32, 1400), got.from);
    try std.testing.expect(got.fresh);
    // one directory for both builds: the new identity's own was never made, and the old build reads the new turn too
    try std.testing.expect(!exists(dir, carry.dirOf(new_identity), ""));
    try std.testing.expectEqual(@as(usize, 2), try view(dir, old_identity, null));
    // a restart of the new build takes the same directory by its alias
    const again = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(again.adopted and again.id == old);
}

test "rows_carry takes the newest directory that proves its key and holds states, then keeps it while it proves" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "newest");
    defer rmTree(dir);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    try seed(a, dir, 9, 5, want(9, key)); // opened after it: the newer
    {
        var f: fake.Fake = .{ .gpa = gpa }; // newer still, its record the same key, and no state
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir, 10, null, 1));
        defer r.deinit();
        try carry.record.write(r.dir, want(10, key));
    }
    const c = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(c.adopted);
    try std.testing.expectEqual(carry.dirOf(9), c.id);
    // a newer directory with the key comes later: the alias keeps the one this identity already reads and writes
    try seed(a, dir, 11, 6, want(11, key));
    try std.testing.expectEqual(carry.dirOf(9), carry.choose(dir, want(new_identity, key), null, null).id);
    // the alias's directory proves the key no more (its record rewritten): the newest that does is taken instead
    var path_buf: [160]u8 = undefined;
    try carry.record.write(try std.fmt.bufPrint(&path_buf, "{s}/{x:0>16}", .{ dir, carry.dirOf(9) }), want(9, key + 1));
    const moved = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(moved.adopted and moved.id == carry.dirOf(11));
    try std.testing.expectEqual(carry.dirOf(11), carry.readAlias(dir, carry.dirOf(new_identity)).?.dir);
}

test "rows_carry refuses another key, proof, keying or a record of another directory; the older one stays as it was" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "refuse");
    defer rmTree(dir);
    const old = carry.dirOf(old_identity);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    var asked: [4]Record = @splat(want(new_identity, key));
    asked[0].key += 1; // another checkpoint, device, driver, facts or probe bits
    asked[1].proof += 1; // this rank's own probe states differ
    asked[2].planned = 1; // the store names its files otherwise
    asked[3].block += 1;
    for (asked) |w| {
        const c = carry.choose(dir, w, null, null);
        try std.testing.expect(!c.adopted);
        try std.testing.expectEqual(carry.dirOf(new_identity), c.id);
        try std.testing.expect(carry.readAlias(dir, carry.dirOf(new_identity)) == null);
    }
    // a record copied in from another directory proves nothing about this one's names
    var copied = want(old_identity, key);
    copied.dir = carry.dirOf(9);
    var path_buf: [160]u8 = undefined;
    const old_dir = try std.fmt.bufPrint(&path_buf, "{s}/{x:0>16}", .{ dir, old });
    try carry.record.write(old_dir, copied);
    try std.testing.expect(!carry.choose(dir, want(new_identity, key), null, null).adopted);
    try carry.record.write(old_dir, want(old_identity, key));
    // the new build then starts cold in its own directory, and the old build still resumes from its own
    const cold = try nextTurn(a, dir, new_identity, null);
    try std.testing.expectEqual(@as(u32, 0), cold.from);
    try std.testing.expectEqual(@as(usize, 1), try view(dir, old_identity, null));
}

test "rows_carry refuses a directory whose names are keyed otherwise, so its scan never drops the older states" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "keying");
    defer rmTree(dir);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    var w = want(new_identity, key);
    w.lookahead = 0; // the new build keeps states with no lookahead: every older name and length means another state
    const c = carry.choose(dir, w, null, null);
    try std.testing.expect(!c.adopted);
    {
        var f: fake.Fake = .{ .gpa = gpa }; // the new build's store opens where it was told, scanning with its keying
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir, new_identity, if (c.adopted) c.id else null, 0));
        r.deinit();
    }
    try std.testing.expectEqual(@as(usize, 1), try view(dir, old_identity, null));
}

/// Other ranks standing in for a TF_GLM_OVN_DISK_LOCAL group: their one answer, and what rank 0 asked them.
const Asked = struct {
    answer: anyerror!bool,
    calls: usize = 0,
    id: u64 = 0,
    key: u64 = 0,
    dir_len: usize = 0,

    fn check(ctx: *anyopaque, d: []const u8, id: u64, k: u64) anyerror!bool {
        const q: *Asked = @ptrCast(@alignCast(ctx));
        q.calls += 1;
        q.id = id;
        q.key = k;
        q.dir_len = d.len;
        return q.answer;
    }

    fn ranks(q: *Asked) carry.Ranks {
        return .{ .ctx = q, .check = check };
    }
};

test "rows_carry refuses when any rank's own directory does not prove it, or a rank cannot answer" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "ranks");
    defer rmTree(dir);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    const old = carry.dirOf(old_identity);
    const alias = carry.dirOf(new_identity);
    var no: Asked = .{ .answer = false };
    try std.testing.expect(!carry.choose(dir, want(new_identity, key), no.ranks(), null).adopted);
    try std.testing.expectEqual(@as(usize, 1), no.calls);
    try std.testing.expect(no.id == old and no.key == key and no.dir_len == dir.len + 17);
    try std.testing.expect(carry.readAlias(dir, alias) == null);
    var lost: Asked = .{ .answer = error.LinkFailed };
    try std.testing.expect(!carry.choose(dir, want(new_identity, key), lost.ranks(), null).adopted);
    try std.testing.expect(carry.readAlias(dir, alias) == null);
    var yes: Asked = .{ .answer = true };
    try std.testing.expect(carry.choose(dir, want(new_identity, key), yes.ranks(), null).adopted);
    // a later start whose ranks refuse drops the alias: the store reads its own directory, all ranks alike
    try std.testing.expect(!carry.choose(dir, want(new_identity, key), no.ranks(), null).adopted);
    try std.testing.expect(carry.readAlias(dir, alias) == null);
    try std.testing.expectEqual(@as(usize, 1), try view(dir, old_identity, null));
}

test "rows_carry survives a crash mid-adoption: the older directory stays usable and the next start completes it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "crash");
    defer rmTree(dir);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    const old = carry.dirOf(old_identity);
    const own = carry.dirOf(new_identity);
    // the crash: the alias whole under its staging name, never renamed (record.replace dies between the two)
    const staged: carry.Alias = .{ .own = own, .dir = old, .key = key };
    var part_buf: [200]u8 = undefined;
    const part = try std.fmt.bufPrintSentinel(&part_buf, "{s}/{x:0>16}.carry.part", .{ dir, own }, 0);
    const fd = platform.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, @sizeOf(carry.Alias)), platform.c.write(fd, std.mem.asBytes(&staged), @sizeOf(carry.Alias)));
    _ = platform.c.close(fd);
    try std.testing.expect(carry.readAlias(dir, own) == null); // a staged alias is no adoption
    // the old build restarted instead: its directory and states as they were
    try std.testing.expectEqual(@as(usize, 1), try view(dir, old_identity, null));
    // the new build's next start completes the adoption, the leftover gone
    const c = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(c.adopted and c.id == old);
    try std.testing.expectEqual(old, carry.readAlias(dir, own).?.dir);
    try std.testing.expect(!exists(dir, own, ".carry.part"));
    try std.testing.expectEqual(@as(usize, 1), try view(dir, new_identity, c.id));
}

test "rows_carry: an identity whose own directory holds states keeps them, and an alias whose directory went goes" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "stale");
    defer rmTree(dir);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    const own = carry.dirOf(new_identity);
    try std.testing.expect(carry.choose(dir, want(new_identity, key), null, null).adopted);
    // the old directory removed (another identity's room, say): the alias names nothing, so it goes
    var path_buf: [160]u8 = undefined;
    rmTree(try std.fmt.bufPrint(&path_buf, "{s}/{x:0>16}", .{ dir, carry.dirOf(old_identity) }));
    const gone = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(!gone.adopted and gone.id == own);
    try std.testing.expect(carry.readAlias(dir, own) == null);
    // the new build's own directory holds a state and an older one with the key is back: it keeps its own, alias gone
    _ = try nextTurn(a, dir, new_identity, null);
    try seed(a, dir, old_identity, 0, want(old_identity, key));
    var alias_buf: [200]u8 = undefined;
    const alias = try std.fmt.bufPrintSentinel(&alias_buf, "{s}/{x:0>16}.carry", .{ dir, own }, 0);
    var left: carry.Alias = .{ .own = own, .dir = carry.dirOf(old_identity), .key = key };
    left.magic +%= 1; // another format first: no alias at all
    try carry.record.replace(alias, std.mem.asBytes(&left));
    try std.testing.expect(carry.readAlias(dir, own) == null);
    left.magic -%= 1;
    try carry.record.replace(alias, std.mem.asBytes(&left));
    try std.testing.expect(carry.readAlias(dir, own) != null);
    const kept = carry.choose(dir, want(new_identity, key), null, null);
    try std.testing.expect(!kept.adopted and kept.id == own);
    try std.testing.expect(carry.readAlias(dir, own) == null);
}

test "rows_carry: a store opened without an adoption keeps today's directory and names" {
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "today");
    defer rmTree(dir);
    var f: fake.Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir, old_identity, null, 1));
    defer r.deinit();
    var h = std.hash.Wyhash.init(0x726f7773); // the directory's hash before the carry existed
    h.update(std.mem.asBytes(&old_identity));
    const today = h.final();
    try std.testing.expectEqual(today, r.keying.identity);
    try std.testing.expectEqual(today, carry.dirOf(old_identity));
    var name: [16]u8 = undefined;
    try std.testing.expectEqualStrings(try std.fmt.bufPrint(&name, "{x:0>16}", .{today}), std.fs.path.basename(r.dir));
}
