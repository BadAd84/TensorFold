//! TF_GLM_OVN_DISK_CARRY's record: what a store directory's states were computed under, on each rank's own disk.
const std = @import("std");
const platform = @import("sys");

/// A store directory's record (file `carry`): its id, its carry key, this rank's own proof and how rank 0 keys names.
pub const Record = extern struct {
    magic: u64 = magic,
    dir: u64, // the directory's own id: every file name in it is keyed by it
    key: u64, // the carry key: the identity without the kernel images (cuda_disk.carryOf)
    proof: u64, // this rank's own image of each probe state, hashed and folded (fold)
    block: u32 = 0, // rank 0's names: the store's block, lookahead and planning (another rank's record holds zeros)
    lookahead: u32 = 0,
    planned: u32 = 0,
    reserved: u32 = 0,
};

const magic: u64 = 0x3179727261634654; // "TFcarry1"
const file_name = "carry";

/// Store directory `dir`'s record on this disk, or null: none, a torn file or another format.
pub fn read(dir: []const u8) ?Record {
    var buf: [1200]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&buf, "{s}/" ++ file_name, .{dir}, 0) catch return null;
    var r: Record = undefined;
    if (!readExact(p, std.mem.asBytes(&r)) or r.magic != magic) return null;
    return r;
}

/// Store directory `dir`'s record replaced whole by `r` (a crash leaves the old record or the new one).
pub fn write(dir: []const u8, r: Record) !void {
    var buf: [1200]u8 = undefined;
    try replace(try std.fmt.bufPrintSentinel(&buf, "{s}/" ++ file_name, .{dir}, 0), std.mem.asBytes(&r));
}

/// Why record `got` does not prove `want` (null: it does): the same directory, carry key, proof and keying.
pub fn refusal(got: ?Record, want: Record) ?[]const u8 {
    const r = got orelse return "it records no carry key";
    if (r.dir != want.dir) return "its record names another directory";
    if (r.key != want.key) return "its carry key differs";
    if (r.proof != want.proof) return "this rank's probe states differ";
    if (r.block != want.block or r.lookahead != want.lookahead or r.planned != want.planned) return "its names are keyed otherwise";
    return null;
}

/// A rank's proof with one more probe state's image hashed in (`h`), folded the same way on every rank.
pub fn fold(proof: u64, h: u64) u64 {
    return std.hash.Wyhash.hash(proof, std.mem.asBytes(&h));
}

/// The hash a rank sends when it could not hash its image: rank 0 then refuses the carry for this start.
pub const unhashed: u64 = std.math.maxInt(u64);

/// Every rank's answer, gathered: yes only when there are answers and each of them is 1.
pub fn all(answers: []const u64) bool {
    for (answers) |v| if (v != 1) return false;
    return answers.len > 0;
}

/// File `p` replaced by `bytes`: written as `p`.part (a crash's leftover removed first), synced, renamed over `p`.
pub fn replace(p: [:0]const u8, bytes: []const u8) !void {
    var buf: [1300]u8 = undefined;
    const part = try std.fmt.bufPrintSentinel(&buf, "{s}.part", .{p}, 0);
    _ = platform.c.unlink(part); // a crash between a write and its rename leaves it, and the create below is exclusive
    const fd = platform.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.CarryWrite;
    const written = put(fd, bytes);
    _ = platform.c.close(fd);
    if (!written or platform.c.rename(part, p) != 0) {
        _ = platform.c.unlink(part);
        return error.CarryWrite;
    }
    syncDir(std.fs.path.dirname(p) orelse ".");
}

/// `bytes` written whole to `fd`, then synced.
fn put(fd: c_int, bytes: []const u8) bool {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return platform.c.fsync(fd) == 0;
}

/// Directory `dir`'s entries made durable after a rename into it (best effort: a failure leaves the rename as it was).
fn syncDir(dir: []const u8) void {
    var buf: [1300]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0) catch return;
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return;
    _ = platform.c.fsync(fd);
    _ = platform.c.close(fd);
}

/// Exactly `out.len` bytes, the whole of file `p` (a file of another size reads as false).
pub fn readExact(p: [:0]const u8, out: []u8) bool {
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return false;
    defer _ = platform.c.close(fd);
    if (platform.c.lseek(fd, 0, platform.c.SEEK.END) != @as(platform.c.off_t, @intCast(out.len))) return false;
    var done: usize = 0;
    while (done < out.len) {
        const n = platform.c.pread(fd, out.ptr + done, out.len - done, @intCast(done));
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

fn testDir(buf: []u8, tag: []const u8) ![:0]const u8 {
    const dir = try std.fmt.bufPrintSentinel(buf, "/tmp/tf-rows-carry-{s}-{d}", .{ tag, platform.c.getpid() }, 0);
    _ = platform.c.mkdir(dir, 0o700);
    return dir;
}

fn removeIn(dir: []const u8, name: []const u8) void {
    var buf: [256]u8 = undefined;
    _ = platform.c.unlink(std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch return);
}

test "rows_carry record: written whole and read back; a torn, foreign or missing file reads as none" {
    var dir_buf: [96]u8 = undefined;
    const dir = try testDir(&dir_buf, "record");
    defer _ = platform.c.rmdir(dir);
    defer removeIn(dir, file_name);
    try std.testing.expect(read(dir) == null);
    const r: Record = .{ .dir = 3, .key = 5, .proof = 7, .block = 256, .lookahead = 1, .planned = 1 };
    try write(dir, r);
    try std.testing.expectEqual(r, read(dir).?);
    // a crash's leftover beside it changes nothing and the next write replaces both
    var part: [128]u8 = undefined;
    const left = try std.fmt.bufPrintSentinel(&part, "{s}/" ++ file_name ++ ".part", .{dir}, 0);
    const fd = platform.c.open(left, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    _ = platform.c.close(fd);
    try std.testing.expectEqual(r, read(dir).?);
    var r2 = r;
    r2.key = 6;
    try write(dir, r2);
    try std.testing.expectEqual(r2, read(dir).?);
    try std.testing.expect(!readExact(left, std.mem.asBytes(&r2))); // the leftover went with the write
    // a file of another size (torn) or another format is no record
    var path_buf: [128]u8 = undefined;
    const p = try std.fmt.bufPrintSentinel(&path_buf, "{s}/" ++ file_name, .{dir}, 0);
    try replace(p, std.mem.asBytes(&r)[0 .. @sizeOf(Record) - 1]);
    try std.testing.expect(read(dir) == null);
    var foreign = r;
    foreign.magic +%= 1;
    try replace(p, std.mem.asBytes(&foreign));
    try std.testing.expect(read(dir) == null);
}

test "rows_carry record: a record proves only the same directory, key, proof and keying" {
    const want: Record = .{ .dir = 3, .key = 5, .proof = 7, .block = 256, .lookahead = 1, .planned = 0 };
    try std.testing.expect(refusal(want, want) == null);
    try std.testing.expect(refusal(null, want) != null);
    const fields = [_][]const u8{ "dir", "key", "proof", "block", "lookahead", "planned" };
    inline for (fields) |f| {
        var got = want;
        @field(got, f) += 1;
        try std.testing.expect(refusal(got, want) != null);
    }
    var other_rank = want; // another rank's record holds zeros for the keying, and so does the question it is asked
    other_rank.block = 0;
    other_rank.lookahead = 0;
    try std.testing.expect(refusal(other_rank, other_rank) == null);
}

test "rows_carry ranks: every rank's answer counts, the last one too; no answer is no" {
    try std.testing.expect(all(&.{ 1, 1, 1 }));
    try std.testing.expect(all(&.{1}));
    try std.testing.expect(!all(&.{ 1, 0, 1 }));
    try std.testing.expect(!all(&.{ 1, 1, 0 }));
    try std.testing.expect(!all(&.{ 1, 2, 1 }));
    try std.testing.expect(!all(&.{ 1, unhashed }));
    try std.testing.expect(!all(&.{}));
    // the fold depends on every hash and their order, from the same start on every rank
    try std.testing.expect(fold(fold(0, 1), 2) != fold(fold(0, 2), 1));
    try std.testing.expect(fold(0, 1) != fold(0, 2));
}
