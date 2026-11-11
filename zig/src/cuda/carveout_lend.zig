//! The display-memory lending daemon's client: one claim over its SEQPACKET socket, mapped with CUDA's VMM.

const std = @import("std");
const linux = std.os.linux;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const arena = @import("arena.zig");

pub const Error = error{ NoDaemon, DaemonRefused, SliceTooSmall, VmmUnavailable, MapFailed };

/// The daemon's answer before any mapping: the connection that holds the claim, the lent descriptor, the granted bytes.
pub const Slab = struct { sock: linux.fd_t, fd: linux.fd_t, size: usize };

/// A claim mapped into the current context; `fd` is the connection whose close gives the memory back to the daemon.
pub const Claim = struct { fd: linux.fd_t, dev: abi.DevicePtr, size: usize, vmm: arena.Vmm };

/// A request's JSON: a key longer than this cannot be sent, and the daemon would refuse it as unknown anyway.
const request_max = 4096;
/// A reply's JSON: the daemon's own replies are a few hundred bytes.
const reply_max = 4096;
/// CMSG_SPACE for two descriptors is 40 bytes, and the daemon sends one.
const control_max = 64;

/// struct sockaddr_un: the family, then the path (108 bytes on Linux).
const SockaddrUn = extern struct { family: linux.sa_family_t = linux.AF.UNIX, path: [path_max]u8 };
/// The bytes a sockaddr_un's path holds, terminator included.
const path_max = 108;

/// Claims `want` bytes from the daemon at `path` and maps the grant for the current device; the socket stays open.
pub fn claim(d: *const Driver, path: []const u8, key: []const u8, want: usize) Error!Claim {
    const s = try slab(path, key, want);
    errdefer _ = linux.close(s.sock);
    errdefer _ = linux.close(s.fd);
    var device: abi.Device = 0;
    if (d.api.cuCtxGetDevice(&device) != abi.success) return error.VmmUnavailable;
    var vmm = arena.Vmm.open(d, device, s.size) catch return error.VmmUnavailable;
    errdefer vmm.close();
    const base = vmm.mapSlice(s.fd, s.size) catch return error.MapFailed;
    _ = linux.close(s.fd); // the mapping holds its own reference to the memory
    return .{ .fd = s.sock, .dev = base, .size = s.size, .vmm = vmm };
}

/// Unmaps the slice, then closes the connection: the close is what gives the memory back to the daemon.
pub fn release(c: Claim) void {
    var vmm = c.vmm; // the mapping is a value here, and its methods take a pointer to it
    vmm.unmapSlice(c.dev, c.size);
    vmm.close();
    _ = linux.close(c.fd);
}

/// A lending daemon lends whole slices of this size; an ask is rounded UP to it, so a grant is never short.
pub const gran: usize = 2 << 20;

/// The protocol half: one alloc request, its reply, and the descriptor that came with it.
pub fn slab(path: []const u8, key: []const u8, want: usize) Error!Slab {
    // Ask whole slices, rounded up: the daemon rounds up too, so an unrounded ask can exceed what is free and be refused.
    if (want == 0) return error.SliceTooSmall;
    const ask = std.mem.alignForward(usize, want, gran);
    const sock = try connect(path);
    errdefer _ = linux.close(sock);
    try send(sock, key, ask);
    const reply = try receive(sock);
    errdefer _ = linux.close(reply.fd);
    if (reply.size < want or reply.size == 0) return error.SliceTooSmall; // less than planned is a refusal
    // The reply's size is the authority; where the descriptor has a length of its own, that is checked too. The offset
    // is saved and restored around the seek: a seek that moved it would drop the caller's first read at EOF.
    const saved = linux.lseek(reply.fd, 0, linux.SEEK.CUR);
    if (linux.errno(saved) == .SUCCESS) {
        const end = linux.lseek(reply.fd, 0, linux.SEEK.END);
        _ = linux.lseek(reply.fd, @intCast(saved), linux.SEEK.SET);
        if (linux.errno(end) == .SUCCESS and @as(usize, @intCast(end)) < reply.size) return error.SliceTooSmall;
    }
    return .{ .sock = sock, .fd = reply.fd, .size = reply.size };
}

/// A connected SEQPACKET socket: the daemon refuses any other type, and a missing or refused path is no daemon.
fn connect(path: []const u8) Error!linux.fd_t {
    if (path.len == 0 or path.len >= path_max) return error.NoDaemon;
    var addr: SockaddrUn = .{ .path = @splat(0) };
    @memcpy(addr.path[0..path.len], path);
    const rc = linux.socket(linux.AF.UNIX, linux.SOCK.SEQPACKET | linux.SOCK.CLOEXEC, 0);
    if (linux.errno(rc) != .SUCCESS) return error.NoDaemon;
    const fd: linux.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    const len: linux.socklen_t = @intCast(@offsetOf(SockaddrUn, "path") + path.len + 1);
    if (linux.errno(linux.connect(fd, @ptrCast(&addr), len)) != .SUCCESS) return error.NoDaemon;
    return fd;
}

/// One request datagram: the key, the op and the ask, as JSON.
fn send(sock: linux.fd_t, key: []const u8, want: usize) Error!void {
    var buf: [request_max]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    const Request = struct { key: []const u8, op: []const u8, size: usize };
    std.json.Stringify.value(Request{ .key = key, .op = "alloc", .size = want }, .{}, &w) catch return error.DaemonRefused;
    const text = w.buffered();
    const rc = linux.sendto(sock, text.ptr, text.len, 0, null, 0);
    if (linux.errno(rc) != .SUCCESS or rc != text.len) return error.DaemonRefused;
}

const Reply = struct { size: usize, fd: linux.fd_t };

/// One reply datagram: its JSON and the descriptor in its ancillary data; a truncated or fd-less reply is refused.
fn receive(sock: linux.fd_t) Error!Reply {
    var buf: [reply_max]u8 = undefined;
    var control: [control_max]u8 align(linux.cmsg_align) = undefined;
    var iov = [_]std.posix.iovec{.{ .base = &buf, .len = buf.len }};
    var msg: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &control, .controllen = control.len, .flags = 0 };
    const rc = linux.recvmsg(sock, &msg, 0);
    if (linux.errno(rc) != .SUCCESS) return error.DaemonRefused;
    if (msg.flags & linux.MSG.TRUNC != 0) return error.DaemonRefused;
    const lent = takeFd(control[0..msg.controllen]) orelse return error.DaemonRefused;
    errdefer _ = linux.close(lent);
    const size = grantedSize(buf[0..rc]) orelse return error.DaemonRefused;
    return .{ .size = size, .fd = lent };
}

/// The first descriptor in the ancillary data; any others are closed, so a daemon that sends two leaks nothing.
fn takeFd(control: []const u8) ?linux.fd_t {
    var found: ?linux.fd_t = null;
    var off: usize = 0;
    while (off + @sizeOf(linux.cmsghdr) <= control.len) {
        const h: *const linux.cmsghdr = @ptrCast(@alignCast(control[off..].ptr));
        if (h.len < @sizeOf(linux.cmsghdr) or off + h.len > control.len) break;
        if (h.level == linux.SOL.SOCKET and h.type == linux.SCM.RIGHTS) {
            const data = control[off + @sizeOf(linux.cmsghdr) .. off + h.len];
            var i: usize = 0;
            while (i + @sizeOf(linux.fd_t) <= data.len) : (i += @sizeOf(linux.fd_t)) {
                const fd: linux.fd_t = @as(*const linux.fd_t, @ptrCast(@alignCast(data[i..].ptr))).*;
                if (found == null) found = fd else _ = linux.close(fd);
            }
        }
        off += std.mem.alignForward(usize, h.len, linux.cmsg_align);
    }
    return found;
}

/// The granted size of a reply: `ok` true and an integer `size`; null for a refusal or anything else.
fn grantedSize(text: []const u8) ?usize {
    var scratch: [reply_max]u8 = undefined; // the same bound as the datagram, so neither is tighter than the other
    var fba = std.heap.FixedBufferAllocator.init(&scratch);
    const doc = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), text, .{}) catch return null;
    const obj = switch (doc) {
        .object => |o| o,
        else => return null,
    };
    const ok = obj.get("ok") orelse return null;
    if (ok != .bool or !ok.bool) return null;
    const size = obj.get("size") orelse return null;
    const n = switch (size) {
        .integer => |i| i,
        else => return null,
    };
    if (n < 0) return null;
    return std.math.cast(usize, n);
}

// -- tests (a fake daemon on a real socket) ---------------------------------------------------------------------------

/// What the fake daemon replies with, and whether it waits for the client's close afterwards.
const Script = struct {
    payload: []const u8,
    lent: ?linux.fd_t = null, // the descriptor to attach, if any
    extra: ?linux.fd_t = null, // a second descriptor, to prove the client closes what it does not keep
    wait_close: bool = false,
    close_after_reply: bool = false, // the daemon closes once it has refused, as the real one does
};

/// What the fake daemon saw: the request bytes, and whether the client closed the connection.
const Observed = struct {
    request: [512]u8 = undefined,
    request_len: usize = 0,
    closed: bool = false,
    failed: ?anyerror = null,
};

/// A fake daemon: a listening SEQPACKET socket under the test's tmp dir, one connection, one scripted reply.
const Daemon = struct {
    buf: [128]u8 = undefined,
    path: [:0]const u8 = "",
    listener: linux.fd_t = -1,
    script: Script = .{ .payload = "" },
    observed: Observed = .{},
    thread: ?std.Thread = null,

    fn start(d: *Daemon, tmp: *std.testing.TmpDir, name: []const u8, script: Script) !void {
        d.script = script;
        d.observed = .{};
        d.path = try std.fmt.bufPrintSentinel(&d.buf, ".zig-cache/tmp/{s}/{s}.sock", .{ tmp.sub_path, name }, 0);
        var addr: SockaddrUn = .{ .path = @splat(0) };
        @memcpy(addr.path[0..d.path.len], d.path);
        const rc = linux.socket(linux.AF.UNIX, linux.SOCK.SEQPACKET | linux.SOCK.CLOEXEC, 0);
        if (linux.errno(rc) != .SUCCESS) return error.SocketFailed;
        d.listener = @intCast(rc);
        errdefer _ = linux.close(d.listener);
        const len: linux.socklen_t = @intCast(@offsetOf(SockaddrUn, "path") + d.path.len + 1);
        if (linux.errno(linux.bind(d.listener, @ptrCast(&addr), len)) != .SUCCESS) return error.BindFailed;
        if (linux.errno(linux.listen(d.listener, 4)) != .SUCCESS) return error.ListenFailed;
        d.thread = try std.Thread.spawn(.{}, serve, .{d});
    }

    fn stop(d: *Daemon) void {
        if (d.thread) |th| th.join();
        d.thread = null;
        if (d.listener >= 0) _ = linux.close(d.listener);
        d.listener = -1;
    }
};

/// The daemon's side: accept, read the request, reply, and (when asked) wait for the client's close.
fn serve(d: *Daemon) void {
    const conn = acceptWithin(d.listener, 5000) orelse {
        d.observed.failed = error.NoConnection;
        return;
    };
    defer _ = linux.close(conn);
    const n = linux.read(conn, &d.observed.request, d.observed.request.len);
    if (linux.errno(n) != .SUCCESS) {
        d.observed.failed = error.ReadFailed;
        return;
    }
    d.observed.request_len = @intCast(n);
    sendReply(conn, d.script);
    if (d.script.close_after_reply) return; // the real daemon closes after a refusal; the reply is already queued
    if (!d.script.wait_close) return;
    var fds = [_]linux.pollfd{.{ .fd = conn, .events = linux.POLL.IN, .revents = 0 }};
    const r = linux.poll(&fds, 1, 5000);
    if (linux.errno(r) != .SUCCESS or r == 0) {
        d.observed.failed = error.NoClose;
        return;
    }
    var one: [1]u8 = undefined;
    const c = linux.read(conn, &one, 1);
    d.observed.closed = linux.errno(c) == .SUCCESS and c == 0;
}

/// The next connection, or null when none arrives within `ms`: a failing test must not hang the suite.
fn acceptWithin(listener: linux.fd_t, ms: i32) ?linux.fd_t {
    var fds = [_]linux.pollfd{.{ .fd = listener, .events = linux.POLL.IN, .revents = 0 }};
    const r = linux.poll(&fds, 1, ms);
    if (linux.errno(r) != .SUCCESS or r == 0) return null;
    const rc = linux.accept(listener, null, null);
    if (linux.errno(rc) != .SUCCESS) return null;
    return @intCast(rc);
}

/// The scripted reply, with its descriptors in the ancillary data.
fn sendReply(conn: linux.fd_t, script: Script) void {
    var control: [control_max]u8 align(linux.cmsg_align) = @splat(0);
    var iov = [_]std.posix.iovec_const{.{ .base = script.payload.ptr, .len = script.payload.len }};
    var msg: linux.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = null, .controllen = 0, .flags = 0 };
    var fds: [2]linux.fd_t = undefined;
    var n: usize = 0;
    if (script.lent) |l| {
        fds[n] = l;
        n += 1;
    }
    if (script.extra) |e| {
        fds[n] = e;
        n += 1;
    }
    if (n > 0) {
        const h: *linux.cmsghdr = @ptrCast(&control);
        h.* = .{ .len = @sizeOf(linux.cmsghdr) + n * @sizeOf(linux.fd_t), .level = linux.SOL.SOCKET, .type = linux.SCM.RIGHTS };
        @memcpy(control[@sizeOf(linux.cmsghdr)..][0 .. n * @sizeOf(linux.fd_t)], std.mem.sliceAsBytes(fds[0..n]));
        msg.control = &control;
        msg.controllen = std.mem.alignForward(usize, h.len, linux.cmsg_align);
    }
    _ = linux.sendmsg(conn, &msg, 0);
}

/// A descriptor the daemon can lend: a memfd holding `marker`, so the client can prove it is the same file.
fn memfd(marker: []const u8) !linux.fd_t {
    const rc = linux.memfd_create("lend", 0);
    if (linux.errno(rc) != .SUCCESS) return error.MemfdFailed;
    const fd: linux.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    // As large as the whole reservation the daemon can grant, so a check against the descriptor's own length
    // passes for any grant a test reports. A memfd is sparse, so this costs no memory.
    if (linux.errno(linux.ftruncate(fd, 2145386496)) != .SUCCESS) return error.MemfdFailed;
    if (linux.errno(linux.pwrite(fd, marker.ptr, marker.len, 0)) != .SUCCESS) return error.MemfdFailed;
    return fd;
}

/// The descriptors this process holds, so a test can prove one was closed rather than leaked.
fn openFds() usize {
    const rc = linux.open("/proc/self/fd", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return 0;
    const fd: linux.fd_t = @intCast(rc);
    defer _ = linux.close(fd);
    var buf: [2048]u8 = undefined;
    var n: usize = 0;
    while (true) {
        const r = linux.getdents64(fd, &buf, buf.len);
        if (linux.errno(r) != .SUCCESS) break;
        const len: usize = @intCast(r);
        if (len == 0) break;
        var off: usize = 0;
        while (off < len) {
            const e: *const linux.dirent64 = @ptrCast(@alignCast(buf[off..].ptr));
            if (e.reclen == 0) break;
            const name: [*:0]const u8 = @ptrCast(&e.name);
            if (name[0] != '.') n += 1;
            off += e.reclen;
        }
    }
    return n;
}

test "a granted claim carries the size the daemon granted, and the descriptor by SCM_RIGHTS" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("lend");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "granted", .{ .payload = "{\"ok\": true, \"base\": 10739515392, \"size\": 4194304}", .lent = lent });
    defer d.stop();
    const s = try slab(d.path, "kindlingai_1", 1 << 20);
    defer _ = linux.close(s.sock);
    defer _ = linux.close(s.fd);
    try t.expectEqual(@as(usize, 4 << 20), s.size); // the grant, not the 1 MiB ask
    var marker: [4]u8 = undefined;
    const n = linux.read(s.fd, &marker, marker.len);
    try t.expectEqual(linux.E.SUCCESS, linux.errno(n));
    try t.expectEqualStrings("lend", &marker); // the daemon's own file, read through the received descriptor
    d.stop();
    var parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, d.observed.request[0..d.observed.request_len], .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try t.expectEqualStrings("kindlingai_1", obj.get("key").?.string);
    try t.expectEqualStrings("alloc", obj.get("op").?.string);
    try t.expectEqual(@as(i64, 2 << 20), obj.get("size").?.integer); // the 1 MiB ask, rounded up to a whole slice
}

test "a refused claim is DaemonRefused" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    const reply = "{\"ok\": false, \"error\": \"not enough free carveout\", \"base\": 1, \"size\": 2, \"free\": 3, \"largest\": 4}";
    try d.start(&tmp, "refused", .{ .payload = reply });
    defer d.stop();
    try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
}

test "a reply that is not the expected JSON is DaemonRefused" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("json");
    defer _ = linux.close(lent);
    // Each carries a descriptor, so the refusal is the JSON's and not a missing fd's.
    const payloads = [_][]const u8{
        "not json at all",
        "[1, 2, 3]",
        "{\"base\": 1, \"size\": 2097152}",
        "{\"ok\": true, \"size\": \"2097152\"}",
        "{\"ok\": true, \"size\": -1}",
        "{\"ok\": true, \"size\": 2097152.5}",
        "{\"ok\": \"yes\", \"size\": 2097152}",
        "{\"ok\": false, \"size\": 2097152}",
        "{\"ok\": false, \"error\": \"unknown op\"}",
    };
    for (payloads, 0..) |payload, i| {
        var name: [8]u8 = undefined;
        var d: Daemon = .{};
        try d.start(&tmp, try std.fmt.bufPrint(&name, "m{d}", .{i}), .{ .payload = payload, .lent = lent });
        defer d.stop();
        try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
        d.stop();
    }
}

test "a zero-byte grant is SliceTooSmall even when nothing was asked for" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("none");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "zero", .{ .payload = "{\"ok\": true, \"size\": 0}", .lent = lent });
    defer d.stop();
    try t.expectError(error.SliceTooSmall, slab(d.path, "kindlingai_1", 0));
}

test "a reply longer than the client's buffer is DaemonRefused, not a truncated parse" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("long");
    defer _ = linux.close(lent);
    // A complete grant followed by whitespace: a truncated read still parses, so only the truncation check refuses it.
    var payload: [reply_max + 64]u8 = undefined;
    const head = "{\"ok\": true, \"size\": 2097152}";
    @memcpy(payload[0..head.len], head);
    @memset(payload[head.len..], ' ');
    var d: Daemon = .{};
    try d.start(&tmp, "long", .{ .payload = &payload, .lent = lent });
    defer d.stop();
    try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
}

test "a refusal that carries a size is refused, not read as a grant" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    defer d.stop();
    const lent = try memfd("refused");
    defer _ = linux.close(lent);
    // "not enough free carveout" carries a size and a descriptor, so trusting the size alone would read it as a grant.
    try d.start(&tmp, "refused-size", .{
        .payload = "{\"ok\": false, \"error\": \"not enough free carveout\", \"base\": 10739515392, \"size\": 2145386496, \"free\": 0, \"largest\": 0}",
        .lent = lent,
    });
    try std.testing.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 4 << 20));
}

test "an ask between two slices is rounded up, so the grant is never less than planned" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    defer d.stop();
    const lent = try memfd("round-up");
    defer _ = linux.close(lent);
    try d.start(&tmp, "round-up", .{ .payload = "{\"ok\": true, \"base\": 10739515392, \"size\": 4194304}", .lent = lent });
    const ask = (3 << 20) + 1;
    const s = try slab(d.path, "kindlingai_1", ask);
    defer _ = linux.close(s.fd);
    defer _ = linux.close(s.sock);
    try std.testing.expect(s.size >= ask); // never quietly less than asked
    d.stop();
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, d.observed.request[0..d.observed.request_len], .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 4194304), parsed.value.object.get("size").?.integer); // rounded up on the wire
}

test "a refusal followed by the daemon closing the connection is still the cause, not a broken pipe" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    defer d.stop();
    // The real daemon answers a refusal and then closes, so the reply arrives with an EOF behind it: the client must
    // read the reply it was given and report that, not surface the close.
    try d.start(&tmp, "close-after-refusal", .{
        .payload = "{\"ok\": false, \"error\": \"not enough free carveout\", \"base\": 10739515392, \"size\": 2145386496, \"free\": 0, \"largest\": 0}",
        .close_after_reply = true,
    });
    try std.testing.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 4 << 20));
}

test "a grant smaller than the ask is SliceTooSmall, not a smaller pool" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("half");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "short", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent });
    defer d.stop();
    try t.expectError(error.SliceTooSmall, slab(d.path, "kindlingai_1", 4 << 20));
}

test "a reply with no descriptor is DaemonRefused" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    try d.start(&tmp, "nofd", .{ .payload = "{\"ok\": true, \"base\": 10739515392, \"size\": 2097152}" });
    defer d.stop();
    try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
}

test "a missing socket is NoDaemon" {
    const t = std.testing;
    try t.expectError(error.NoDaemon, slab(".zig-cache/tmp/absent/d.sock", "kindlingai_1", 1 << 20));
}

test "the daemon sees the socket close as the release" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("held");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "release", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent, .wait_close = true });
    defer d.stop();
    const s = try slab(d.path, "kindlingai_1", 1 << 20);
    _ = linux.close(s.fd); // the mapping would hold its own reference; the socket is the claim
    _ = linux.close(s.sock);
    d.stop();
    try t.expect(d.observed.failed == null);
    try t.expect(d.observed.closed);
}

/// A driver whose only reachable call fails: a claim that gets past the protocol half stops here, not in CUDA.
const NoCuda = struct {
    var calls: u32 = 0;
    fn ctxGetDevice(_: *abi.Device) callconv(.c) abi.Result {
        calls += 1;
        return 201; // CUDA_ERROR_INVALID_CONTEXT
    }
    fn driver() Driver {
        var d: Driver = undefined;
        d.api.cuCtxGetDevice = ctxGetDevice;
        return d;
    }
};

test "claim refuses a missing socket, a refused claim and a short grant before it touches CUDA" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const d = NoCuda.driver();
    NoCuda.calls = 0;
    try t.expectError(error.NoDaemon, claim(&d, ".zig-cache/tmp/absent/d.sock", "kindlingai_1", 1 << 20));
    const lent = try memfd("half");
    defer _ = linux.close(lent);
    var refused: Daemon = .{};
    try refused.start(&tmp, "claim-refused", .{ .payload = "{\"ok\": false, \"error\": \"unknown op\"}" });
    defer refused.stop();
    try t.expectError(error.DaemonRefused, claim(&d, refused.path, "kindlingai_1", 1 << 20));
    var short: Daemon = .{};
    try short.start(&tmp, "claim-short", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent });
    defer short.stop();
    try t.expectError(error.SliceTooSmall, claim(&d, short.path, "kindlingai_1", 4 << 20));
    try t.expectEqual(@as(u32, 0), NoCuda.calls); // every refusal above happened before any driver call
}

test "a claim the protocol half accepts reaches the driver, and a driver that cannot answer is VmmUnavailable" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("vmm");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "vmm", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent });
    defer d.stop();
    const drv = NoCuda.driver();
    NoCuda.calls = 0;
    try t.expectError(error.VmmUnavailable, claim(&drv, d.path, "kindlingai_1", 1 << 20));
    try t.expectEqual(@as(u32, 1), NoCuda.calls);
}

test "a second descriptor in the reply is closed, not leaked" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const first = try memfd("one!");
    defer _ = linux.close(first);
    const second = try memfd("two!");
    defer _ = linux.close(second);
    const baseline = openFds(); // before the daemon exists, so its listener is not in the count
    var d: Daemon = .{};
    // wait_close holds the daemon's connection open, so the count below is not racing its close.
    try d.start(&tmp, "two", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = first, .extra = second, .wait_close = true });
    defer d.stop();
    const s = try slab(d.path, "kindlingai_1", 1 << 20);
    // the listener, the client's socket, the daemon's connection and the one descriptor kept: the second was closed
    try t.expectEqual(baseline + 4, openFds());
    _ = linux.close(s.sock);
    _ = linux.close(s.fd);
    d.stop();
    try t.expectEqual(baseline, openFds());
}
