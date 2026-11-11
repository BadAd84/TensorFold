//! GB10 display memory as device memory: a DRM dumb buffer, mapped and registered with CUDA, carved into cache spans.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const DriverError = @import("driver.zig").Error;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;

pub const width = 4096; // pixels a framebuffer row
pub const bpp = 32;
pub const row_bytes = width * bpp / 8; // the unit a carveout's size rounds down to
pub const default_bytes: usize = 1792 << 20; // what GB10's display reservation holds with room to spare
pub const default_card = "/dev/dri/card0";
pub const alignment = 256; // cuMemAlloc's own

/// struct drm_mode_create_dumb (drm_mode.h): height, width and bpp in; handle, pitch and size out.
pub const CreateDumb = extern struct { height: u32, width: u32, bpp: u32, flags: u32 = 0, handle: u32 = 0, pitch: u32 = 0, size: u64 = 0 };
/// struct drm_mode_map_dumb: the handle in, the offset to mmap the card at out.
pub const MapDumb = extern struct { handle: u32, pad: u32 = 0, offset: u64 = 0 };
/// struct drm_mode_destroy_dumb.
pub const DestroyDumb = extern struct { handle: u32 };

pub const ioctl_create_dumb = linux.IOCTL.IOWR('d', 0xB2, CreateDumb);
pub const ioctl_map_dumb = linux.IOCTL.IOWR('d', 0xB3, MapDumb);
pub const ioctl_destroy_dumb = linux.IOCTL.IOWR('d', 0xB4, DestroyDumb);

pub const Error = DriverError || error{ CardUnavailable, DumbBufferRefused, MapRefused, RegisterRefused, NoDaemon, DaemonRefused, SliceTooSmall, VmmUnavailable, MapFailed };

/// Which way the display memory is obtained: a DRM dumb buffer, or a display-memory daemon lending its reservation.
pub const Backend = enum { drm, daemon };

/// What TF_CUDA_CARVEOUT asked for: the backend and the size in bytes (whole framebuffer rows for drm).
pub const Request = struct { backend: Backend, bytes: usize, socket: []const u8 = default_socket, key: []const u8 = default_key };

/// `bytes` in whole framebuffer rows; zero below one row.
pub fn roundBytes(bytes: usize) usize {
    return bytes / row_bytes * row_bytes;
}

/// The options: the flag turns it on, the source picks the card or a daemon, plus that daemon's socket and key.
pub fn requested(flag: bool, source: ?[]const u8, socket: ?[]const u8, key: ?[]const u8, key_file: ?[]const u8, mib: ?[]const u8) error{ InvalidCharacter, Overflow, CarveoutTooSmall, InvalidEnvironment }!?Request {
    // Off: not this source's business, the walk skips them too. On: a mistyped value is refused here and at startup.
    if (!flag) return null;
    _ = try @import("sys").environment.parse("TF_CUDA_CARVEOUT_SOURCE", source);
    _ = try @import("sys").environment.parse("TF_CUDA_CARVEOUT_SOCKET", socket);
    _ = try @import("sys").environment.parse("TF_CUDA_CARVEOUT_KEY", key);
    if (key == null) _ = try @import("sys").environment.parse("TF_CUDA_CARVEOUT_KEY_FILE", key_file);
    const backend: Backend = if (if (source) |v| std.mem.eql(u8, v, "daemon") else false) .daemon else .drm;
    const where = socket orelse default_socket;
    const whose = key orelse try keyFrom(if (key_file) |f| f else null);
    const m = mib orelse return .{ .backend = backend, .bytes = default_bytes, .socket = where, .key = whose };
    if (m.len == 0) return .{ .backend = backend, .bytes = default_bytes, .socket = where, .key = whose };
    _ = try @import("sys").environment.parse("TF_CUDA_CARVEOUT_MIB", m);
    const bytes = try std.math.mul(usize, try std.fmt.parseInt(usize, m, 10), 1 << 20);
    if (backend == .drm) {
        if (roundBytes(bytes) == 0) return error.CarveoutTooSmall;
        return .{ .backend = backend, .bytes = roundBytes(bytes), .socket = where, .key = whose };
    }
    if (bytes < gran) return error.CarveoutTooSmall;
    return .{ .backend = backend, .bytes = bytes, .socket = where, .key = whose };
}

/// The key: the setting, else the key file's first line, else the default; a per-box key names that file.
fn keyFrom(key_file: ?[]const u8) error{InvalidEnvironment}![]const u8 {
    const Held = struct {
        var text: ?[]const u8 = null;
    };
    const path = key_file orelse return default_key;
    if (path.len == 0) return default_key;
    if (Held.text) |held| return held;
    var pbuf: [256]u8 = undefined;
    if (path.len >= pbuf.len) return error.InvalidEnvironment;
    @memcpy(pbuf[0..path.len], path);
    pbuf[path.len] = 0;
    const fd_rc = linux.openat(linux.AT.FDCWD, @ptrCast(&pbuf), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(fd_rc) != .SUCCESS) return error.InvalidEnvironment;
    const fd: linux.fd_t = @intCast(fd_rc);
    defer _ = linux.close(fd);
    var buf: [256]u8 = undefined;
    const n = linux.read(fd, &buf, buf.len);
    if (linux.errno(n) != .SUCCESS) return error.InvalidEnvironment;
    const line = std.mem.sliceTo(buf[0..n], '\n');
    if (line.len == 0) return error.InvalidEnvironment;
    const dup = std.heap.page_allocator.dupe(u8, line) catch return error.InvalidEnvironment; // outlives this call
    Held.text = dup;
    return dup;
}

/// First-fit placement of 256-byte aligned spans in `size` bytes; with nothing given back it is a bump allocator.
pub const Spans = struct {
    pub const max = 64;
    pub const Range = struct { at: usize, len: usize };

    size: usize,
    live: [max]Range = undefined, // sorted by offset
    n: usize = 0,

    /// The offset of a new span of `len` bytes, or null when no gap (or no slot) holds it.
    pub fn take(s: *Spans, len: usize) ?usize {
        if (len == 0 or s.n == max) return null;
        var at: usize = 0;
        for (0..s.n + 1) |i| {
            const end = if (i < s.n) s.live[i].at else s.size;
            if (at <= end and end - at >= len) {
                std.mem.copyBackwards(Range, s.live[i + 1 .. s.n + 1], s.live[i..s.n]);
                s.live[i] = .{ .at = at, .len = len };
                s.n += 1;
                return at;
            }
            if (i < s.n) at = std.mem.alignForward(usize, s.live[i].at + s.live[i].len, alignment);
        }
        return null;
    }

    /// Gives back the span at `at`; an offset no span starts at is ignored.
    pub fn release(s: *Spans, at: usize) void {
        for (s.live[0..s.n], 0..) |r, i| if (r.at == at) {
            std.mem.copyForwards(Range, s.live[i .. s.n - 1], s.live[i + 1 .. s.n]);
            s.n -= 1;
            return;
        };
    }

    /// Bytes no live span holds (alignment gaps count as free).
    pub fn free(s: *const Spans) usize {
        var used: usize = 0;
        for (s.live[0..s.n]) |r| used += r.len;
        return s.size - used;
    }
};

/// The reference daemon's socket and key (kindling spark-os's dispramd); another daemon sets both.
pub const default_socket = "/run/dispram/dispram.sock";
pub const default_key = "kindlingai_1";
pub const gran = @import("carveout_lend.zig").gran; // one owner for the slice size

/// The registered framebuffer: owns the card's fd, the dumb buffer, its mapping and the CUDA registration.
pub const Carveout = struct {
    d: *const Driver,
    backend: Backend = .drm,
    fd: linux.fd_t,
    handle: u32,
    host: []align(std.heap.page_size_min) u8,
    dev: abi.DevicePtr,
    flags: c_uint, // the registration flags the driver took
    spans: Spans,
    vmm: ?@import("arena.zig").Vmm = null, // the daemon slice's mapping; the card path has none

    /// Maps `want` bytes (whole rows) of `card`'s display memory into the current context.
    pub fn open(d: *const Driver, card: [*:0]const u8, want: usize) Error!Carveout {
        const size = roundBytes(want);
        if (size == 0) return error.Invalid;
        const fd_rc = linux.open(card, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
        if (linux.errno(fd_rc) != .SUCCESS) return error.CardUnavailable;
        const fd: linux.fd_t = @intCast(fd_rc);
        errdefer _ = linux.close(fd);
        var create: CreateDumb = .{ .height = @intCast(size / row_bytes), .width = width, .bpp = bpp };
        if (linux.errno(linux.ioctl(fd, ioctl_create_dumb, @intFromPtr(&create))) != .SUCCESS) return error.DumbBufferRefused;
        errdefer destroy(fd, create.handle);
        if (create.pitch != row_bytes or create.size < size) return error.DumbBufferRefused;
        var map: MapDumb = .{ .handle = create.handle };
        if (linux.errno(linux.ioctl(fd, ioctl_map_dumb, @intFromPtr(&map))) != .SUCCESS) return error.MapRefused;
        const addr = linux.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, @intCast(map.offset));
        if (linux.errno(addr) != .SUCCESS) return error.MapRefused;
        const base: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(addr);
        errdefer _ = linux.munmap(base, size);
        const flags = try register(d, base, size);
        errdefer _ = d.api.cuMemHostUnregister(base);
        var dev: abi.DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, base, 0), "cuMemHostGetDevicePointer");
        return .{ .d = d, .fd = fd, .handle = create.handle, .host = base[0..size], .dev = dev, .flags = flags, .spans = .{ .size = size } };
    }

    /// DEVICEMAP first; a driver that sees the range as I/O memory wants IOMEMORY said too.
    fn register(d: *const Driver, base: [*]u8, len: usize) Error!c_uint {
        const plain = abi.host_register_devicemap;
        if (d.api.cuMemHostRegister_v2(base, len, plain) == abi.success) return plain;
        const io = plain | abi.host_register_iomemory;
        d.check(d.api.cuMemHostRegister_v2(base, len, io), "cuMemHostRegister") catch return error.RegisterRefused;
        return io;
    }

    fn destroy(fd: linux.fd_t, handle: u32) void {
        var x: DestroyDumb = .{ .handle = handle };
        _ = linux.ioctl(fd, ioctl_destroy_dumb, @intFromPtr(&x));
    }

    /// Releases in reverse order of open; every span taken from it must be out of use.
    pub fn close(c: *Carveout) void {
        switch (c.backend) {
            .drm => {
                _ = c.d.api.cuMemHostUnregister(c.host.ptr);
                _ = linux.munmap(c.host.ptr, c.host.len);
                destroy(c.fd, c.handle);
                _ = linux.close(c.fd);
            },
            .daemon => @import("carveout_lend.zig").release(.{ .fd = c.fd, .dev = c.dev, .size = c.spans.size, .vmm = c.vmm.? }),
        }
        c.* = undefined;
    }

    /// The daemon backend: claims what a lending daemon holds, over its socket with its key, keeping the socket open.
    pub fn openDaemon(d: *const Driver, socket: []const u8, key: []const u8, want: usize) Error!Carveout {
        const claimed = try @import("carveout_lend.zig").claim(d, socket, key, want);
        return .{ .d = d, .backend = .daemon, .fd = claimed.fd, .handle = 0, .host = undefined, .dev = claimed.dev, .flags = 0, .spans = .{ .size = claimed.size }, .vmm = claimed.vmm };
    }

    /// A zeroed span of `len` bytes, 256-byte aligned; null when it does not fit (the caller falls back).
    pub fn take(c: *Carveout, len: usize) Error!?DeviceBuffer {
        const at = c.spans.take(len) orelse return null;
        errdefer c.spans.release(at);
        const b: DeviceBuffer = .{ .d = c.d, .ptr = c.dev + at, .len = len, .borrowed = true };
        try b.fill8(0, null);
        return b;
    }

    /// Returns a span `take` gave out.
    pub fn give(c: *Carveout, b: DeviceBuffer) void {
        c.spans.release(b.ptr - c.dev);
    }

    pub fn bytes(c: *const Carveout) usize {
        return c.spans.size;
    }

    pub fn freeBytes(c: *const Carveout) usize {
        return c.spans.free();
    }
};

test "DRM ioctl numbers are drm_mode.h's" {
    try std.testing.expectEqual(@as(u32, 0xC02064B2), ioctl_create_dumb);
    try std.testing.expectEqual(@as(u32, 0xC01064B3), ioctl_map_dumb);
    try std.testing.expectEqual(@as(u32, 0xC00464B4), ioctl_destroy_dumb);
}

test "dumb buffer structs have drm_mode.h's layout" {
    try std.testing.expectEqual(32, @sizeOf(CreateDumb));
    try std.testing.expectEqual(16, @offsetOf(CreateDumb, "handle"));
    try std.testing.expectEqual(20, @offsetOf(CreateDumb, "pitch"));
    try std.testing.expectEqual(24, @offsetOf(CreateDumb, "size"));
    try std.testing.expectEqual(16, @sizeOf(MapDumb));
    try std.testing.expectEqual(8, @offsetOf(MapDumb, "offset"));
    try std.testing.expectEqual(4, @sizeOf(DestroyDumb));
}

test "spans are 256-byte aligned and refused once full" {
    var s: Spans = .{ .size = 4096 };
    try std.testing.expectEqual(@as(?usize, 0), s.take(100));
    try std.testing.expectEqual(@as(?usize, 256), s.take(1000));
    try std.testing.expectEqual(@as(?usize, 1280), s.take(2816));
    try std.testing.expectEqual(4096 - 3916, s.free());
    try std.testing.expectEqual(@as(?usize, null), s.take(1));
    try std.testing.expectEqual(@as(?usize, null), s.take(0));
    var big: Spans = .{ .size = 1 << 20 };
    try std.testing.expectEqual(@as(?usize, null), big.take((1 << 20) + 1));
    try std.testing.expectEqual(@as(?usize, 0), big.take(1 << 20));
}

test "a given-back span's room is taken again, first fit" {
    var s: Spans = .{ .size = 4096 };
    _ = s.take(512).?;
    const mid = s.take(1024).?;
    _ = s.take(512).?;
    s.release(mid);
    s.release(12345); // no span starts there
    try std.testing.expectEqual(@as(?usize, 512), s.take(768));
    try std.testing.expectEqual(@as(?usize, 2048), s.take(512)); // 1280..1536 is too small
    try std.testing.expectEqual(@as(?usize, 1280), s.take(256));
    try std.testing.expectEqual(@as(usize, 1536), s.free());
}

test "a span list holds at most max spans" {
    var s: Spans = .{ .size = 1 << 20 };
    for (0..Spans.max) |_| _ = s.take(1).?;
    try std.testing.expectEqual(@as(?usize, null), s.take(1));
}

test "the carveout is opt-in, sized in whole rows, with a source a value selects" {
    try std.testing.expectEqual(@as(?Request, null), try requested(false, null, null, null, null, null));
    try std.testing.expectEqual(@as(?Request, null), try requested(false, "daemon", null, null, null, "64")); // off: the source alone does nothing
    try std.testing.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = default_bytes }), try requested(true, null, null, null, null, null)); // drm is the default source
    try std.testing.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = default_bytes }), try requested(true, "drm", null, null, null, ""));
    try std.testing.expectEqual(@as(?Request, .{ .backend = .drm, .bytes = 64 << 20 }), try requested(true, "drm", null, null, null, "64"));
    try std.testing.expectEqual(@as(?Request, .{ .backend = .daemon, .bytes = default_bytes }), try requested(true, "daemon", null, null, null, null));
    try std.testing.expectEqual(@as(?Request, .{ .backend = .daemon, .bytes = 64 << 20 }), try requested(true, "daemon", null, null, null, "64"));
    try std.testing.expectError(error.InvalidEnvironment, requested(true, "drm", null, null, null, "lots"));
    try std.testing.expectError(error.InvalidEnvironment, requested(true, "magic", null, null, null, null)); // a third source is refused
    try std.testing.expectEqual(@as(?Request, null), try requested(false, "magic", null, null, null, null)); // off: not validated
    try std.testing.expectEqual(0, roundBytes(row_bytes - 1));
    try std.testing.expectEqual(row_bytes, roundBytes(row_bytes + 1));
}

test "the driver-facing code compiles on Linux" {
    if (comptime builtin.os.tag == .linux) std.testing.refAllDecls(Carveout);
}
