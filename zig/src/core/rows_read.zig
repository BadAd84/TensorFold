//! Disk states read back: each block and the state's own file checked, then copied into a new kept state of its family.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const image = @import("lanes").rows_image;
const chain = @import("rows_chain.zig");
const file = @import("lanes").rows_file;
const rows_store = @import("rows_store.zig");
const write = @import("rows_write.zig");
const select = @import("rows_select.zig");
const prefetch = @import("rows_prefetch.zig");
const local = @import("rows_local.zig");
const readers = @import("lanes").disk_readers;
const Rows = rows_store.Rows;
const Name = chain.Name;

/// Disk state `i` read into a new kept state of its family; a file that fails a check goes, with every state naming it.
/// TF_GLM_OVN_DISK_LOCAL: every other rank reads its own part meanwhile, and a rank without its part fails it for all.
pub fn read(r: *Rows, i: usize) !pc.Saved {
    const f = r.family.vtable;
    const s = r.states.items[i];
    const t0 = write.nowNs();
    var plan = try prefetch.Plan.of(r, r.gpa, &s);
    defer plan.deinit(r.gpa);
    const ahead = if (r.fetch) |p| p.take(s.name, &plan) else null; // read on the I/O thread, its files checked there
    const saved = try r.img.blank(r.family.ptr, s.at);
    errdefer f.drop(r.family.ptr, saved);
    if (r.mirror) |m| try begin(m, r, &plan, s.at, saved);
    const got = fill(r, &plan, ahead, saved);
    if (r.mirror) |m| if (!try m.end(m.ctx, if (got) |_| true else |_| false)) {
        _ = try got; // rank 0's own failure already forgot its file and the states naming it
        r.counts.failed += 1;
        pc.note("imprint: a rank lacked its part of {d} disk tokens; every rank forgets the state", .{s.at});
        if (r.find(s.name)) |k| r.remove(k);
        return error.RankPartMissing;
    };
    const bytes = try got;
    const ns = if (ahead) |a| a.ns else write.nowNs() - t0;
    if (ns > 0 and bytes > 64 << 20) r.read_rate = 0.5 * r.read_rate + 0.5 * @as(f64, @floatFromInt(bytes)) / (@as(f64, @floatFromInt(ns)) / 1e9);
    r.counts.read += bytes;
    r.counts.reads += 1;
    r.touch(i);
    if (!@import("builtin").is_test) std.log.info("imprint: read {d} tokens from disk ({d} MiB in {d:.0} ms{s})", .{ s.at, bytes >> 20, @as(f64, @floatFromInt(ns)) / 1e6, if (ahead != null) ", read ahead" else "" });
    return saved;
}

/// The plan's files into `saved`: from a read ahead, by the reader pool, or one after another; their bytes on disk.
fn fill(r: *Rows, plan: *const prefetch.Plan, ahead: ?*prefetch.Read, saved: pc.Saved) !u64 {
    if (ahead) |a| {
        for (plan.pieces) |p| try r.img.scatter(r.family.ptr, saved, plan.rangesOf(p), a.payload(p));
        return a.disk;
    }
    if (r.readers > 1 or r.mirror != null) return piped(r, plan, saved);
    var bytes: u64 = 0;
    for (plan.pieces) |p| bytes += try one(r, p.name, p.kind, prefetch.ext(p.kind), p.at, plan.rangesOf(p), saved);
    return bytes;
}

/// TF_GLM_OVN_DISK_LOCAL: the other ranks start on their parts of the plan's files (blocks, then the own file).
fn begin(m: *const local.Mirror, r: *Rows, plan: *const prefetch.Plan, at: u32, saved: pc.Saved) !void {
    const files = try r.gpa.alloc(local.File, plan.pieces.len);
    defer r.gpa.free(files);
    for (plan.pieces, files) |p, *x| x.* = .{ .name = p.name, .kind = p.kind, .window = p.window, .k = p.k };
    try m.begin(m.ctx, saved, at, files);
}

/// TF_GLM_OVN_DISK_READERS (and DISK_LOCAL): the plan's files read by `r.readers` threads into slots, each scattered
/// here as it fills, the state's own file last (it seals the state); a file that fails its check goes as in `one`.
fn piped(r: *Rows, plan: *const prefetch.Plan, saved: pc.Saved) !u64 {
    const n = plan.pieces.len;
    var slot: u64 = 0;
    for (plan.pieces[0 .. n - 1]) |p| slot = @max(slot, p.len);
    if (r.mirror) |m| if (r.readers_stage.pin == null) {
        r.readers_stage.pin = m.pin;
    };
    try r.readers_stage.ensure(slot, 2 * @as(usize, r.readers), plan.pieces[n - 1].len);
    const disk = try r.gpa.alloc(u64, n);
    defer r.gpa.free(disk);
    @memset(disk, 0);
    var c: Piped = .{ .r = r, .plan = plan, .saved = saved, .disk = disk };
    var failed: ?usize = null;
    readers.pipe(r.readers, n, n - 1, &r.readers_stage, &c, &failed) catch |err| {
        const k = failed orelse return err;
        return bad(r, plan.pieces[k].name, prefetch.ext(plan.pieces[k].kind), err);
    };
    var bytes: u64 = 0;
    for (disk) |d| bytes += d;
    return bytes;
}

/// One restore's files as the reader pool takes them: each read and checked on a reader thread, scattered on this one.
const Piped = struct {
    r: *Rows,
    plan: *const prefetch.Plan,
    saved: pc.Saved,
    disk: []u64,

    pub fn span(c: *Piped, i: usize) u64 {
        return std.mem.alignForward(u64, c.plan.pieces[i].len, file.alignment);
    }

    pub fn read(c: *Piped, i: usize, into: file.Stage) ![]const u8 {
        const p = c.plan.pieces[i];
        var buf: [1200]u8 = undefined;
        const path = try file.path(&buf, c.r.dir, &chain.hex(p.name), prefetch.ext(p.kind), false);
        var o = try file.open(path, p.kind, p.name);
        defer o.close();
        if (o.trailer.payload != p.len or o.trailer.block != c.r.keying.block or o.trailer.at != p.at) return error.RowsRead;
        const got = try file.payload(&o, into);
        c.disk[i] = o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer);
        return got;
    }

    pub fn consume(c: *Piped, i: usize, payload: []const u8) !void {
        try c.r.img.scatter(c.r.family.ptr, c.saved, c.plan.rangesOf(c.plan.pieces[i]), payload);
    }
};

/// One file's payload read past the page cache, checked, and copied into `ranges` of `saved`; its bytes on disk.
fn one(r: *Rows, name: Name, kind: file.Kind, ext: []const u8, at: u32, ranges: []const image.Range, saved: pc.Saved) !u64 {
    const n = image.total(ranges);
    var buf: [1200]u8 = undefined;
    const p = try file.path(&buf, r.dir, &chain.hex(name), ext, false);
    var o = file.open(p, kind, name) catch |err| return bad(r, name, ext, err);
    defer o.close();
    if (o.trailer.payload != n or o.trailer.block != r.keying.block or o.trailer.at != at) return bad(r, name, ext, error.RowsRead);
    const stage = try r.staged(n);
    const got = file.payload(&o, stage) catch |err| return bad(r, name, ext, err);
    try r.img.scatter(r.family.ptr, saved, ranges, got);
    return o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer);
}

/// A file that failed its checks deleted, with every state that names it; the error passed on.
fn bad(r: *Rows, name: Name, ext: []const u8, err: anyerror) anyerror {
    r.counts.failed += 1;
    pc.note("imprint: a disk {s} failed its check ({s}); it and the states naming it are forgotten", .{ if (std.mem.eql(u8, ext, ".blk")) "block" else "state", @errorName(err) });
    write.fence(r);
    r.unlink(name, if (std.mem.eql(u8, ext, ".blk")) file.Kind.block else file.Kind.state);
    if (std.mem.eql(u8, ext, ".blk")) _ = r.blocks.remove(chain.stem(name));
    var i: usize = 0;
    var wants: std.ArrayList(write.Want) = .empty;
    defer wants.deinit(r.gpa);
    while (i < r.states.items.len) {
        const s = &r.states.items[i];
        var names = std.mem.eql(u8, &s.name, &name);
        if (!names) {
            wants.clearRetainingCapacity();
            r.wantsOf(r.gpa, s, &wants) catch {};
            for (wants.items) |w| names = names or std.mem.eql(u8, &w.name, &name);
        }
        if (names) r.remove(i) else i += 1;
    }
    return err;
}

/// State `i`'s own file still on disk with its trailer, checked before kept states go to make room for reading it.
fn present(r: *Rows, i: usize) !void {
    const st = r.states.items[i];
    var buf: [1200]u8 = undefined;
    const p = try file.path(&buf, r.dir, &chain.hex(st.name), ".st", false);
    var o = file.open(p, .state, st.name) catch |err| return bad(r, st.name, ".st", err);
    defer o.close();
    if (o.trailer.block != r.keying.block or o.trailer.at != st.at) return bad(r, st.name, ".st", error.RowsRead);
}

/// A disk state reaching further than `have`, read back into a new RAM entry where that costs less than prefilling.
pub fn recall(s: *pc.Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*pc.Entry) ?*pc.Entry {
    const r = s.disk orelse return have;
    write.reap(r);
    const chosen = select.choose(s, prompt, starts, spans, have).restore orelse return have;
    const name = r.states.items[chosen].name;
    const bytes = s.family.vtable.bytes(s.family.ptr, r.states.items[chosen].at);
    if (bytes > s.budget or s.most == 0) return have;
    const at = r.states.items[chosen].at;
    if (!ready(s, at, bytes)) present(r, chosen) catch return have; // nothing goes to make room for a state whose file is gone
    if (r.spill != .off) r.touch(chosen); // the newest: the spills that make room for it push older states off first
    while (!ready(s, at, bytes)) {
        if (s.family.vtable.trim) |trim| {
            const spare = s.spare();
            trim(s.family.ptr, s.budget -| s.held -| bytes);
            if (s.spare() < spare) continue;
        }
        s.remove(s.victimBut(have, prompt, born) orelse return have);
        s.counts.evicted += 1;
    }
    const i = r.find(name) orelse return have; // a spill that made room went past the disk cap and took it
    const st = r.states.items[i];
    const e = s.gpa.create(pc.Entry) catch return have;
    e.* = .{ .tokens = &.{}, .at = st.at, .saved = undefined, .bytes = bytes, .born = born, .used = s.clock, .last = &.{}, .shared = st.shared };
    var ok = false;
    defer if (!ok) {
        s.gpa.free(e.tokens);
        s.gpa.free(e.decode_spans);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    };
    e.tokens = s.gpa.dupe(u32, st.tokens) catch return have;
    e.decode_spans = s.gpa.dupe(modes.Span, st.spans) catch return have;
    e.last = s.gpa.dupe(u32, prompt) catch return have;
    e.saved = read(r, i) catch return have;
    if (s.family.vtable.charged) |f| e.bytes = f(s.family.ptr, e.saved);
    @import("prompt_cache_index.zig").add(s, e) catch {
        s.family.vtable.drop(s.family.ptr, e.saved);
        return have;
    };
    @import("prompt_cache_policy.zig").link(s, e);
    ok = true;
    s.held += e.bytes;
    return e;
}

fn ready(s: *const pc.Store, at: u32, bytes: u64) bool {
    if (s.held > s.budget or s.entries.items.len >= s.most) return false;
    const f = s.family.vtable;
    // a pool without trim reads into spare storage that takes the state, as a keep would
    if (f.trim == null) if (f.reuses) |reuses| if (reuses(s.family.ptr, at)) return true;
    const room = s.budget - s.held;
    const spare = s.spare();
    return spare <= room and bytes <= room - spare;
}

/// How much of a prompt the server resumes, and from where, with the estimated wait for the rest.
pub const Resume = struct {
    tokens: u32 = 0, // the prompt's tokens
    ram: u32 = 0, // tokens a kept state in RAM covers
    disk: u32 = 0, // tokens a state on disk covers
    disk_bytes: u64 = 0,
    from: enum { ram, disk, none } = .none,
    seconds: f64 = 0, // the estimated wait: the disk read, then prefilling what is left
};

/// Where `prompt`'s next request would resume, read on the engine thread.
pub fn resumable(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) Resume {
    var out: Resume = .{ .tokens = @intCast(prompt.len) };
    const have = s.find(prompt, starts, spans);
    if (have) |e| out.ram = e.at;
    out.from = if (out.ram > 0) .ram else .none;
    const r = s.disk orelse return finishResume(s, out);
    const chosen = select.choose(s, prompt, starts, spans, have);
    if (chosen.disk) |i| {
        const st = r.states.items[i];
        out.disk = st.at;
        out.disk_bytes = st.own +| st.rows;
    }
    if (chosen.restore != null) {
        out.from = .disk;
        out.seconds = chosen.read_seconds;
    }
    return finishResume(s, out);
}

fn finishResume(s: *const pc.Store, x: Resume) Resume {
    var out = x;
    const start = if (out.from == .disk) out.disk else out.ram;
    const rate = if (s.disk) |r| r.prefill_rate else 0;
    if (select.knownRate(rate)) {
        const total = out.seconds + @as(f64, @floatFromInt(out.tokens -| start)) / rate;
        if (std.math.isFinite(total)) out.seconds = total;
    }
    return out;
}
