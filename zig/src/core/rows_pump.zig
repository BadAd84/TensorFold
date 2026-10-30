//! TF_GLM_OVN_DISK_PUMP=N: a kept state's write staged at most N pieces a round into a reusable ring as it is written.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const image = @import("lanes").rows_image;
const io = @import("rows_io.zig");
const sync = @import("rows_write_sync.zig");
const Rows = @import("rows_store.zig").Rows;

/// Ring slots a round's pieces may use: the worker writes a few rounds behind the engine before staging waits.
const rounds_ahead = 4;

/// The smallest ring slot (an own file's piece): a state whose blocks are small stages its own file in few steps.
const min_slot: u64 = 4 << 20;

/// The knob, read when the store opens: pieces staged a round (unset, empty or 0: off, a state staged in one step).
pub fn knob() !u32 {
    const v = platform.c.getenv("TF_GLM_OVN_DISK_PUMP") orelse return 0;
    const s = std.mem.span(v);
    if (s.len == 0) return 0;
    return std.fmt.parseInt(u32, s, 10) catch error.BadOvnKnob;
}

/// One block file's group and index (its ranges are taken when it is staged).
pub const Block = struct { window: u32, k: u32 };

/// The job the worker holds while the engine stages its payloads: block files into ring slots, then the own file.
pub const Pump = struct {
    job: *io.Job = undefined,
    entry: *pc.Entry,
    parts: []image.Part,
    blocks: []Block, // the job's block files in order
    own: []image.Range, // the own file's image ranges
    next: usize = 0, // the next file to stage
    own_i: usize = 0,
    own_off: u64 = 0,
    own_done: u64 = 0, // own payload bytes staged
};

/// A pump for `prepared` whose job writes `count` files (the last its own); the worker's ring made on first use.
pub fn make(r: *Rows, w: *io.Worker, prepared: *const sync.Job, count: usize) !*Pump {
    var slot: u64 = min_slot;
    for (prepared.wants) |want| slot = @max(slot, want.bytes);
    if (w.ring) |g| {
        if (g.slot < slot) return error.SnapshotSize;
    } else w.ring = try io.Ring.create(@intCast(slot), @as(u64, r.pump_blocks) * rounds_ahead);
    const p = try r.gpa.create(Pump);
    errdefer r.gpa.destroy(p);
    const parts = try r.gpa.dupe(image.Part, prepared.parts);
    errdefer r.gpa.free(parts);
    const blocks = try r.gpa.alloc(Block, count - 1);
    errdefer r.gpa.free(blocks);
    p.* = .{ .entry = prepared.entry, .parts = parts, .blocks = blocks, .own = try r.gpa.dupe(image.Range, prepared.own) };
    return p;
}

pub fn free(r: *Rows, p: *Pump) void {
    r.gpa.free(p.parts);
    r.gpa.free(p.blocks);
    r.gpa.free(p.own);
    r.gpa.destroy(p);
}

pub fn active(r: *const Rows) bool {
    return r.pump != null;
}

/// Whether a step can stage now: the own file's pieces need no slot, a block file needs a free one.
pub fn ready(r: *const Rows) bool {
    const p = r.pump orelse return false;
    if (p.next >= p.blocks.len) return true;
    const g = (r.worker orelse return false).ring orelse return false;
    return g.head - g.tail.load(.acquire) < g.count;
}

/// The staging given up: the worker fails the job at its first file not staged, and the reap frees its slots.
pub fn abandon(r: *Rows) void {
    const p = r.pump orelse return;
    p.job.abandoned.store(true, .release);
    r.pump = null;
    free(r, p);
}

/// An entry leaving RAM: the save staging from it stops.
pub fn dropped(r: *Rows, e: *pc.Entry) void {
    if (r.pump) |p| if (p.entry == e) abandon(r);
}

/// A job reaped (engine thread): its unreturned ring slots freed; the pump staging it, if any, ends with it.
pub fn reaped(r: *Rows, j: *const io.Job) void {
    if (r.worker) |w| if (w.ring) |g| g.reclaim(j);
    if (r.pump) |p| if (p.job == j) {
        r.pump = null;
        free(r, p);
    };
}

/// Up to the knob's pieces staged (a block file each, or a slot's bytes of the own file); false when none could be.
pub fn step(r: *Rows) bool {
    const p = r.pump orelse return false;
    const w = r.worker orelse return false;
    const j = p.job;
    if (j.done.load(.acquire) or j.generation != w.generation.load(.acquire)) return false; // the reap ends it
    if (r.img.whole) |whole| if (!whole(r.family.ptr, p.entry.saved)) {
        fail(r, p, error.NotWhole);
        return true;
    };
    var n: u32 = 0;
    while (n < r.pump_blocks and p.next < j.files.len) : (n += 1) {
        const more = piece(r, w.ring.?, p) catch |err| {
            fail(r, p, err);
            return true;
        };
        if (!more) break;
    }
    if (p.next == j.files.len) {
        r.pump = null;
        free(r, p);
    }
    return n > 0;
}

fn fail(r: *Rows, p: *Pump, err: anyerror) void {
    r.counts.failed += 1;
    pc.note("imprint: staging {d} tokens failed ({s}); the save is skipped", .{ p.job.at, @errorName(err) });
    abandon(r);
}

/// One piece staged: the next block file into a ring slot, or the own file's next slot of bytes; false: ring full.
fn piece(r: *Rows, g: *io.Ring, p: *Pump) !bool {
    const j = p.job;
    const f = &j.files[p.next];
    var ranges: std.ArrayList(image.Range) = .empty;
    defer ranges.deinit(r.gpa);
    if (p.next < p.blocks.len) {
        const b = p.blocks[p.next];
        try image.blockRanges(p.parts, r.keying.block, b.window, b.k, &ranges, r.gpa);
        const bytes: usize = @intCast(image.total(ranges.items));
        if (bytes > g.slot) return error.SnapshotSize;
        const s = g.take() orelse return false;
        j.slots += 1;
        f.payload = s[0..bytes];
        f.ring = true;
        try r.img.gather(r.family.ptr, p.entry.saved, ranges.items, f.payload);
        if (r.mirror) |m| try m.file(m.ctx, p.entry.saved, j.at, .{ .name = f.trailer.name, .kind = .block, .window = b.window, .k = b.k });
        p.next += 1;
        j.staged_files.store(p.next, .release);
        return true;
    }
    var n: u64 = 0;
    while (p.own_i < p.own.len and n < g.slot) {
        const x = p.own[p.own_i];
        const take = @min(x.len - p.own_off, g.slot - n);
        try ranges.append(r.gpa, .{ .off = x.off + p.own_off, .len = take });
        n += take;
        p.own_off += take;
        if (p.own_off == x.len) {
            p.own_i += 1;
            p.own_off = 0;
        }
    }
    if (n > 0) try r.img.gather(r.family.ptr, p.entry.saved, ranges.items, f.payload[@intCast(p.own_done)..][0..@intCast(n)]);
    p.own_done += n;
    if (p.own_i == p.own.len) {
        // TF_GLM_OVN_DISK_LOCAL: every other rank stages its own part of the state's file once rank 0's is whole
        if (r.mirror) |m| try m.file(m.ctx, p.entry.saved, j.at, .{ .name = j.name, .kind = .state });
        p.next += 1;
        j.staged_files.store(p.next, .release);
    }
    return true;
}
