//! TF_GLM_OVN_DISK_CARRY, rank 0: the carry key (the identity less its kernels), its proof, and the store's directory.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const cuda_disk = @import("cuda_disk.zig");
const disk = @import("cuda_glm_disk.zig");
const rows = api.prompt_cache.rows;
const dc = glm.disk_carry;
const record = lanes.carry_record;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Allocator = std.mem.Allocator;

/// A start with the knob on: what the directory was chosen by (its key, rank 0's proof and keying), and the choice.
pub const Carry = struct { want: record.Record, choice: rows.carry.Choice };

/// This start's key and proof, then its directory (rows_carry.choose); null if the probe failed: its own, as knob off.
pub fn start(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, backend: lanes.backend.Backend, e: *Engine, facts: []const u64, probed: u64, id: u64, rules: api.prompt_cache.Rules, local: bool) !?Carry {
    var parts: std.ArrayList(u64) = .empty;
    try parts.append(a, probed); // the identity's own probe state, every rank's bytes
    const proof = prove(a, backend, e, &parts) catch |err| {
        std.log.warn("glm: TF_GLM_OVN_DISK_CARRY: no carry this start, the carry probe failed ({s})", .{@errorName(err)});
        return null;
    };
    const want: record.Record = .{ .dir = rows.carry.dirOf(id), .key = try cuda_disk.carryOf(a, io, ctx, o.dir, facts, parts.items), .proof = proof, .block = rows.block_tokens, .lookahead = rules.lookahead, .planned = @intFromBool(rules.planned) };
    const asked: ?rows.carry.Ranks = if (local) .{ .ctx = e, .check = check } else null;
    const c = rows.carry.choose(root, want, asked);
    std.log.info("glm: TF_GLM_OVN_DISK_CARRY: {s} (directory {x:0>16}, carry key {x:0>16}, {d} probe hashes)", .{ c.why, c.id, want.key, parts.items.len });
    return .{ .want = want, .choice = c };
}

/// After the store opened in `r.dir`: rank 0's record there, and with TF_GLM_OVN_DISK_LOCAL every other rank's own.
pub fn records(e: *Engine, r: *const rows.Rows, c: Carry, local: bool) !void {
    var mine = c.want;
    mine.dir = r.keying.identity; // its own directory, or the adopted one (whose record proved the same)
    if (record.refusal(record.read(r.dir), mine) != null) record.write(r.dir, mine) catch |err| std.log.warn("glm: TF_GLM_OVN_DISK_CARRY: the record in {s} was not written ({s})", .{ r.dir, @errorName(err) });
    if (!local) return;
    const rest = disk.Rest.of(e);
    defer rest.done();
    try dc.records(e, r.dir, mine.dir, mine.key);
}

/// Rows_carry's question to every other rank (TF_GLM_OVN_DISK_LOCAL): its own directory proves the key and its proof.
fn check(ctx: *anyopaque, dir: []const u8, id: u64, key: u64) anyerror!bool {
    const e: *Engine = @ptrCast(@alignCast(ctx));
    const rest = disk.Rest.of(e);
    defer rest.done();
    return dc.check(e, dir, id, key);
}

/// The carry probe through the engine, each rank's image of each state hashed into `parts`; rank 0's own hashes folded.
fn prove(a: Allocator, backend: lanes.backend.Backend, e: *Engine, parts: *std.ArrayList(u64)) !u64 {
    var buf: [2]u32 = undefined;
    const marks = dc.marks(e.capacity, &buf);
    if (marks.len == 0) return error.CarryContextShort;
    const toks = try a.alloc(u32, marks[marks.len - 1] + dc.tail);
    for (toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000); // the identity probe's prompt, further
    const Hook = struct {
        e: *Engine,
        saved: [2]*anyopaque = undefined,
        n: usize = 0,
        failed: ?anyerror = null,
        fn at(p: *anyopaque, _: *lanes.Stream, m: u32) void {
            const h: *@This() = @ptrCast(@alignCast(p));
            if (h.n == h.saved.len) {
                h.failed = error.ProbeMarks;
                return;
            }
            h.saved[h.n] = glm.native.snap_save(h.e, null, m) catch |err| {
                h.failed = err;
                return;
            };
            h.n += 1;
        }
    };
    var hook: Hook = .{ .e = e };
    defer for (hook.saved[0..hook.n]) |saved| glm.native.snap_drop(e, saved);
    var s = try lanes.Stream.init(a, .{ .id = "carry probe", .prompt = toks, .max_new = 1, .drafts = e.mtp, .reuse = .{ .marks = marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    // the ranks back asleep after the pass and its hashes on every way out, as the identity's probe leaves them
    defer glm.disk_ranks.rest(e) catch |err| std.log.warn("glm: the ranks did not sleep after the carry probe ({s})", .{@errorName(err)});
    {
        defer backend.vtable.release(backend.ptr, &s); // once, whether the pass ran or failed
        try backend.vtable.prefill(backend.ptr, &s);
    }
    if (hook.failed) |err| return err;
    if (hook.n != marks.len) return error.ProbeUnmarked;
    var out: [64]u64 = undefined;
    var proof: u64 = 0;
    for (hook.saved[0..hook.n], 0..) |saved, i| {
        const k: *Kept = @ptrCast(@alignCast(saved));
        const hs = try dc.hashes(e, k.id, i == 0, &out);
        for (hs) |v| if (v == record.unhashed) return error.CarryUnhashed;
        proof = record.fold(proof, hs[0]);
        try parts.appendSlice(a, hs);
    }
    return proof;
}
