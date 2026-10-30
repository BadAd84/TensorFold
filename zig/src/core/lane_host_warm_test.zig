//! A background warm pass and the real request that arrives while it runs, through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

/// A request's tokens and how it finished, as the host reports them.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    tokens: usize = 0,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Chunks the fake ran for prompt-only streams.
const WarmChunks = struct {
    n: std.atomic.Value(usize) = .init(0),
    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        if (s.max_new == 0) _ = w.n.fetchAdd(1, .monotonic);
    }
};

test "a background warm pass steps aside for a request that arrives during it, and the request runs" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 48 };
    const Arrival = struct { // the real request arrives during the warm pass's first chunk
        engine: Engine,
        request: *const Request,
        box: *Box,
        sent: bool = false,
        fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.sent or s.max_new != 0 or chunk != 0) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.box, .event = Box.event }) catch {};
        }
    };
    var arrival: Arrival = .{ .engine = host.engine(), .request = &request, .box = &real };
    target.prefill_hook = Arrival.hook;
    target.prefill_hook_ctx = &arrival;
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, warm.wait()); // it stepped aside at its next chunk
    try std.testing.expectEqual(Reason.length, real.wait());
    try std.testing.expectEqual(@as(usize, 48), real.tokens);
}

test "a warm pass waits for a quiet engine: a request sent right after the reply goes first and the pass runs no chunk" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: WarmChunks = .{};
    target.prefill_hook = WarmChunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.warm_quiet = 30 * std.time.ns_per_s; // far past the next request: it must step aside, not time out
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 48 };
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    while (true) { // the engine thread has taken the pass: the request comes after it, as a client's next one does
        host.lock();
        const taken = host.queued.items.len == 0;
        host.unlock();
        if (taken) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try host.engine().submit(2, &request, .{ .ctx = &real, .event = Box.event });
    try std.testing.expectEqual(Reason.length, real.wait());
    try std.testing.expectEqual(@as(usize, 48), real.tokens);
    try std.testing.expectEqual(Reason.cancelled, warm.wait());
    try std.testing.expectEqual(@as(usize, 0), chunks.n.load(.monotonic)); // the GPU never ran a chunk of it
}

test "a warm pass with no request after it runs once the engine has been quiet" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: WarmChunks = .{};
    target.prefill_hook = WarmChunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.warm_quiet = 20 * std.time.ns_per_ms;
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const t0 = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    try std.testing.expectEqual(Reason.length, warm.wait());
    try std.testing.expect(std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - t0 >= 20 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(usize, 3), chunks.n.load(.monotonic)); // every chunk ran
}

test "a warm pass its cache keeps nothing of runs no chunk: no planned mark, and the end a lookahead short" {
    const gpa = std.testing.allocator;
    const pc = api.prompt_cache;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: WarmChunks = .{};
    target.prefill_hook = WarmChunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
            return error.NothingToKeep;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(_: *anyopaque, _: pc.Saved) void {}
    };
    // Flash Next's rules: states at chunk starts only, a token of lookahead; a short prompt's only start is 0
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .lookahead = 1, .planned = true, .warm = .on }, 400);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var warm: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true, .history_len = prompt.len, .born = 4 };
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, warm.wait());
    try std.testing.expectEqual(@as(usize, 0), chunks.n.load(.monotonic)); // the GPU never ran a chunk of it
}

/// Status polls taken at each prompt chunk, the warm pass's apart from the request's.
const FillPolls = struct {
    engine: Engine = undefined,
    warm: std.ArrayList(api.Status) = .empty,
    real: std.ArrayList(api.Status) = .empty,
    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const p: *@This() = @ptrCast(@alignCast(ctx));
        var status: api.Status = .{};
        p.engine.status(&status, &.{});
        (if (s.max_new == 0) &p.warm else &p.real).append(std.testing.allocator, status) catch {};
    }
};

test "a warm pass is no fill on the status page: its chunks leave the fill keys empty, a request's own are counted" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    try host.start();
    defer host.stop();
    var polls: FillPolls = .{ .engine = host.engine() };
    defer polls.warm.deinit(gpa);
    defer polls.real.deinit(gpa);
    target.prefill_hook = FillPolls.hook;
    target.prefill_hook_ctx = &polls;
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    try std.testing.expectEqual(Reason.length, warm.wait());
    try host.engine().submit(2, &request, .{ .ctx = &real, .event = Box.event });
    try std.testing.expectEqual(Reason.length, real.wait());
    try std.testing.expectEqual(@as(usize, 3), polls.warm.items.len); // one poll a chunk
    for (polls.warm.items) |s| {
        try std.testing.expect(s.fill_prompt_tokens == null and s.fill_tokens_left == null);
        try std.testing.expect(s.fill_run == 0 and !s.fill_unknown);
    }
    try std.testing.expectEqual(@as(usize, 3), polls.real.items.len);
    for (polls.real.items) |s| {
        try std.testing.expectEqual(@as(?u32, prompt.len), s.fill_prompt_tokens);
        try std.testing.expect(s.fill_run != 0);
    }
}

/// A paused turn on a warm host (bytes 64 + 4 a token, room for four 8-token states): the regenerate's cached tokens.
fn pausedTurn(lookahead: u32) !struct { cached: ?u32, same: bool } {
    const gpa = std.testing.allocator;
    const pc = api.prompt_cache;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .lookahead = lookahead, .warm = if (lookahead == 0) .on else .waits, .min_prompt = 0, .min_gap = 4 }, 400);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Run = struct {
        mutex: std.Io.Mutex = .init,
        cached: ?u32 = null,
        tokens: [8]u32 = undefined,
        n: usize = 0,
        done: bool = false,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| for (t) |x| if (b.n < b.tokens.len) {
                    b.tokens[b.n] = x;
                    b.n += 1;
                },
                .finished => b.done = true,
                else => {},
            }
        }
        fn go(b: *@This(), engine: Engine, id: Id, r: *const Request) !void {
            try engine.submit(id, r, .{ .ctx = b, .event = event });
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) return;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const x = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 };
    const z = [_]u32{ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209 };
    const y = [_]u32{ 300, 301, 302, 303, 304, 305, 306, 307, 308, 309 };
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; // history 8, then a two-token generation prompt
    const warm = t1[0..9].* ++ [_]u32{ 50, 51, 52, 53, 54, 55 }; // the reply as sent and the next turn's opening
    var runs: [6]Run = @splat(.{});
    for ([_][]const u32{ &x, &z }, 0..) |p, i| try runs[i].go(e, i + 1, &.{ .prompt = p, .max_tokens = 2, .history_len = 8 });
    try runs[2].go(e, 3, &.{ .prompt = &t1, .max_tokens = 2, .history_len = 8 });
    try runs[3].go(e, 4, &.{ .prompt = &warm, .max_tokens = 0, .background = true, .history_len = warm.len, .born = t1.len });
    try runs[4].go(e, 5, &.{ .prompt = &y, .max_tokens = 2, .history_len = 8 }); // its keep needs room too
    try runs[5].go(e, 6, &.{ .prompt = &t1, .max_tokens = 2, .history_len = 8 }); // regenerated
    try std.testing.expect(store.held <= store.budget);
    return .{ .cached = runs[5].cached, .same = runs[5].n == 2 and std.mem.eql(u32, runs[5].tokens[0..2], runs[2].tokens[0..2]) };
}

test "a finished warm pass keeps its states with its request's length: the history it extends outlives its keep and later ones" {
    for ([_]u32{ 0, 1 }) |lookahead| { // the 27B keeps the pass's end; Nemotron on CUDA a mark one token before it
        const got = try pausedTurn(lookahead);
        try std.testing.expectEqual(@as(?u32, 8), got.cached);
        try std.testing.expect(got.same);
    }
}

/// Warm-pass chunks run while another request's prompt still fills (a stepped backend), the pass sent during its first.
fn warmBesideFill(quiet: u64) !usize {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.warm_quiet = quiet; // 0: qwen27, qwen4 and Flash Next's; else a filling prompt never wakes the wait
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    const Arrival = struct { // the warm pass arrives during the request's first fill chunk
        engine: Engine,
        request: *const Request,
        box: *Box,
        sent: bool = false,
        filled: bool = false, // the request's last fill chunk ran
        beside: usize = 0,
        fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (s.max_new == 0) {
                if (!a.filled) a.beside += 1;
                return;
            }
            if (chunk == 2) a.filled = true;
            if (a.sent or chunk != 0) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.box, .event = Box.event }) catch {};
        }
    };
    var arrival: Arrival = .{ .engine = host.engine(), .request = &warm_request, .box = &warm };
    target.prefill_hook = Arrival.hook;
    target.prefill_hook_ctx = &arrival;
    try host.engine().submit(1, &request, .{ .ctx = &real, .event = Box.event });
    try std.testing.expectEqual(Reason.length, real.wait());
    try std.testing.expectEqual(@as(usize, 4), real.tokens);
    try std.testing.expectEqual(Reason.cancelled, warm.wait()); // dropped before it touched the backend
    return arrival.beside;
}

test "a warm pass sent while another prompt fills by steps is dropped untouched, with or without a quiet wait" {
    try std.testing.expectEqual(@as(usize, 0), try warmBesideFill(0));
    try std.testing.expectEqual(@as(usize, 0), try warmBesideFill(1));
}

test "a warm pass taken into a batch behind real prompts is dropped untouched; they run" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: WarmChunks = .{};
    target.prefill_hook = WarmChunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock()); // short prompts batch (many_rows 8)
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3, .warm_turns = true });
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var boxes: [3]Box = @splat(.{});
    const requests = [3]Request{ .{ .prompt = &prompt, .max_tokens = 4 }, .{ .prompt = &prompt, .max_tokens = 4 }, .{ .prompt = &prompt, .max_tokens = 0, .background = true } };
    for (&requests, &boxes, 0..) |*r, *b, i| try host.engine().submit(i + 1, r, .{ .ctx = b, .event = Box.event }); // queued before the thread
    try host.start();
    defer host.stop();
    for (boxes[0..2]) |*b| try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(Reason.cancelled, boxes[2].wait());
    try std.testing.expectEqual(@as(usize, 0), chunks.n.load(.monotonic)); // it never ran ahead of the batch
}

/// Conversation A's latest state, then B's reply pass and a request for C on a store with room for one state.
fn otherLatest(spares: bool) !struct { a_after_pass: bool, b_kept: bool, a_after_request: bool, c_kept: bool, flag: bool } {
    const gpa = std.testing.allocator;
    const pc = api.prompt_cache;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    // GLM's rules (a token of lookahead, warm .waits); 150 bytes: an 8-row state (96) leaves no room for a 9-row (100)
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .lookahead = 1, .warm = .waits, .min_prompt = 0, .min_gap = 4 }, 150);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.cache = &store;
    host.warm_spares = spares;
    try host.start();
    defer host.stop();
    const a = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 };
    const b = [_]u32{ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209 };
    const c = [_]u32{ 300, 301, 302, 303, 304, 305, 306, 307, 308, 309 };
    const short = [_]u32{7}; // a pass with no mark: it looks up, keeps nothing and ends
    const Has = struct {
        fn at(st: *pc.Store, h: *LaneHost, p: []const u32, row: u32) bool {
            h.lock(); // the engine thread is idle between these requests: the store holds still
            defer h.unlock();
            for (st.entries.items) |e| if (!e.expired and e.at == row and std.mem.eql(u32, e.tokens, p[0 .. row + 1])) return true;
            return false;
        }
    };
    const e = host.engine();
    const requests = [4]Request{
        .{ .prompt = &a, .max_tokens = 2, .history_len = 8 }, // A's turn: its latest state at its history
        .{ .prompt = &b, .max_tokens = 0, .background = true, .history_len = b.len, .born = b.len }, // B's reply pass
        .{ .prompt = &short, .max_tokens = 0, .background = true, .history_len = 1, .born = 1 },
        .{ .prompt = &c, .max_tokens = 2, .history_len = 8 }, // C's turn: a request
    };
    var boxes: [4]Box = @splat(.{});
    var flag = false; // the store's guard left set after a pass ended, where a later caller's eviction would see it
    var seen: [4]struct { a: bool, b: bool, c: bool } = undefined;
    for (&requests, &boxes, &seen, 0..) |*r, *x, *got, i| {
        try e.submit(i + 1, r, .{ .ctx = x, .event = Box.event });
        _ = x.wait();
        flag = flag or store.spares_only;
        got.* = .{ .a = Has.at(&store, &host, &a, 8), .b = Has.at(&store, &host, &b, 9), .c = Has.at(&store, &host, &c, 8) };
    }
    return .{ .a_after_pass = seen[1].a, .b_kept = seen[1].b, .a_after_request = seen[3].a, .c_kept = seen[3].c, .flag = flag };
}

test "warm_spares: a reply pass evicts no other conversation's latest state and keeps nothing; a request still evicts" {
    const on = try otherLatest(true);
    try std.testing.expect(on.a_after_pass and !on.b_kept and !on.flag);
    try std.testing.expect(!on.a_after_request and on.c_kept); // a real request's keep evicts as before
    const off = try otherLatest(false); // the guard off (other families): the pass's keep pushes A's latest state out
    try std.testing.expect(!off.a_after_pass and off.b_kept);
}
