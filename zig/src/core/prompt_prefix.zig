//! Whether kept ids begin a prompt, and how many ids two lists share: every walk over kept states asks here.
const std = @import("std");
const platform = @import("sys");

// TF_GLM_OVN_KEPT_WALK=1 (production's IdsOf and common_prefix): the same answers from fewer compared ids.
const knob = "TF_GLM_OVN_KEPT_WALK";

/// Ids `common` compares a block before it looks for the first difference inside that block.
const block = 1024;

var read = false;
var gate = false;

/// Whole-list compares this thread ran (`starts` past its gate, one a `common`): the tests and the walk bench read it.
pub threadlocal var full: u64 = 0;

/// The knob, read once per process (unset, empty or "0": off); every rank and lane sees the same environment.
pub fn gated() bool {
    if (!read) {
        const v = platform.c.getenv(knob);
        gate = parse(if (v) |p| std.mem.span(p) else "");
        read = true;
    }
    return gate;
}

fn parse(s: []const u8) bool {
    return s.len > 0 and !std.mem.eql(u8, s, "0");
}

/// Tests only: the knob reads as `on` from now on, as if the environment held it.
pub fn fake(on: bool) void {
    if (!@import("builtin").is_test) @compileError("prompt_prefix.fake is for tests");
    read = true;
    gate = on;
}

/// `std.mem.eql(u32, prompt[0..tokens.len], tokens)`, false when `tokens` is longer; gated, the last id goes first.
pub fn starts(prompt: []const u32, tokens: []const u32) bool {
    return if (gated()) ends(prompt, tokens) else whole(prompt, tokens);
}

/// `starts` with the last id compared first, knob or not (the snapshot index's lookup always asked this way).
pub fn ends(prompt: []const u32, tokens: []const u32) bool {
    const n = tokens.len;
    if (n > prompt.len) return false;
    if (n > 0 and prompt[n - 1] != tokens[n - 1]) return false; // a prefix ends on the prompt's id at its length
    return whole(prompt, tokens);
}

fn whole(prompt: []const u32, tokens: []const u32) bool {
    if (tokens.len > prompt.len) return false;
    full += 1;
    return if (gated()) same(prompt[0..tokens.len], tokens) else std.mem.eql(u32, prompt[0..tokens.len], tokens);
}

/// Ids `same` folds into one vector test (std.mem.eql tests every 32 bytes: 3.6 ms at 600k ids on the laptop, 41 us).
const fold = 64;

/// `std.mem.eql(u32, a, b)` for lists of one length, `fold` ids a test: xor and or in vectors, then one reduction.
fn same(a: []const u32, b: []const u32) bool {
    std.debug.assert(a.len == b.len);
    const V = @Vector(8, u32);
    var i: usize = 0;
    while (i + fold <= a.len) : (i += fold) {
        var x: V = @splat(0);
        inline for (0..fold / 8) |k| x |= @as(V, a[i + k * 8 ..][0..8].*) ^ @as(V, b[i + k * 8 ..][0..8].*);
        if (@reduce(.Or, x) != 0) return false;
    }
    for (a[i..], b[i..]) |p, q| if (p != q) return false;
    return true;
}

/// How many leading ids `a` and `b` share (`std.mem.indexOfDiff`, else the shorter length); gated, a block at a time.
pub fn common(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    full += 1;
    if (!gated()) return std.mem.indexOfDiff(u32, a, b) orelse n;
    var at: usize = 0;
    while (at < n) : (at += block) { // a vector compare a block, the scalar search only inside the one that differs
        const end = @min(n, at + block);
        if (!same(a[at..end], b[at..end])) return at + std.mem.indexOfDiff(u32, a[at..end], b[at..end]).?;
    }
    return n;
}

/// Whether `a` and `b` may share more than `n` ids (they must agree at `n`); always true with the knob off.
pub fn beyond(a: []const u32, b: []const u32, n: usize) bool {
    return !gated() or (n < a.len and n < b.len and a[n] == b[n]);
}

/// `starts(prompt, tokens)` asked at most once and, gated, only when the walk needs it (off: at once, as before).
pub const Lazy = struct {
    prompt: []const u32,
    tokens: []const u32,
    known: ?bool = null,

    pub fn of(prompt: []const u32, tokens: []const u32) Lazy {
        return .{ .prompt = prompt, .tokens = tokens, .known = if (gated()) null else whole(prompt, tokens) };
    }

    pub fn get(l: *Lazy) bool {
        if (l.known == null) l.known = starts(l.prompt, l.tokens);
        return l.known.?;
    }
};

test "kept walk: the knob is on for any value but empty and 0" {
    try std.testing.expect(!parse("") and !parse("0"));
    try std.testing.expect(parse("1") and parse("on") and parse("00"));
}
