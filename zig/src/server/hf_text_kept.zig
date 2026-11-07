//! HfText's kept-piece encoders for the Text vtable: a request's text kept with byte provenance, then its pieces or
//! prefix pieces against that kept state, every tokenizer failure read as the request's problem text.
const std = @import("std");
const tokenizer = @import("tokenizer");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;
const HfText = @import("hf_text.zig").HfText;

fn self(ctx: *anyopaque) *HfText {
    return @ptrCast(@alignCast(ctx));
}

pub fn encodeMaskedFn(ctx: *anyopaque, a: Allocator, s: []const u8, spans: []const model_text.Span, problem: *[]const u8) model_text.Error![]u32 {
    return self(ctx).tok.encodeMasked(a, s, spans) catch |err| return masked(err, problem);
}

pub fn keepFn(ctx: *anyopaque, a: Allocator, s: []const u8, spans: []const model_text.Span, problem: *[]const u8) model_text.Error!model_text.Kept {
    const k = try a.create(tokenizer.kept.Kept);
    k.* = tokenizer.kept.keep(&self(ctx).tok, a, s, spans) catch |err| return masked(err, problem);
    return .{ .ids = k.ids, .state = k };
}

pub fn piecesFn(ctx: *anyopaque, a: Allocator, kept: model_text.Kept, s: []const u8, spans: []const model_text.Span, problem: *[]const u8) model_text.Error!model_text.Pieces {
    const k: *const tokenizer.kept.Kept = @ptrCast(@alignCast(kept.state.?));
    const p = tokenizer.kept.pieces(&self(ctx).tok, a, k, s, spans) catch |err| return masked(err, problem);
    return .{ .head = p.head, .mid = p.mid, .tail = p.tail };
}

pub fn prefixFn(ctx: *anyopaque, a: Allocator, kept: model_text.Kept, s: []const u8, spans: []const model_text.Span, problem: *[]const u8) model_text.Error!?model_text.Pieces {
    const k: *const tokenizer.kept.Kept = @ptrCast(@alignCast(kept.state.?));
    const p = (tokenizer.kept.prefix(&self(ctx).tok, a, k, s, spans) catch |err| return masked(err, problem)) orelse return null;
    return .{ .head = p.head, .mid = p.mid };
}

/// A masked encoding's failure as the request reads it.
fn masked(err: tokenizer.Error, problem: *[]const u8) model_text.Error {
    if (err == error.OutOfMemory) return error.OutOfMemory;
    problem.* = if (err == error.UnsafeNormalizedAddedTokenMask)
        "normalized added tokens cannot be matched safely through this normalizer with request text"
    else if (err == error.UnsafeAddedTokenMask)
        "request text tokenizes to an added-token id in this tokenizer; the prompt cannot be encoded safely"
    else
        "the tokenizer cannot encode this prompt with request byte provenance";
    return error.Template;
}
