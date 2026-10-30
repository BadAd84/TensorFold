//! Where a template's history ends: a final `{% if add_generation_prompt %}` block that nothing else reads.
const std = @import("std");
const ast = @import("ast.zig");

const flag = "add_generation_prompt";

/// The index in `body` of a final generation block, when rendering with the flag off stops right before it.
pub fn split(body: []const ast.Stmt) ?usize {
    var last = body.len;
    while (last > 0 and emptyOutput(body[last - 1])) last -= 1;
    if (last == 0) return null;
    const n = switch (body[last - 1]) {
        .if_ => |n| n,
        else => return null,
    };
    if (n.elifs.len > 0 or n.else_body.len > 0) return null;
    if (n.test_expr.* != .name or !std.mem.eql(u8, n.test_expr.name.id, flag)) return null;
    // the test is the flag's only mention: everything before it renders alike with the flag on or off
    var seen: usize = 0;
    stmts(body, &seen);
    return if (seen == 1) last - 1 else null;
}

fn emptyOutput(st: ast.Stmt) bool {
    if (st != .output) return false;
    for (st.output) |pc| if (pc != .text or pc.text.len > 0) return false;
    return true;
}

fn name(n: *const ast.Name, seen: *usize) void {
    if (std.mem.eql(u8, n.id, flag)) seen.* += 1;
}

fn target(t: ast.Target, seen: *usize) void {
    switch (t) {
        .name => |n| name(n, seen),
        .tuple => |items| for (items) |x| target(x, seen),
        .ns => |ns| name(ns.name, seen),
    }
}

fn expr(e: *const ast.Expr, seen: *usize) void {
    if (e.* == .name) name(e.name, seen);
    ast.eachChild(error{}, e, seen, child) catch |err| switch (err) {};
}

fn child(seen: *usize, e: *ast.Expr) error{}!void {
    expr(e, seen);
}

fn stmts(body: []const ast.Stmt, seen: *usize) void {
    for (body) |st| switch (st) {
        .output => |pieces| for (pieces) |pc| if (pc == .expr) expr(pc.expr, seen),
        .for_ => |f| {
            target(f.target, seen);
            expr(f.iter, seen);
            if (f.filter) |x| expr(x, seen);
            stmts(f.body, seen);
            stmts(f.else_body, seen);
        },
        .if_ => |n| ifs(n, seen),
        .assign => |a| {
            target(a.target, seen);
            expr(a.value, seen);
        },
        .block_set => |b| {
            target(b.target, seen);
            if (b.filter) |x| expr(x, seen);
            stmts(b.body, seen);
        },
        .macro => |m| {
            name(m.name, seen);
            for (m.params) |p| name(p, seen);
            for (m.defaults) |d| expr(d, seen);
            stmts(m.body, seen);
        },
        .scoped => |s| stmts(s.body, seen),
        .break_, .continue_ => {},
    };
}

fn ifs(n: *const ast.If, seen: *usize) void {
    expr(n.test_expr, seen);
    stmts(n.body, seen);
    for (n.elifs) |e| ifs(e, seen);
    stmts(n.else_body, seen);
}
