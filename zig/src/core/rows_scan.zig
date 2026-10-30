//! A store opened: the files under its directory read back, states checked by name and blocks by size; leftovers go.
const std = @import("std");
const platform = @import("sys");
const modes = @import("cache_modes.zig");
const chain = @import("rows_chain.zig");
const file = @import("lanes").rows_file;
const dirs = @import("lanes").learned_dirs;
const write = @import("rows_write.zig");
const rows_store = @import("rows_store.zig");
const Rows = rows_store.Rows;
const State = rows_store.State;
const Span = modes.Span;

/// The files under the store's directory: states read back and checked by name, blocks by size; leftovers go.
pub fn scan(r: *Rows) !void {
    const d = platform.c.opendir(r.dir) orelse return error.RowsDir;
    defer _ = platform.c.closedir(d);
    while (platform.c.readdir(d)) |ent| {
        if (ent.type != platform.c.DT.REG) continue;
        const name = std.mem.sliceTo(&ent.name, 0);
        var buf: [1200]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ r.dir, name }, 0) catch continue;
        if (std.mem.endsWith(u8, name, ".part")) {
            _ = platform.c.unlink(p);
            continue;
        }
        const n = parseName(name) orelse continue;
        if (std.mem.endsWith(u8, name, ".blk")) {
            const bytes = dirs.fileBytes(p) catch continue;
            try r.blocks.put(r.gpa, n, .{ .bytes = bytes });
        } else if (std.mem.endsWith(u8, name, ".st")) {
            readState(r, p) catch {
                _ = platform.c.unlink(p);
            };
        }
    }
}

/// A state file's metadata checked against its name, kept for lookups (its payload is checked when read).
fn readState(r: *Rows, p: [:0]const u8) !void {
    var o = try file.open(p, .state, null);
    defer o.close();
    const m = try file.meta(&o, r.gpa);
    defer r.gpa.free(m);
    if (m.len < 12) return error.RowsRead;
    const counts = std.mem.bytesAsSlice(u32, m[0..12]);
    const nt: usize = counts[0];
    const ns: usize = counts[1];
    const nst: usize = counts[2];
    if (m.len != 12 + 4 * (nt + 2 * ns + nst) or nt == 0) return error.RowsRead;
    var s: State = .{ .name = o.trailer.name, .at = o.trailer.at, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer), .rows = 0, .used = 0, .shared = o.trailer.flags & 1 != 0 };
    errdefer Rows.freeState(r.gpa, &s);
    s.tokens = try r.gpa.dupe(u32, @alignCast(std.mem.bytesAsSlice(u32, m[12..][0 .. 4 * nt])));
    s.spans = try r.gpa.dupe(Span, @alignCast(std.mem.bytesAsSlice(Span, m[12 + 4 * nt ..][0 .. 8 * ns])));
    s.starts = try r.gpa.dupe(u32, @alignCast(std.mem.bytesAsSlice(u32, m[12 + 4 * nt + 8 * ns ..][0 .. 4 * nst])));
    if (s.at == 0 or s.tokens.len != s.at + r.keying.lookahead or o.trailer.block != r.keying.block) return error.RowsRead;
    if (!std.mem.eql(u8, &chain.state(r.keying, s.tokens, s.spans, s.starts, s.at), &s.name)) return error.RowsRead;
    modes.validate(s.spans) catch return error.RowsRead;
    r.clock += 1;
    s.used = r.clock;
    try r.states.append(r.gpa, s);
}

/// Each state's blocks counted; a state missing one, or that its family cannot lay out, goes; then idle blocks.
pub fn settle(r: *Rows) void {
    var i: usize = 0;
    while (i < r.states.items.len) {
        const s = &r.states.items[i];
        var wants: std.ArrayList(write.Want) = .empty;
        defer wants.deinit(r.gpa);
        const whole = if (r.wantsOf(r.gpa, s, &wants)) |_| for (wants.items) |w| {
            if (!r.blocks.contains(chain.stem(w.name))) break false;
        } else true else |_| false;
        if (!whole) {
            var gone = r.states.orderedRemove(i);
            rows_store.unlinkName(r.dir, gone.name, ".st");
            Rows.freeState(r.gpa, &gone);
            continue;
        }
        r.ref(wants.items, 1);
        s.rows = 0;
        for (wants.items) |w| s.rows += w.bytes;
        i += 1;
    }
    r.collect();
}

/// A file's stem: the first 16 bytes of its name in hex (32 digits), then `.blk` or `.st`.
fn parseName(name: []const u8) ?chain.Stem {
    if (name.len < 33 or name[32] != '.') return null;
    var n: chain.Stem = undefined;
    _ = std.fmt.hexToBytes(&n, name[0..32]) catch return null;
    return n;
}
