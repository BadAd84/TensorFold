//! TF_GLM_OVN_KDA_CHUNKED: a prompt chunk's KDA chain in chunked (WY) form, production's kda_chunk.cu (kda_chunked.py).
//! A prompt row's bits depend on its 32-row sub-chunk's alignment, so states keep and resume on that grid (snapGrid).

const std = @import("std");
const cuda = @import("cuda");
const kda = @import("cuda_kda.zig");

/// Rows of a sub-chunk (kda_chunk.cu's BT): sub-chunks sit at absolute positions, multiples of it.
pub const sub_rows = 32;
/// The kernel's block (kda_chunk.cu's NT) and its dynamic shared memory, sizeof(Smem) (asserted in the .cu).
pub const threads = 256;
pub const smem_bytes = 95488;

/// TF_GLM_OVN_KDA_CHUNKED: unset or 0 keeps the serial prompt chain; any other value runs prompt chunks chunked.
pub fn enabled() bool {
    return cuda.ovn.on("KDA_CHUNKED");
}

/// The prompt cache's grid while the knob is on (cuda_native.snapGridNow): states keep and resume on sub-chunk starts.
pub fn snapGrid() ?u32 {
    return if (enabled()) sub_rows else null;
}

/// glm_kda_chunk.cu's module and entry, loaded only while the knob is on.
pub const Kernel = struct {
    mod: cuda.Module,
    f: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernel {
        var mod = try cuda.Module.load(d, cuda.kernels.glm_kda_chunk);
        errdefer mod.unload();
        const f = try mod.function("tf_glm_kda_chunk");
        try f.allowDynamicShared(smem_bytes);
        std.log.info("glm: TF_GLM_OVN_KDA_CHUNKED: prompt chunks' KDA chain in 32-row sub-chunks (WY), states kept on 32 rows", .{});
        return .{ .mod = mod, .f = f };
    }

    pub fn unload(k: *Kernel) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// A prompt chunk's chain on the chunked kernel: `pos` is the absolute position of the chunk's first row.
pub const Prompt = struct { kernel: *const Kernel, pos: usize };

/// The chunked chain of a prompt chunk from `pos` when the knob loaded the kernel (Modules.chunked), else null.
pub fn prompt(k: *const ?Kernel, pos: usize) ?Prompt {
    return if (k.*) |*x| .{ .kernel = x, .pos = pos } else null;
}

/// kda_chunked.chain: `rows` rows of `c` from c.state_in into state_out, read-outs into out (bf16 [rows][heads * 128]).
/// The conv window is read, not shifted (the caller's convShift), and replay's saved rows are not written.
pub fn chain(p: Prompt, stream: cuda.Stream, c: kda.Chain, rows: usize, out: u64, state_out: u64) !void {
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(c.heads)));
    a.add(c.p);
    a.add(@as(c_int, @intCast(c.p_stride)));
    a.add(@as(c_int, @intCast(c.b_off)));
    a.add(c.a);
    a.add(@as(c_int, @intCast(c.a_stride)));
    a.add(c.g);
    a.add(@as(c_int, @intCast(c.g_stride)));
    for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias, c.norm_w }) |v| a.add(v);
    a.add(c.eps);
    a.add(c.lower);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(p.pos % sub_rows)));
    a.add(out);
    a.add(state_out);
    const cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(c.heads) }, .block = .{ .x = threads }, .shared = smem_bytes };
    try cuda.launch.launch(p.kernel.f, cfg, stream, &a);
}

test "the chunked chain's shared memory fits a GB10 block's opt-in 99 KiB, its sub-chunks a 32-row grid" {
    try std.testing.expect(smem_bytes <= 99 * 1024);
    try std.testing.expectEqual(@as(u32, 32), sub_rows);
}
