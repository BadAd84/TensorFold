//! Every prompt-cache test module, referenced from prompt_cache.zig so each test target reaches them.
test {
    _ = @import("prompt_cache_index_test.zig");
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_learn_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("prompt_cache_interior_test.zig");
    _ = @import("prompt_cache_branch_test.zig");
    _ = @import("prompt_cache_fork_test.zig");
    _ = @import("prompt_cache_give_test.zig");
    _ = @import("prompt_cache_lifetime_test.zig");
    _ = @import("prompt_cache_scan_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = @import("prompt_prefix_test.zig");
    _ = @import("prompt_cache_repeat_test.zig");
    _ = @import("prompt_cache_tail_test.zig");
}
