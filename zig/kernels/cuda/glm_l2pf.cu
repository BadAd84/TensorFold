// L2 prefetch of weight ranges for GLM-5.3-Flash's decode windows (glm_cuda's port of the glm5_next family's
// l2pf.cu, v1.8): a small kernel on a side stream asks the memory system to bring the next kernels' weights
// into L2 while the main stream waits on an all-gather or runs latency-bound glue. It only loads or prefetches
// (no store, atomic or reduction), so no result can change.
//
// TABLE [n, 2] i64: (address, bytes) pieces, each 16-byte aligned and at most 32 KiB (the host splits).
// MODE 0 ("bulk"): one cp.async.bulk.prefetch.L2 a piece (the TMA unit walks it; sm_90+).
// MODE 1 ("lines"): prefetch.global.L2::evict_last on every 128-byte line of a piece, a warp's lanes on
// adjacent lines. MODE 2 ("touch"): ld.global.cg of every line's first 16 bytes (a real load: the kernel
// lasts as long as the reads).

#include <stdint.h>

extern "C" __global__ void __launch_bounds__(512) tf_glm_l2pf(
    const long long* __restrict__ table, int n, int mode, unsigned* __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    if (mode == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            const unsigned long long a = (unsigned long long)table[2 * i];
            const unsigned bytes = (unsigned)table[2 * i + 1];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(bytes) : "memory");
#else
            for (unsigned o = 0; o < bytes; o += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(a + o));
#endif
        }
        return;
    }
    unsigned acc = 0;
    for (int i = warp; i < n; i += warps) {             // a warp a piece, its lanes on adjacent lines
        const unsigned long long a = (unsigned long long)table[2 * i];
        const unsigned bytes = (unsigned)table[2 * i + 1];
        for (unsigned o = lane * 128u; o < bytes; o += 32u * 128u) {
            if (mode == 1) {
                asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a + o));
            } else {
                unsigned x, y, z, w;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a + o));
                acc ^= x ^ y ^ z ^ w;
            }
        }
    }
    if (mode == 2 && acc == 0x9e3779b9u && sink != nullptr) sink[0] = acc;  // keeps the loads; never true in practice
}
