// Own kernels where the Python family used torch ops (integer work, copies, rounding), with the same bits.

#include <cuda_bf16.h>
#include <stdint.h>

constexpr long long PAGE = 512;  // rows a KV page holds

// sparse.select_tokens' assembly, a block a row: K pools' 4 tokens, then the tail; on pages each token's row in them.
template <class Pool>
__device__ __forceinline__ void dsa_tokens(const Pool* __restrict__ pools, const int* __restrict__ pos, int K,
                                           int* __restrict__ tokens, int* __restrict__ counts,
                                           const int* __restrict__ pt, long long page_rows) {
    const int r = blockIdx.x;
    const long long q = static_cast<long long>(*pos) + r;
    const long long npool = (q + 1) / 4;
    const int width = 4 * K + 3;
    int* row = tokens + static_cast<long long>(r) * width;
    auto at = [&](long long t) { return static_cast<int>(pt ? pt[t / PAGE] * page_rows + t % PAGE : t); };
    for (int i = threadIdx.x; i < 4 * K; i += blockDim.x)
        row[i] = at(static_cast<long long>(pools[static_cast<long long>(r) * K + i / 4]) * 4 + i % 4);
    if (threadIdx.x < 3) {
        const long long t = npool * 4 + threadIdx.x;
        row[4 * K + threadIdx.x] = t <= q ? at(t) : -1;
    }
    if (threadIdx.x == 0) {
        int tail = 0;
        for (int t = 0; t < 3; ++t) tail += npool * 4 + t <= q;
        counts[r] = npool > K ? 4 * K + tail : 0;
    }
}

extern "C" __global__ void tf_glm_dsa_tokens(const long long* __restrict__ pools, const int* __restrict__ pos, int K,
                                             int* __restrict__ tokens, int* __restrict__ counts,
                                             const int* __restrict__ pt, long long page_rows) {
    dsa_tokens(pools, pos, K, tokens, counts, pt, page_rows);
}

// tf_glm_dsa_tokens from int32 pools (TF_GLM_OVN_POOLS32: the prompt indexer split's gathered picks), the same bytes.
extern "C" __global__ void tf_glm_dsa_tokens_i32(const int* __restrict__ pools, const int* __restrict__ pos, int K,
                                                 int* __restrict__ tokens, int* __restrict__ counts,
                                                 const int* __restrict__ pt, long long page_rows) {
    dsa_tokens(pools, pos, K, tokens, counts, pt, page_rows);
}

#include "glm_ops_seg.cuh"

// TF_GLM_OVN_POOLS32: n int64 pool ids (each below 2^31) as int32, for the prompt indexer split's gather.
extern "C" __global__ void tf_glm_pools_i32(const long long* __restrict__ src, int* __restrict__ dst, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int>(src[i]);
}

// pos_dev + a (sparse.select_tokens' at = pos_dev + a for a row block past the first): one int32 on the device.
extern "C" __global__ void tf_glm_add_i32(const int* __restrict__ x, int a, int* __restrict__ out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *x + a;
}

// forward.shared_expert's b.ey[:, slot].copy_(b.sy): fp32 rows into slot `slot`, as is or bf16 round-to-nearest-even.
extern "C" __global__ void tf_glm_put_slot(const float* __restrict__ src, void* __restrict__ dst, int rows, int d,
                                           int slots, int slot, int bf16) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(rows) * d) return;
    const long long r = i / d, c = i % d;
    const long long at = (r * slots + slot) * d + c;
    if (bf16) reinterpret_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16_rn(src[i]);
    else reinterpret_cast<float*>(dst)[at] = src[i];
}

// A strided destination's copy_ (Fp4Linear's out= column slice): `rows` rows of row_bytes between pitches, as is.
extern "C" __global__ void tf_glm_copy_rows(const unsigned char* __restrict__ src, long long src_pitch,
                                            unsigned char* __restrict__ dst, long long dst_pitch, int rows,
                                            int row_bytes) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(rows) * row_bytes) return;
    const long long r = i / row_bytes, c = i % row_bytes;
    dst[r * dst_pitch + c] = src[r * src_pitch + c];
}
