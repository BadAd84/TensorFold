// TF_GLM_OVN_LATENT_MMA_EXACT: a byte-exact CUDA-core alternative to TF_GLM_OVN_LATENT_MMA's prompt absorb/expand.
#pragma once

#include "glm_glue_latent_ovn.cuh"

namespace glm_latent_exact {

using glm_latent_ovn::bf8;

// tf_glm_absorb_q bit for bit with ABSORB_VEC's 16-byte loads; ORDER true is base's fma seed, TREE true base's fold.
template <bool ORDER, bool TREE>
__device__ void absorb(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                       int64_t rb) {
  constexpr int D = 256, LW = 512;
  __shared__ float part[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = (int64_t)blockIdx.z * rb;
  float w[8][8];
#pragma unroll
  for (int rep = 0; rep < 8; ++rep) bf8(w[rep], wk + (h * D + k0 + 32 * rep) * LW + n0);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[8];
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) qv[rep] = bf(q[(r * heads + h) * D + k0 + 32 * rep]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float a;
      if constexpr (ORDER) {  // base: seed element 1 (fmul) then element 0, then 2..7, as Triton orders the products
        a = fmul(w[1][j], qv[1]);
        a = __fmaf_rn(w[0][j], qv[0], a);
#pragma unroll
        for (int rep = 2; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      } else {  // off: seed element 0 then 1..7; the eight products are the same, which two add first changes the bits
        a = fmul(w[0][j], qv[0]);
#pragma unroll
        for (int rep = 1; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      }
      if constexpr (TREE) {  // base: fold the high lane bits first (the base kernel's xor 16, 8, 4)
        for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      } else {  // off: fold the low bits first; the same partials summed in a differently grouped butterfly
        for (int o = 4; o <= 16; o <<= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      }
      if ((t & 28) == 0) part[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = part[t];
    if constexpr (TREE) {  // base: cross-warp fold xor 2 then 1
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    } else {
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    }
    if ((t & 3) == 0) qa[(r * heads + h) * LW + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

// tf_glm_expand_v bit for bit with ABSORB_VEC's 16-byte loads; ORDER and TREE as absorb (one-step combine fixed).
template <bool ORDER, bool TREE>
__device__ void expand(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                       int64_t heads, int64_t rb) {
  constexpr int DV = 256, LW = 512;
  __shared__ float part[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 16, r0 = (int64_t)blockIdx.z * rb;
  float w[8][8];
#pragma unroll
  for (int rep = 0; rep < 8; ++rep) bf8(w[rep], wv + (h * DV + n0 + nb + 2 * rep) * LW + kb);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float o[8];
    bf8(o, ol + (r * heads + h) * LW + kb);
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) {
      float a;
      if constexpr (ORDER) {  // base: seed element 1 (fmul) then 0, then 2..7
        a = fmul(w[rep][1], o[1]);
        a = __fmaf_rn(w[rep][0], o[0], a);
#pragma unroll
        for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      } else {  // off: seed element 0 then 1..7
        a = fmul(w[rep][0], o[0]);
#pragma unroll
        for (int j = 1; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      }
      if constexpr (TREE) {  // base: the 32-lane butterfly xor 16, 8, 4, 2, 1
        for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      } else {  // off: xor 1, 2, 4, 8, 16
        for (int s = 1; s <= 16; s <<= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      }
      if (lane == 0) part[(nb + 2 * rep) * 2 + (warp & 1)] = a;
    }
    __syncthreads();
    if (t < 32) {
      float v = part[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 1) == 0) out[(r * heads + h) * DV + n0 + t / 2] = __float2bfloat16_rn(v);
    }
    __syncthreads();
  }
}

}  // namespace glm_latent_exact

// All switches base: prompt absorb on the CUDA cores, byte for byte tf_glm_absorb_q (rowBlocks launches it unchanged).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_exact(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                        int64_t rb) {
  glm_latent_exact::absorb<true, true>(q, wk, qa, rows, heads, rb);
}

// ORDER off (the within-thread fma sequence is natural, not Triton's seed order): byte-changing, for timing alone.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_exact_order(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                              int64_t heads, int64_t rb) {
  glm_latent_exact::absorb<false, true>(q, wk, qa, rows, heads, rb);
}

// TREE off (the butterfly folds the low lane bits first): byte-changing, for timing alone.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_exact_tree(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                             int64_t heads, int64_t rb) {
  glm_latent_exact::absorb<true, false>(q, wk, qa, rows, heads, rb);
}

// All switches base: prompt expand on the CUDA cores, byte for byte tf_glm_expand_v.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_exact(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                        int64_t heads, int64_t rb) {
  glm_latent_exact::expand<true, true>(ol, wv, out, rows, heads, rb);
}

// ORDER off: byte-changing, for timing alone.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_exact_order(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                              int64_t heads, int64_t rb) {
  glm_latent_exact::expand<false, true>(ol, wv, out, rows, heads, rb);
}

// TREE off: byte-changing, for timing alone.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_exact_tree(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                             int64_t heads, int64_t rb) {
  glm_latent_exact::expand<true, false>(ol, wv, out, rows, heads, rb);
}
