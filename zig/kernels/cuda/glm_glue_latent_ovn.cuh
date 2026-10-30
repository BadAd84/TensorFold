// TF_GLM_OVN_ABSORB_VEC and TF_GLM_OVN_KVB8: glm_glue_latent.cuh's absorb and expand reading a thread's 8 weights of a
// row by one load: bf16 kv_b by one 16-byte load, or E4M3 kv_b (8 bytes) times its fp32 scale a 128-column block.
#pragma once

#include "glm_glue_latent.cuh"
#include <cuda_fp8.h>

namespace glm_latent_ovn {

constexpr int KB = 128;  // latent columns one E4M3 kv_b scale covers (the Python family's qmm.make_f8 block)

// A bf16 value from its 16 bits, through the same bf() the scalar kernels call.
__device__ __forceinline__ float bf_of(uint32_t bits) {
  __nv_bfloat16_raw raw;
  raw.x = (unsigned short)(bits & 0xffffu);
  return bf(__nv_bfloat16(raw));
}

// 8 bf16 at p (16-byte aligned) by one load, element j from the low half of word j / 2 first (little-endian).
__device__ __forceinline__ void bf8(float* out, const __nv_bfloat16* p) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    out[2 * k] = bf_of(words[k]);
    out[2 * k + 1] = bf_of(words[k] >> 16);
  }
}

// 8 E4M3 codes at p (8-byte aligned) by one load: each code's exact value times s (latent.py's _absorb_q8, _expand_v8).
__device__ __forceinline__ void e4m3x8(float* out, const unsigned char* p, float s) {
  const uint2 c = *reinterpret_cast<const uint2*>(p);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint16_t bits = (uint16_t)(((k < 2 ? c.x : c.y) >> (16 * (k & 1))) & 0xffffu);
    const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
    out[2 * k] = __fmul_rn(v.x, s);
    out[2 * k + 1] = __fmul_rn(v.y, s);
  }
}

// tf_glm_absorb_q's block and sums; F8: wk E4M3 [heads][256][512], sk fp32 [heads][256][512 / KB].
template <bool F8>
__device__ void absorb(const __nv_bfloat16* q, const void* wk, const float* sk, __nv_bfloat16* qa, int64_t rows,
                       int64_t heads, int64_t rb) {
  constexpr int D = 256, LW = 512;
  __shared__ float part[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = blockIdx.z * rb;
  float w[8][8];
#pragma unroll
  for (int rep = 0; rep < 8; ++rep) {
    const int64_t row = h * D + k0 + 32 * rep;
    if constexpr (F8)
      e4m3x8(w[rep], static_cast<const unsigned char*>(wk) + row * LW + n0, sk[row * (LW / KB) + n0 / KB]);
    else
      bf8(w[rep], static_cast<const __nv_bfloat16*>(wk) + row * LW + n0);
  }
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[8];
    for (int rep = 0; rep < 8; ++rep) qv[rep] = bf(q[(r * heads + h) * D + k0 + 32 * rep]);
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);
      a = __fmaf_rn(w[0][j], qv[0], a);
      for (int rep = 2; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      if ((t & 28) == 0) part[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = part[t];
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * LW + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

// tf_glm_expand_v's block and sums; F8: wv E4M3 [heads][256][512], sv fp32 [heads][256][512 / KB], o read as there.
template <bool F8>
__device__ void expand(const __nv_bfloat16* ol, const void* wv, const float* sv, __nv_bfloat16* out, int64_t rows,
                       int64_t heads, int64_t rb) {
  constexpr int DV = 256, LW = 512;
  __shared__ float part[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 16, r0 = blockIdx.z * rb;
  float w[8][8];
#pragma unroll
  for (int rep = 0; rep < 8; ++rep) {
    const int64_t row = h * DV + n0 + nb + 2 * rep;
    if constexpr (F8)
      e4m3x8(w[rep], static_cast<const unsigned char*>(wv) + row * LW + kb, sv[row * (LW / KB) + kb / KB]);
    else
      bf8(w[rep], static_cast<const __nv_bfloat16*>(wv) + row * LW + kb);
  }
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float o[8];
    if constexpr (F8) {
      for (int j = 0; j < 8; ++j) o[j] = bf(ol[(r * heads + h) * LW + kb + j]);
    } else {
      bf8(o, ol + (r * heads + h) * LW + kb);
    }
    for (int rep = 0; rep < 8; ++rep) {
      float a = fmul(w[rep][1], o[1]);
      a = __fmaf_rn(w[rep][0], o[0], a);
      for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
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

}  // namespace glm_latent_ovn

// TF_GLM_OVN_ABSORB_VEC: tf_glm_absorb_q / tf_glm_expand_v bit for bit, kv_b (and expand's o) by 16-byte loads.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q_vec(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                        int64_t heads, int64_t rb) {
  glm_latent_ovn::absorb<false>(q, wk, nullptr, qa, rows, heads, rb);
}
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v_vec(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                        int64_t heads, int64_t rb) {
  glm_latent_ovn::expand<false>(ol, wv, nullptr, out, rows, heads, rb);
}

// TF_GLM_OVN_KVB8 (a numerics trade): kv_b as E4M3 codes and fp32 scales, the same blocks and sums.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q8(const __nv_bfloat16* q, const unsigned char* wk, const float* sk, __nv_bfloat16* qa, int64_t rows,
                     int64_t heads, int64_t rb) {
  glm_latent_ovn::absorb<true>(q, wk, sk, qa, rows, heads, rb);
}
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v8(const __nv_bfloat16* ol, const unsigned char* wv, const float* sv, __nv_bfloat16* out,
                     int64_t rows, int64_t heads, int64_t rb) {
  glm_latent_ovn::expand<true>(ol, wv, sv, out, rows, heads, rb);
}
