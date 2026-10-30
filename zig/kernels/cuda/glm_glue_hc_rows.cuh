// TF_GLM_OVN_HC_ROWS and TF_GLM_OVN_HC_DEC: hc_partial's dots and squares on other layouts, every row's bits as today.
#pragma once

#include "glm_glue_hc.cuh"

namespace glm_hc_rows {

// K block b of R rows by 128 threads in tf_glm_hc_partial's lanes and chains, a step's weights read once for R rows.
template <int R>
__device__ __forceinline__ void dots(const __nv_bfloat16* x, const __nv_bfloat16* fn, int64_t wide, int64_t kb,
                                     int b, int64_t r0, int64_t rows, int t, float* acc, int stride,
                                     float (*sq)[4]) {
  const int lane = t & 31, warp = t >> 5, c = t & 15, m0 = (t >> 4) & 7;
  float ss[R];  // acc(q, m) at acc[q * stride + m], owned by lane c == 0; sq[q][warp]: the warps' sums of squares
#pragma unroll
  for (int q = 0; q < R; ++q) {
    ss[q] = 0.0f;
    if (c == 0)
      for (int i = 0; i < 3; ++i) acc[q * stride + m0 + 8 * i] = 0.0f;
  }
  for (int64_t step = 0; step < kb / 128; ++step) {
    const int64_t base = b * kb + step * 128;
    float w[3][8];
#pragma unroll
    for (int i = 0; i < 3; ++i) {  // the step's weights, once for the R rows (tf_glm_hc_partial: once every 4)
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
      const __nv_bfloat16* hw = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int j = 0; j < 8; ++j) w[i][j] = bf(hw[j]);
    }
#pragma unroll
    for (int q0 = 0; q0 < R; q0 += 4) {  // 4 rows' loads in flight at once, as tf_glm_hc_partial's
      float xv[4][8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {  // plain loads: the row kernel wrote x in place before its barrier
        uint4 u = make_uint4(0, 0, 0, 0);
        if (r0 + q0 + q < rows) u = *reinterpret_cast<const uint4*>(x + (r0 + q0 + q) * wide + base + c * 8);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
        for (int j = 0; j < 8; ++j) xv[q][j] = bf(h[j]);
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float d = __fmul_rn(xv[q][1], w[i][1]);
          d = __fmaf_rn(xv[q][0], w[i][0], d);
#pragma unroll
          for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[q][j], w[i][j], d);
#pragma unroll
          for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
          float* a = acc + (q0 + q) * stride + m0 + 8 * i;
          if (c == 0) *a = *a + d;
        }
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float xt = r0 + q0 + q < rows ? bf(x[(r0 + q0 + q) * wide + base + t]) : 0.0f;
        ss[q0 + q] = __fmaf_rn(xt, xt, ss[q0 + q]);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < R; ++q) {
    for (int o = 16; o > 0; o >>= 1) ss[q] += __shfl_xor_sync(0xffffffffu, ss[q], o);
    if (lane == 0) sq[q][warp] = ss[q];
  }
}

// tf_glm_hc_partial16 / 32: block (R rows, K block b) of 128 threads, PART as tf_glm_hc_partial writes it.
template <int R>
__device__ __forceinline__ void partial(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide,
                                        int nb, int64_t rows) {
  __shared__ float acc[R][24];
  __shared__ float sq[R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R;
  const int b = blockIdx.y, t = threadIdx.x;
  dots<R>(x, fn, wide, wide / nb, b, r0, rows, t, &acc[0][0], 24, sq);
  __syncthreads();
  for (int u = t; u < R * 25; u += 128) {
    const int q = u / 25, m = u % 25;
    if (r0 + q >= rows) break;  // u rises with q: every later u is a later row
    part[((r0 + q) * nb + b) * 32 + m] = m < 24 ? acc[q][m] : (sq[q][0] + sq[q][2]) + (sq[q][1] + sq[q][3]);
  }
}

// GLUE_ROW with HC_ROWS: glm_hc::rows4 on R rows a block, dots<R> its partials; PART [R][16][32] dynamic shared.
template <typename W, int R>
__device__ __forceinline__ void rows(__nv_bfloat16* x, const float* g, int64_t rs, int world, const __nv_bfloat16* fn,
                                     const float* base, const float* scale, const W* nw, __nv_bfloat16* out, float* xs,
                                     float* post, float* comb, float eps, float hc_eps, int iters, int64_t rows) {
  constexpr int D = 4096, S = 4, NB = 16, KB = S * D / NB, QUARTERS = D / KB, CHUNKS = KB / 8;
  constexpr int64_t wide = S * D;
  extern __shared__ __align__(16) unsigned char smem[];
  float* part = reinterpret_cast<float*>(smem);
  __shared__ float sq[2][R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R;
  const int tid = threadIdx.x, grp = tid >> 7, t = tid & 127;
  for (int qq = 0; qq < QUARTERS; ++qq) {
    for (int u = tid; u < R * CHUNKS; u += 256) {
      const int64_t row = r0 + u / CHUNKS;
      if (row < rows) glm_glue::hc_post8(x, x, g, post, comb, rs, D, world, row, qq * KB + u % CHUNKS * 8);
    }
    __syncthreads();
    for (int h = 0; h < 2; ++h) {
      const int b = (grp + 2 * h) * QUARTERS + qq;
      dots<R>(x, fn, wide, KB, b, r0, rows, t, part + b * 32, NB * 32, sq[grp]);
      __syncthreads();
      if (t < R) part[(t * NB + b) * 32 + 24] = (sq[grp][t][0] + sq[grp][t][2]) + (sq[grp][t][1] + sq[grp][t][3]);
      __syncthreads();
    }
  }
  for (int q = 0; q < R && r0 + q < rows; ++q)
    glm_hc::finish<W>(x + r0 * wide, part, base, scale, nw, out + r0 * D, xs + r0 * (D / 64), post + r0 * S,
                      comb + r0 * 16, eps, hc_eps, NB, iters, q);
}

// TF_GLM_OVN_HC_DEC: hc.cu's block (K block, part p, 4 rows), warp w < 8 mixing row 8 p + w, warp 8 the squares.
template <bool PDL>
__device__ __forceinline__ void partial48(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t rows) {
  constexpr int WIDE = 16384, NB = 16, KB = WIDE / NB, SUB = 128, STEPS = KB / SUB, RPB = 4;
  const int b = blockIdx.x, p = blockIdx.y;
  const int64_t r0 = (int64_t)blockIdx.z * RPB, r1 = r0 + RPB < rows ? r0 + RPB : rows;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = lane >> 4, l = lane & 15, m = p * 8 + warp;
  uint4 wu[4];  // half h = lane / 16: steps 2 i + h; lane l = lane % 16: their inputs 8 l .. 8 l + 7
  if (warp < 8) {  // with PDL, launched dependent: the weights load before griddepcontrol.wait, x after it
    const __nv_bfloat16* wr = fn + (int64_t)m * WIDE + b * KB;
    for (int i = 0; i < 4; ++i) wu[i] = __ldg(reinterpret_cast<const uint4*>(wr + (2 * i + h) * SUB + 8 * l));
  }
  if constexpr (PDL) qmm_frag::grid_wait();
  if (warp < 8) {
    float wf[4][8];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j) wf[i][j] = bf(reinterpret_cast<const __nv_bfloat16*>(&wu[i])[j]);
    for (int64_t r = r0; r < r1; ++r) {
      const __nv_bfloat16* xr = x + r * WIDE + b * KB;
      uint4 xu[4];
      for (int i = 0; i < 4; ++i) xu[i] = *reinterpret_cast<const uint4*>(xr + (2 * i + h) * SUB + 8 * l);
      float s[4];
      for (int i = 0; i < 4; ++i) {
        const __nv_bfloat16* hx = reinterpret_cast<const __nv_bfloat16*>(&xu[i]);
        float d = __fmul_rn(bf(hx[1]), wf[i][1]);
        d = __fmaf_rn(bf(hx[0]), wf[i][0], d);
        for (int j = 2; j < 8; ++j) d = __fmaf_rn(bf(hx[j]), wf[i][j], d);
        for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
        s[i] = d;
      }
      float acc = 0.0f;  // partial8's chains and order: steps added 0, 1, .., 7
      for (int i = 0; i < 4; ++i) {
        acc = acc + s[i];                                  // step 2 i (half 0)
        acc = acc + __shfl_sync(0xffffffffu, s[i], 16);  // step 2 i + 1 (half 1)
      }
      if (lane == 0) part[(r * NB + b) * 32 + m] = acc;
    }
  } else if (p == 0) {
    for (int64_t r = r0; r < r1; ++r) {
      const __nv_bfloat16* xr = x + r * WIDE + b * KB;
      float w[4];
      for (int j = 0; j < 4; ++j) {  // partial8's warp j: input 32 j + lane of each step
        float ss = 0.0f;
        for (int st = 0; st < STEPS; ++st) {
          const float xt = bf(xr[st * SUB + 32 * j + lane]);
          ss = __fmaf_rn(xt, xt, ss);
        }
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        w[j] = ss;
      }
      if (lane == 0) part[(r * NB + b) * 32 + 24] = (w[0] + w[2]) + (w[1] + w[3]);
    }
  }
}

}  // namespace glm_hc_rows

// TF_GLM_OVN_HC_ROWS=16 / 32: prompt rows' partials, R rows a 128-thread block.
#define TF_GLM_HC_PARTIAL_ROWS(name, R)                                                                            \
  extern "C" __global__ void __launch_bounds__(128, 4)                                                             \
      name(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {    \
    glm_hc_rows::partial<R>(x, fn, part, wide, nb, rows);                                                          \
  }
TF_GLM_HC_PARTIAL_ROWS(tf_glm_hc_partial16, 16)
TF_GLM_HC_PARTIAL_ROWS(tf_glm_hc_partial32, 32)
#undef TF_GLM_HC_PARTIAL_ROWS

// TF_GLM_OVN_HC_ROWS with GLUE_ROW: tf_glm_hc_row's arguments, R rows a block; bf16 or float32 norm weight.
#define TF_GLM_HC_ROW_ROWS(name, W, R)                                                                             \
  extern "C" __global__ void __launch_bounds__(256)                                                                \
      name(__nv_bfloat16* x, const float* g, int64_t rs, int world, const __nv_bfloat16* fn, const float* base,   \
           const float* scale, const W* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,    \
           float hc_eps, int iters, int64_t rows) {                                                                \
    glm_hc_rows::rows<W, R>(x, g, rs, world, fn, base, scale, nw, out, xs, post, comb, eps, hc_eps, iters, rows);  \
  }
TF_GLM_HC_ROW_ROWS(tf_glm_hc_row16, __nv_bfloat16, 16)
TF_GLM_HC_ROW_ROWS(tf_glm_hc_row16_f32w, float, 16)
TF_GLM_HC_ROW_ROWS(tf_glm_hc_row32, __nv_bfloat16, 32)
TF_GLM_HC_ROW_ROWS(tf_glm_hc_row32_f32w, float, 32)
#undef TF_GLM_HC_ROW_ROWS

// TF_GLM_OVN_HC_DEC: decode windows' partials on hc.cu's layout; _pdl launched dependent (TF_GLM_OVN_PDL_EARLY).
extern "C" __global__ void __launch_bounds__(288)
    tf_glm_hc_partial48(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t rows) {
  glm_hc_rows::partial48<false>(x, fn, part, rows);
}
extern "C" __global__ void __launch_bounds__(288)
    tf_glm_hc_partial48_pdl(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t rows) {
  glm_hc_rows::partial48<true>(x, fn, part, rows);
}
