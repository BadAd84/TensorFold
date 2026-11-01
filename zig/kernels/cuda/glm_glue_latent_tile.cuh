// TF_GLM_OVN_LATENT_TILE: tf_glm_absorb_q / tf_glm_expand_v for prompt rows, byte identical, q / o reused per block.
#pragma once

#include "glm_glue_latent.cuh"
#include "glm_glue_latent_ovn.cuh"

namespace glm_latent_tile {

constexpr int D = 256, LW = 512, DV = 256;  // head width (qk and v), latent width

// tf_glm_absorb_q over G strips (128 G threads): q[r,:256] staged once a row, each thread one strip's 8 columns.
template <int G, bool RED>
__device__ void absorb(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                       int64_t rb) {
  __shared__ float qsmem[D];       // q[r,:256] as the bf() floats the scalar kernel reads
  __shared__ float red[G * 128];   // each strip's 128 partials (one base block's smem)
  const int tid = threadIdx.x, g = tid >> 7, t = tid & 127;  // strip in the block, thread in the strip
  const int k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, strip = (int64_t)blockIdx.y * G + g, r0 = (int64_t)blockIdx.z * rb;
  const int64_t n0 = strip * 32 + nq * 8;
  float* rg = red + g * 128;
  float w[8][8];
#pragma unroll
  for (int rep = 0; rep < 8; ++rep) glm_latent_ovn::bf8(w[rep], wk + (h * D + k0 + 32 * rep) * LW + n0);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    if (tid < 32) glm_latent_ovn::bf8(qsmem + tid * 8, q + (r * heads + h) * D + tid * 8);  // stage q[r,:256]
    __syncthreads();
    float qv[8];
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) qv[rep] = qsmem[k0 + 32 * rep];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);  // tf_glm_absorb_q's chain: seed rep 1, then rep 0, then rep 2..7 (k = k0 + 32 rep)
      a = __fmaf_rn(w[0][j], qv[0], a);
#pragma unroll
      for (int rep = 2; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      // RED reverses the butterfly into a different, valid association: a wrong tree for glm-latent-tile-check only
      if constexpr (RED) {
        for (int o = 4; o <= 16; o <<= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      } else {
        for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);  // butterfly over the 8 k0 a warp
      }
      if ((t & 28) == 0) rg[(nq * 8 + j) * 4 + warp] = a;  // one slot a (column, warp)
    }
    __syncthreads();
    float v = rg[t];  // cross-warp tree: shfl.xor 2, 1 over the 4 warps, then round to nearest even bf16
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * LW + strip * 32 + t / 4] = __float2bfloat16_rn(v);
  }
}

// tf_glm_expand_v over G output blocks (128 G threads): o_lat[r,:512] staged once a row, each thread one block's 8.
template <int G, bool RED>
__device__ void expand(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                       int64_t heads, int64_t rb) {
  __shared__ float osmem[LW];     // o_lat[r,:512] as the bf() floats the scalar kernel reads
  __shared__ float red[G * 32];   // each output block's 32 partials (one base block's smem)
  const int tid = threadIdx.x, g = tid >> 7, t = tid & 127;
  const int lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, blk = (int64_t)blockIdx.y * G + g, r0 = (int64_t)blockIdx.z * rb;
  const int64_t n0 = blk * 16;
  float* rg = red + g * 32;
  float w[8][8];
#pragma unroll
  for (int rep = 0; rep < 8; ++rep) glm_latent_ovn::bf8(w[rep], wv + (h * DV + n0 + nb + 2 * rep) * LW + kb);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    if (tid < 64) glm_latent_ovn::bf8(osmem + tid * 8, ol + (r * heads + h) * LW + tid * 8);  // stage o_lat[r,:512]
    __syncthreads();
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = osmem[kb + j];
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) {
      float a = fmul(w[rep][1], o[1]);  // tf_glm_expand_v's chain: seed j 1, then j 0, then j 2..7 (o contiguous)
      a = __fmaf_rn(w[rep][0], o[0], a);
#pragma unroll
      for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      // RED reverses the butterfly into a different, valid association: a wrong tree for glm-latent-tile-check only
      if constexpr (RED) {
        for (int s = 1; s <= 16; s <<= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      } else {
        for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);  // full butterfly, a warp's half
      }
      if (lane == 0) rg[(nb + 2 * rep) * 2 + (warp & 1)] = a;
    }
    __syncthreads();
    if (t < 32) {
      float v = rg[t];  // cross-half: shfl.xor 1 joins the two 256-latent halves, then round to nearest even bf16
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 1) == 0) out[(r * heads + h) * DV + n0 + t / 2] = __float2bfloat16_rn(v);
    }
  }
}

constexpr int G = 4;  // strips a block: grid.y / 4 fewer q / o reads, flat registers, 512 threads (one block a SM)

}  // namespace glm_latent_tile

// TF_GLM_OVN_LATENT_TILE: tf_glm_absorb_q / tf_glm_expand_v bit for bit, q / o reused across four strips a block.
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_absorb_tile(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                       int64_t rb) {
  glm_latent_tile::absorb<glm_latent_tile::G, false>(q, wk, qa, rows, heads, rb);
}
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_expand_tile(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                       int64_t heads, int64_t rb) {
  glm_latent_tile::expand<glm_latent_tile::G, false>(ol, wv, out, rows, heads, rb);
}

// Controls for glm-latent-tile-check (never a serve path): a reversed butterfly, so the bytes MUST differ from base.
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_absorb_tile_red(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                           int64_t heads, int64_t rb) {
  glm_latent_tile::absorb<glm_latent_tile::G, true>(q, wk, qa, rows, heads, rb);
}
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_expand_tile_red(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                           int64_t heads, int64_t rb) {
  glm_latent_tile::expand<glm_latent_tile::G, true>(ol, wv, out, rows, heads, rb);
}
