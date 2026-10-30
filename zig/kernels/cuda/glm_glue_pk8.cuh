// TF_GLM_OVN_PK8 (TRADE, production's TF_GLM_KV=fp8 pooled keys): tf_glm_pool_keys and tf_glm_scores on kv8 rows.
// Writer: prod-src families/glm5_next/cuda/sparse.py _pool_keys + kv8.store_row; readers: sparse.py _scores (FP8),
// codes taken to bf16 exactly as prompt_scores.cu does, each pool's power-of-two scale on its dots before `scale`.
#pragma once

#include "glm_glue_index.cuh"
#include "glm_glue_attn.cuh"
#include "glm_seg_table.cuh"
#include "glm_pk8_row.cuh"

namespace glm_pk8 {

// pool_keys_body's pooled key (the same bf16 values), stored as its kv8 row: codes of value * inv, then the scale.
__device__ __forceinline__ void pool_keys_body(const __nv_bfloat16* ik, const __nv_bfloat16* ig,
                                               const __nv_bfloat16* ape, unsigned char* pk, const int* pos,
                                               int64_t rows, const int* pt, int64_t ps) {
  const int64_t P = pos[0], p = P / 4 + blockIdx.x;
  if (4 * p + 3 > P + rows - 1) return;
  unsigned char* out = pk + row_at(pt, ps, p);
  float x[4];
  for (int j = 0; j < 4; ++j) {
    const int d = threadIdx.x * 4 + j;
    float l[4], e[4];
    for (int q = 0; q < 4; ++q) l[q] = glm_triton::bf(ig[(4 * p + q) * D + d]) + glm_triton::bf(ape[q * D + d]);
    const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
    for (int q = 0; q < 4; ++q) e[q] = glm_triton::exp_approx(l[q] - m);
    const float s = ((e[0] + e[1]) + e[2]) + e[3];
    float t = 0.0f;
    for (int q = 0; q < 4; ++q) {
      const float tq = glm_triton::rbf(
          __fmul_rn(glm_triton::rbf(glm_triton::div_full(e[q], s)), glm_triton::bf(ik[(4 * p + q) * D + d])));
      t = q == 0 ? tq : t + tq;
    }
    x[j] = __bfloat162float(__float2bfloat16_rn(t));  // store_row: the bf16 row is what is quantized
  }
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  float sc, inv;
  scale_of(amax, sc, inv);
  uint32_t word = 0;
  for (int j = 0; j < 4; ++j)
    word |= (uint32_t)__nv_cvt_float_to_fp8(__fmul_rn(x[j], inv), __NV_SATFINITE, __NV_E4M3) << (8 * j);
  *reinterpret_cast<uint32_t*>(out + threadIdx.x * 4) = word;
  if (threadIdx.x == 0) *reinterpret_cast<float*>(out + D) = sc;
}

// glm_attn::scores_body on kv8 rows: the same tiles, dots and epilogue, each dot times its pool's scale first.
__device__ __forceinline__ void scores_body(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                                            const unsigned char* pk, float* out, const int* pos, int64_t np,
                                            int64_t nb, float scale, float wscale, int64_t heads, int loop,
                                            const int* pt, int64_t ps) {
  constexpr int HP = 32, BP = 64;
  __shared__ __align__(16) __nv_bfloat16 sm[4 * 2 * 16 * D];  // q's 32 rows, then each warp's two 16-pool buffers
  auto at = [](int row, int u) { return row * D + ((u ^ (row & 7)) << 3); };
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int64_t r = blockIdx.x, npool = ((int64_t)pos[0] + r + 1) / 4;
  for (int idx = threadIdx.x; idx < HP * (D / 8); idx += 128) {
    const int hh = idx / (D / 8), u = idx % (D / 8);
    const bool ok = hh < heads;
    glm_attn::cp16(sm + at(hh, u), qi + (r * heads + (ok ? hh : 0)) * D + u * 8, ok);
  }
  glm_attn::commit();
  float wv[4];
  for (int i = 0; i < 4; ++i) {
    const int64_t hh = g + 8 * i;
    wv[i] = hh < heads ? glm_triton::fmul(__bfloat162float(wts[r * w_stride + hh]), wscale) : 0.0f;
  }
  // A block's pools for this warp, rows p0 .. p0 + 15: `fetch` reads a lane's 8 chunks of 8 codes and its 4 pools'
  // scales into registers (zeros and 1 past npool), `put` converts the codes into buffer `buf` after the dots.
  uint2 fc[8];
  float kn[2][2], kc[2][2];
  auto fetch = [&](int64_t p0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int idx = lane + t * 32, row = idx / (D / 8), u = idx % (D / 8);
      const int64_t p = p0 + row;
      fc[t] = codes8(pk + row_at(pt, ps, p < npool ? p : 0), u * 8, p < npool);
    }
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) {
        const int64_t p = p0 + nt * 8 + q * 2 + e;
        kn[nt][e] = row_scale(pk + row_at(pt, ps, p < npool ? p : 0), p < npool);
      }
  };
  auto put = [&](int buf) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int idx = lane + t * 32, row = idx / (D / 8), u = idx % (D / 8);
      *reinterpret_cast<uint4*>(sm + (warp * 2 + buf) * 16 * D + at(row, u)) = bf16x8_of(fc[t]);
    }
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) kc[nt][e] = kn[nt][e];
  };
  const int64_t first = (int64_t)blockIdx.y * loop;
  auto live = [&](int j) { const int64_t pb = first + j; return j < loop && pb < nb && pb * BP < npool; };
  glm_attn::wait<0>();
  __syncthreads();
  uint32_t aq[D / 16][2][4];
#pragma unroll
  for (int kq = 0; kq < D / 16; ++kq)
#pragma unroll
    for (int i = 0; i < 2; ++i) glm_attn::ldm4<false>(aq[kq][i], sm + at(i * 16 + (lane & 15), kq * 2 + (lane >> 4)));
  __syncthreads();  // q's fragments are in registers: the key buffers take its place
  if (live(0)) {
    fetch(first * BP + warp * 16);
    put(0);
  }
  for (int j = 0; j < loop; ++j) {
    const int64_t pb = first + j;
    if (pb >= nb) break;
    const int64_t p0 = pb * BP + warp * 16;
    if (!live(j)) {  // no later block is live either: nothing is fetched past here
      for (int e = lane; e < 16; e += 32)
        if (p0 + e < np) out[r * np + p0 + e] = -INFINITY;
      continue;
    }
    const bool next = live(j + 1);
    if (next) fetch(p0 + BP);  // its loads run under this block's dots
    float ks[2][2];
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) ks[nt][e] = kc[nt][e];
    __syncwarp();  // the stores of buffer j & 1 (an earlier step) are seen by every lane
    float acc[2][2][4] = {};
    const __nv_bfloat16* keys = sm + (warp * 2 + (j & 1)) * 16 * D;
#pragma unroll
    for (int kq = 0; kq < D / 16; ++kq) {
      uint32_t b[4];
      glm_attn::ldm4<false>(b, keys + at((lane >> 4) * 8 + (lane & 7), kq * 2 + ((lane >> 3) & 1)));
      for (int nt = 0; nt < 2; ++nt)
        for (int i = 0; i < 2; ++i) glm_router::mma16816(acc[i][nt], aq[kq][i], b + 2 * nt);
    }
    __syncwarp();  // every lane is done with buffer j & 1 before step j + 1 puts into it
    if (next) put((j + 1) & 1);
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) {
        const int64_t p = p0 + nt * 8 + q * 2 + e;
        const float k = ks[nt][e];
        const float v0 = fmaxf(__fmul_rn(__fmul_rn(acc[0][nt][e], k), scale), 0.0f),
                    v1 = fmaxf(__fmul_rn(__fmul_rn(acc[0][nt][e + 2], k), scale), 0.0f);
        const float v2 = fmaxf(__fmul_rn(__fmul_rn(acc[1][nt][e], k), scale), 0.0f),
                    v3 = fmaxf(__fmul_rn(__fmul_rn(acc[1][nt][e + 2], k), scale), 0.0f);
        float s = glm_triton::fmul(wv[1], v1);
        s = __fmaf_rn(wv[0], v0, s);
        s = __fmaf_rn(wv[2], v2, s);
        s = __fmaf_rn(wv[3], v3, s);
        for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
        if (g == 0 && p < np) out[r * np + p] = p < npool ? s : -INFINITY;
      }
  }
}

}  // namespace glm_pk8

// tf_glm_pool_keys with kv8 rows: a warp a completed pool, its row's codes and scale.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_pool_keys_pk8(const __nv_bfloat16* ik, const __nv_bfloat16* ig, const __nv_bfloat16* ape,
                         unsigned char* pk, const int* pos, int64_t rows, const int* pt, int64_t ps) {
  glm_pk8::pool_keys_body(ik, ig, ape, pk, pos, rows, pt, ps);
}

// tf_glm_scores on kv8 rows: block (row, L pool blocks) of 4 warps, 16 pools a warp.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_scores_pk8(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const unsigned char* pk,
                      float* out, const int* pos, int64_t np, int64_t nb, float scale, float wscale, int64_t heads,
                      int loop, const int* pt, int64_t ps) {
  glm_pk8::scores_body(qi, wts, w_stride, pk, out, pos, np, nb, scale, wscale, heads, loop, pt, ps);
}

// tf_glm_pool_keys_seg with kv8 rows (TF_GLM_OVN_SEG_DSA): a segment a grid y, its rows / 4 + 2 blocks.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_pool_keys_seg_pk8(const glm_seg::Seg* segs, const glm_seg::Cache* cache, const __nv_bfloat16* ape) {
  const glm_seg::Seg g = segs[blockIdx.y];
  if ((int64_t)blockIdx.x >= g.rows / 4 + 2) return;
  const glm_seg::Cache k = cache[blockIdx.y];
  glm_pk8::pool_keys_body(reinterpret_cast<const __nv_bfloat16*>(k.ik), reinterpret_cast<const __nv_bfloat16*>(k.ig),
                          ape, reinterpret_cast<unsigned char*>(k.pk), g.pos, g.rows, k.pt, k.ps);
}

// tf_glm_scores_seg on kv8 rows: a listed segment a grid z, into its scores rows.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_scores_seg_pk8(const glm_seg::Seg* segs, const int* list, const glm_seg::Cache* cache,
                          const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, float* scores,
                          float scale, float wscale, int64_t heads) {
  const int s = glm_seg::pick(list, blockIdx.z);
  const glm_seg::Seg g = segs[s];
  if ((int64_t)blockIdx.x >= g.rows || (int64_t)blockIdx.y >= g.nb) return;
  const glm_seg::Cache k = cache[s];
  glm_pk8::scores_body(qi + g.row0 * heads * 128, wts + g.row0 * w_stride, w_stride,
                       reinterpret_cast<const unsigned char*>(k.pk), scores + g.soff, g.pos, g.np, g.nb, scale, wscale,
                       heads, 1, k.pt, k.ps);
}
