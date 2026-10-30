// TF_GLM_OVN_PK8 (TRADE): tf_glm_index_scores on kv8 pooled keys (glm_pk8_row.cuh), production's FP8 scores
// (prod-src families/glm5_next/cuda/prompt_scores.cu: codes to bf16 exactly, each pool's scale on its dots).
#pragma once

#include "glm_index_scores.cuh"
#include "glm_pk8_row.cuh"

// tf_glm_index_scores with each tile converted from codes and its 64 scales kept beside it; the same grid and args.
extern "C" __global__ void __launch_bounds__(glm_index::RB * 32)
    tf_glm_index_scores_pk8(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                            const unsigned char* pk, float* out, const int* pos, int64_t rows, int64_t np, int64_t nb,
                            float scale, float wscale, int64_t heads, int loop, const int* pt, int64_t ps) {
  using namespace glm_index;
  __shared__ __align__(16) __nv_bfloat16 tile[2][BP * D];  // two key tiles
  __shared__ float ksm[2][BP];                              // and their pools' scales
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int64_t r = (int64_t)blockIdx.x * RB + warp, P = pos[0];
  const bool live = r < rows;
  const int64_t npool = live ? (P + r + 1) / 4 : 0;
  const int64_t last = min(rows, (int64_t)(blockIdx.x + 1) * RB) - 1, nmax = (P + last + 1) / 4;  // the CTA's largest
  uint32_t aq[KQ][2][4];
#pragma unroll
  for (int kq = 0; kq < KQ; ++kq)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int h0 = i * 16 + g, h1 = h0 + 8;
      const __nv_bfloat16* b0 = qi + ((live ? r : 0) * heads + (h0 < heads ? h0 : 0)) * D + kq * 16 + 2 * q;
      const __nv_bfloat16* b1 = qi + ((live ? r : 0) * heads + (h1 < heads ? h1 : 0)) * D + kq * 16 + 2 * q;
      const bool o0 = live && h0 < heads, o1 = live && h1 < heads;
      aq[kq][i][0] = o0 ? *reinterpret_cast<const uint32_t*>(b0) : 0u;
      aq[kq][i][1] = o1 ? *reinterpret_cast<const uint32_t*>(b1) : 0u;
      aq[kq][i][2] = o0 ? *reinterpret_cast<const uint32_t*>(b0 + 8) : 0u;
      aq[kq][i][3] = o1 ? *reinterpret_cast<const uint32_t*>(b1 + 8) : 0u;
    }
  float wv[4];
  for (int i = 0; i < 4; ++i) {
    const int64_t hh = g + 8 * i;
    wv[i] = live && hh < heads ? __fmul_rn(__bfloat162float(wts[r * w_stride + hh]), wscale) : 0.0f;
  }
  // The CTA's 256 threads take tile pb in 4 chunks each: `fetch` reads their codes and (chunk 0 of a pool) the pool's
  // scale into registers (zeros and 1 past the CTA's visible pools), `put` converts them into buffer `buf`.
  uint2 fc[4];
  float fs[4];
  auto fetch = [&](int64_t pb) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = threadIdx.x + k * RB * 32, row = idx / (D / 8), u = idx % (D / 8);
      const int64_t p = pb * BP + row;
      const bool ok = p < nmax;
      const unsigned char* src = pk + glm_pk8::row_at(pt, ps, ok ? p : 0);
      fc[k] = glm_pk8::codes8(src, u * 8, ok);
      fs[k] = glm_pk8::row_scale(src, ok && u == 0);
    }
  };
  auto put = [&](int buf) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = threadIdx.x + k * RB * 32, row = idx / (D / 8), u = idx % (D / 8);
      *reinterpret_cast<uint4*>(&tile[buf][tile_at(row, u)]) = glm_pk8::bf16x8_of(fc[k]);
      if (u == 0) ksm[buf][row] = fs[k];
    }
  };
  const int64_t first = (int64_t)blockIdx.y * loop;
  auto any = [&](int j) { return j < loop && first + j < nb && (first + j) * BP < nmax; };
  if (any(0)) {
    fetch(first);
    put(0);
  }
  for (int j = 0; j < loop; ++j) {
    const int64_t pb = first + j, p0 = pb * BP;
    if (pb >= nb) break;
    __syncthreads();  // tile j is in, and every warp is done with tile j - 1 (the buffer tile j + 1 takes)
    const bool next = any(j + 1);
    if (next) fetch(pb + 1);  // its loads run under this tile's dots
    if (live && p0 >= npool) {  // the row sees none of these pools: -inf
      for (int e = lane; e < BP; e += 32)
        if (p0 + e < np) out[r * np + p0 + e] = -INFINITY;
    } else if (live) {
      for (int gi = 0; gi < BP / 16; ++gi) {
        float acc[2][2][4] = {};
        const __nv_bfloat16* keys = tile[j & 1] + gi * 16 * D;
#pragma unroll
        for (int kq = 0; kq < KQ; ++kq) {
          uint32_t b[4];
          ldm4(b, keys + tile_at((lane >> 4) * 8 + (lane & 7), kq * 2 + ((lane >> 3) & 1)));
          for (int nt = 0; nt < 2; ++nt)
            for (int i = 0; i < 2; ++i) mma16816(acc[i][nt], aq[kq][i], b + 2 * nt);
        }
        for (int nt = 0; nt < 2; ++nt)
          for (int e = 0; e < 2; ++e) {
            const int c = gi * 16 + nt * 8 + q * 2 + e;
            const float ks = ksm[j & 1][c];
            const float v0 = fmaxf(__fmul_rn(__fmul_rn(acc[0][nt][e], ks), scale), 0.0f),
                        v1 = fmaxf(__fmul_rn(__fmul_rn(acc[0][nt][e + 2], ks), scale), 0.0f);
            const float v2 = fmaxf(__fmul_rn(__fmul_rn(acc[1][nt][e], ks), scale), 0.0f),
                        v3 = fmaxf(__fmul_rn(__fmul_rn(acc[1][nt][e + 2], ks), scale), 0.0f);
            float s = __fmul_rn(wv[1], v1);
            s = __fmaf_rn(wv[0], v0, s);
            s = __fmaf_rn(wv[2], v2, s);
            s = __fmaf_rn(wv[3], v3, s);
            for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
            const int64_t p = p0 + c;
            if (g == 0 && p < np) out[r * np + p] = p < npool ? s : -INFINITY;
          }
      }
    }
    if (next) put((j + 1) & 1);
  }
}
