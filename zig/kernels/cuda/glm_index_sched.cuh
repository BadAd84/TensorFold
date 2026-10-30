// GLM-5.3-Flash DSA indexer: tf_glm_index_scores with production's 0084 v2 stores and bound (TF_GLM_OVN_INDEX_SCHED).
// Schedule from prod-src tensorfold/families/glm5_next/cuda/prompt_scores.cu (Apache-2.0): a warp's 64 scores of a
// tile staged in shared memory and stored as one float2 a lane, the CTA held to 128 registers for 2 CTAs an SM.
#pragma once

#include "glm_index_scores.cuh"

// tf_glm_index_scores bit for bit (the same dots and epilogue on the same values); only the stores and the register
// bound differ. Grid (cdiv(rows, 8), cdiv(nb, loop)).
extern "C" __global__ void __launch_bounds__(glm_index::RB * 32, 2)
    tf_glm_index_scores_sched(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                              const __nv_bfloat16* pk, float* out, const int* pos, int64_t rows, int64_t np, int64_t nb,
                              float scale, float wscale, int64_t heads, int loop, const int* pt, int64_t ps) {
  using namespace glm_index;
  __shared__ __align__(16) __nv_bfloat16 tile[2][BP * D];  // two key tiles
  __shared__ __align__(16) float stage_out[RB][BP];         // a warp's 64 scores of a tile before the row's store
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
  auto stage = [&](int buf, int64_t pb) {
    for (int idx = threadIdx.x; idx < BP * (D / 8); idx += RB * 32) {
      const int row = idx / (D / 8), u = idx % (D / 8);
      const int64_t p = pb * BP + row;
      const bool ok = p < nmax;
      cp16(&tile[buf][tile_at(row, u)], pk + key_at(pt, ps / 2, ok ? p : 0) + u * 8, ok);
    }
  };
  // the warp's staged scores into row r's columns p0 .. p0 + 63 (as one float2 a lane where the row allows it)
  float* st = stage_out[warp];
  auto store = [&](int64_t p0) {
    __syncwarp();
    float* o = out + r * np + p0;
    if (p0 + BP <= np && (np & 1) == 0)
      *reinterpret_cast<float2*>(o + 2 * lane) = *reinterpret_cast<const float2*>(&st[2 * lane]);
    else
      for (int e = 0; e < 2; ++e)
        if (p0 + 2 * lane + e < np) o[2 * lane + e] = st[2 * lane + e];
    __syncwarp();  // every lane has read st before the next tile writes it
  };
  const int64_t first = (int64_t)blockIdx.y * loop;
  auto any = [&](int j) { return j < loop && first + j < nb && (first + j) * BP < nmax; };
  if (any(0)) stage(0, first);
  cp_commit();
  for (int j = 0; j < loop; ++j) {
    const int64_t pb = first + j, p0 = pb * BP;
    if (pb >= nb) break;
    cp_wait();
    __syncthreads();  // tile j is in, and every warp is done with tile j - 1
    if (any(j + 1)) stage((j + 1) & 1, pb + 1);
    cp_commit();
    if (!live) continue;
    if (p0 >= npool) {  // the row sees none of these pools: -inf
      st[2 * lane] = -INFINITY;
      st[2 * lane + 1] = -INFINITY;
      store(p0);
      continue;
    }
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
          const float v0 = fmaxf(__fmul_rn(acc[0][nt][e], scale), 0.0f),
                      v1 = fmaxf(__fmul_rn(acc[0][nt][e + 2], scale), 0.0f);
          const float v2 = fmaxf(__fmul_rn(acc[1][nt][e], scale), 0.0f),
                      v3 = fmaxf(__fmul_rn(acc[1][nt][e + 2], scale), 0.0f);
          float s = __fmul_rn(wv[1], v1);
          s = __fmaf_rn(wv[0], v0, s);
          s = __fmaf_rn(wv[2], v2, s);
          s = __fmaf_rn(wv[3], v3, s);
          for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
          const int c = gi * 16 + nt * 8 + q * 2 + e;
          if (g == 0) st[c] = p0 + c < npool ? s : -INFINITY;
        }
    }
    store(p0);
  }
}
