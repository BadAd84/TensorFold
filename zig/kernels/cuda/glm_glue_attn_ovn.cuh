// TF_GLM_OVN_MLA_FUSEMERGE: glm_glue_attn.cuh's chunk pass stopped at a chunk's last live key tile, and fused with
// tf_glm_merge (one CTA a row and head block walks the row's chunks, merging each in registers as tf_glm_merge does).
#pragma once

#include "glm_glue_attn.cuh"

namespace glm_attn_ovn {

// Key tiles of chunk `start` up to its last live key (sparse: keys below n; dense: keys through limit); a live chunk.
__device__ __forceinline__ int live_tiles(int64_t start, int64_t n, int64_t limit, int sparse) {
  const int64_t past = start + glm_attn::CH, keys = sparse ? n : limit + 1;
  return (int)(((keys < past ? keys : past) - start + glm_attn::KT - 1) / glm_attn::KT);
}

// glm_attn::chunks_wide with the tile loop bounded; FUSED: blockIdx.z 0 walks chunks 0 .. nch - 1 and writes `out`.
template <int HB, bool FP8, bool FUSED>
__device__ void chunks(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const float* scales, const int* pos,
                       const int* tok, const int* cnt, float* po, float* pm, float* pl, __nv_bfloat16* out, int nch,
                       int64_t rows, int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  using namespace glm_attn;
  constexpr int MT = HB / 16, FT = FUSED ? MT : 1;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + HB * LW;
  __nv_bfloat16* ps = kvs + 2 * KT * LW;
  float* red = reinterpret_cast<float*>(ps + HB * KT);
  float* stat = red + HB * 4;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
  const int kg = warp & 3, hh = warp >> 2;
  const int64_t r = blockIdx.x, h0 = (int64_t)blockIdx.y * HB;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r;
  if (FUSED && sparse && n == 0) return;  // tf_glm_merge leaves a row with count 0 alone
  int64_t c = FUSED ? 0 : (int64_t)blockIdx.z, start = 0;
  const __nv_bfloat16* keys_at = lc;
  auto load_kv = [&](int tile) {  // BF16: the tile's rows by cp.async into its buffer
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    for (int idx = t; idx < KT * (LW / 8); idx += blockDim.x) {
      const int key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      cp16(dst + wide(key, u), keys_at + src * LW + u * 8, ok);
    }
  };
  constexpr int PER = FP8 ? KT * (LW / 8) / 256 : 1;
  uint2 pre_c[PER];
  float pre_s[PER];
  const uint32_t page_rows = pt ? (uint32_t)(pstep / LW) : 1u;
  auto fetch = [&](int tile) {  // FP8: a tile's codes and scales into registers, as chunks_wide's fetch
    const int64_t first = start + tile * KT;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int idx = t + p * 256, key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      const unsigned char* codes = reinterpret_cast<const unsigned char*>(lc) + src * LW;
      const float* row_scales = scales + src;
      if (pt && ok) {
        const uint32_t off = sparse ? (uint32_t)src % page_rows : (uint32_t)(ki - start);
        const unsigned char* page = sparse ? codes - (int64_t)off * LW
                                           : reinterpret_cast<const unsigned char*>(lc) + (int64_t)pt[c] * pstep;
        codes = page + (int64_t)off * LW;
        row_scales = reinterpret_cast<const float*>(page + CH * LW) + off;
      }
      pre_c[p] = ok ? *reinterpret_cast<const uint2*>(codes + u * 8) : make_uint2(0u, 0u);
      pre_s[p] = ok ? *row_scales : 1.0f;
    }
  };
  auto store = [&](int tile) {
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int idx = t + p * 256, key = idx / (LW / 8), u = idx % (LW / 8);
      uint32_t words[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint16_t bits = (uint16_t)(((j < 2 ? pre_c[p].x : pre_c[p].y) >> (16 * (j & 1))) & 0xffffu);
        const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
        const __nv_bfloat162 b = __floats2bfloat162_rn(__fmul_rn(v.x, pre_s[p]), __fmul_rn(v.y, pre_s[p]));
        words[j] = *reinterpret_cast<const uint32_t*>(&b);
      }
      *reinterpret_cast<uint4*>(dst + wide(key, u)) = make_uint4(words[0], words[1], words[2], words[3]);
    }
  };
  float m[MT][2], l[MT][2], o[MT][8][4];
  float fm[FT][2], fl[FT][2], fo[FT][8][4];  // FUSED: tf_glm_merge's running m, l and o
  for (int i = 0; i < FT; ++i)
    for (int e = 0; e < 2; ++e) fm[i][e] = -INFINITY, fl[i][e] = 0.0f;
  for (int i = 0; i < FT; ++i)
    for (int j = 0; j < 8; ++j)
      for (int e = 0; e < 4; ++e) fo[i][j][e] = 0.0f;
  for (int ci = 0; ci < (FUSED ? nch : 1); ++ci) {
    if (FUSED) c = ci;
    start = c * CH;
    const bool live = sparse ? start < n : start <= limit;
    keys_at = !FP8 && pt && !sparse && live ? lc + ((int64_t)pt[c] * (pstep / 2) - start * LW) : lc;
    const int tiles = live ? live_tiles(start, n, limit, sparse) : 0;
    for (int i = 0; i < MT; ++i)
      for (int e = 0; e < 2; ++e) m[i][e] = -INFINITY, l[i][e] = 0.0f;
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < 8; ++j)
        for (int e = 0; e < 4; ++e) o[i][j][e] = 0.0f;
    if (live) {
      if (!FUSED || ci == 0)  // the query stays in shared memory for every chunk of the row
        for (int idx = t; idx < HB * (LW / 8); idx += blockDim.x) {
          const int hd = idx / (LW / 8), u = idx % (LW / 8);
          const bool ok = h0 + hd < heads;
          cp16(qs + wide(hd, u), qa + (r * heads + (ok ? h0 + hd : 0)) * LW + u * 8, ok);
        }
      if constexpr (FP8) {
        fetch(0);
        store(0);
        fetch(1);
        commit();
      } else {
        load_kv(0);
        commit();
        load_kv(1);
        commit();
      }
    }
    for (int tile = 0; tile < tiles; ++tile) {
      if constexpr (FP8) {
        wait<0>();
      } else {
        if (tile + 1 < tiles)
          wait<1>();
        else
          wait<0>();
      }
      __syncthreads();
      if constexpr (FP8) {
        if (tile + 1 < tiles) store(tile + 1);
        if (tile + 2 < tiles) fetch(tile + 2);
      }
      const __nv_bfloat16* kv = kvs + (tile & 1) * KT * LW;
      const int64_t first = start + tile * KT;
      float sc[4];
      if (hh < MT) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
        for (int kk = 0; kk < LW; kk += 32) {
          uint32_t a0[4], a1[4], b[4];
          ldm4<false>(a0, qs + wide(hh * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
          ldm4<false>(a1, qs + wide(hh * 16 + (lane & 15), ((kk + 16) >> 3) + (lane >> 4)));
          ldm4<false>(b, kv + wide(kg * 8 + (lane & 7), (kk >> 3) + (lane >> 3)));
          glm_router::mma16816(acc, a0, b);
          glm_router::mma16816(acc, a1, b + 2);
        }
        for (int e = 0; e < 2; ++e) {
          const int64_t ki = first + kg * 8 + q * 2 + e;
          const bool ok = sparse ? ki < n : ki <= limit;
          sc[e] = ok ? __fmul_rn(acc[e], scale) : -INFINITY;
          sc[e + 2] = ok ? __fmul_rn(acc[e + 2], scale) : -INFINITY;
        }
        for (int hf = 0; hf < 2; ++hf) {
          const float mx = quad_max(fmaxf(sc[hf * 2], sc[hf * 2 + 1]));
          if (q == 0) red[(hh * 16 + hf * 8 + g) * 4 + kg] = mx;
        }
      }
      __syncthreads();
      if (t < HB * 4) {
        float v = red[t];
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
        if ((t & 3) == 0) stat[t / 4] = v;
      }
      __syncthreads();
      float alpha[MT][2];
      for (int i = 0; i < MT; ++i)
        for (int hf = 0; hf < 2; ++hf) {
          const float tm = stat[i * 16 + hf * 8 + g];
          const bool active = tm != -INFINITY;
          const float nm = active ? fmaxf(m[i][hf], tm) : m[i][hf];
          alpha[i][hf] = active ? (m[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[i][hf] - nm)) : 1.0f;
          if (i == hh) {
            float p[2];
            for (int e = 0; e < 2; ++e) {
              const int64_t ki = first + kg * 8 + q * 2 + e;
              const bool ok = sparse ? ki < n : ki <= limit;
              p[e] = active && ok ? glm_triton::exp_approx(sc[hf * 2 + e] - nm) : 0.0f;
            }
            const int hr = i * 16 + hf * 8 + g, col = kg * 8 + q * 2;
            *reinterpret_cast<__nv_bfloat162*>(ps + narrow(hr, col >> 3) + (col & 7)) = __floats2bfloat162_rn(p[0], p[1]);
            const float s = quad_sum(p[0] + p[1]);
            if (q == 0) red[hr * 4 + kg] = s;
          }
          m[i][hf] = nm;
        }
      __syncthreads();
      if (t < HB * 4) {
        float v = red[t];
        v = v + __shfl_xor_sync(0xffffffffu, v, 2);
        v = v + __shfl_xor_sync(0xffffffffu, v, 1);
        if ((t & 3) == 0) stat[HB + t / 4] = v;
      }
      for (int i = 0; i < MT; ++i)
        for (int j = 0; j < 8; ++j)
          for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], alpha[i][e / 2]);
#pragma unroll
      for (int k16 = 0; k16 < KT; k16 += 16) {
        uint32_t a[MT][4];
        for (int i = 0; i < MT; ++i) ldm4<false>(a[i], ps + narrow(i * 16 + (lane & 15), (k16 >> 3) + (lane >> 4)));
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          uint32_t b[4];
          ldm4<true>(b, kv + wide(k16 + (lane & 15), ((warp * 64 + j * 8) >> 3) + (lane >> 4)));
          for (int i = 0; i < MT; ++i) {
            glm_router::mma16816(o[i][j], a[i], b);
            glm_router::mma16816(o[i][j + 1], a[i], b + 2);
          }
        }
      }
      __syncthreads();
      for (int i = 0; i < MT; ++i)
        for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], stat[HB + i * 16 + hf * 8 + g]);
      if constexpr (!FP8) {
        if (tile + 2 < tiles) load_kv(tile + 2);
        commit();
      }
    }
    // The tiles past the last live key, once: each is tile max -inf, alpha 1, p and keys +0 (zero-filled), so it
    // multiplies o by 1, adds two all-zero mma steps and adds +0 to l, and a second such tile changes nothing more.
    if (live && tiles < CH / KT) {
      const uint32_t z[4] = {0u, 0u, 0u, 0u};
      for (int i = 0; i < MT; ++i)
        for (int j = 0; j < 8; ++j)
          for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], 1.0f);
#pragma unroll
      for (int k16 = 0; k16 < KT; k16 += 16)
#pragma unroll
        for (int j = 0; j < 8; j += 2)
          for (int i = 0; i < MT; ++i) {
            glm_router::mma16816(o[i][j], z, z);
            glm_router::mma16816(o[i][j + 1], z, z + 2);
          }
      for (int i = 0; i < MT; ++i)
        for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], 1.0f, 0.0f);
    }
    if constexpr (FUSED) {  // tf_glm_merge's step on chunk ci's (m, l, o), the values it would read back from po/pm/pl
      float a[FT][2], b[FT][2];
      for (int i = 0; i < FT; ++i)
        for (int hf = 0; hf < 2; ++hf) {
          const float cm = m[i][hf], cl = l[i][hf];
          const bool active = cl > 0.0f;
          const float nm = active ? fmaxf(fm[i][hf], cm) : fm[i][hf];
          b[i][hf] = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
          a[i][hf] = ci == 0 ? 0.0f
                             : (active ? (fm[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(fm[i][hf] - nm)) : 1.0f);
          fl[i][hf] = ci == 0 ? __fmaf_rn(b[i][hf], cl, 0.0f) : __fmaf_rn(fl[i][hf], a[i][hf], __fmul_rn(b[i][hf], cl));
          fm[i][hf] = nm;
        }
      for (int i = 0; i < FT; ++i)
        for (int j = 0; j < 8; ++j)
          for (int e = 0; e < 4; ++e) {
            const float x = o[i][j][e], bb = b[i][e / 2];
            fo[i][j][e] = ci == 0 ? __fmaf_rn(bb, x, 0.0f) : __fmaf_rn(fo[i][j][e], a[i][e / 2], __fmul_rn(bb, x));
          }
    }
  }
  if constexpr (FUSED) {  // tf_glm_merge's bf16(o / l) for the block's real heads
    for (int i = 0; i < FT; ++i)
      for (int hf = 0; hf < 2; ++hf) {
        const int64_t hd = h0 + i * 16 + g + hf * 8;
        if (hd >= heads) continue;
        __nv_bfloat16* row = out + (r * heads + hd) * LW + warp * 64 + q * 2;
        for (int j = 0; j < 8; ++j)
          *reinterpret_cast<__nv_bfloat162*>(row + j * 8) =
              __halves2bfloat162(__float2bfloat16_rn(glm_triton::div_full(fo[i][j][hf * 2], fl[i][hf])),
                                 __float2bfloat16_rn(glm_triton::div_full(fo[i][j][hf * 2 + 1], fl[i][hf])));
      }
  } else {
    for (int i = 0; i < MT; ++i)
      for (int e = 0; e < 4; ++e) {
        const int64_t hd = h0 + i * 16 + g + (e / 2) * 8;
        if (hd >= heads) continue;
        const int64_t base = (c * rows + r) * heads + hd;
        for (int j = 0; j < 8; ++j) po[base * LW + warp * 64 + j * 8 + q * 2 + e % 2] = o[i][j][e];
        if (warp == 0 && q == 0 && e % 2 == 0) {
          pm[base] = m[i][e / 2];
          pl[base] = l[i][e / 2];
        }
      }
  }
}

}  // namespace glm_attn_ovn

// Bounded tile loops (decode windows and unfused prompt launches): tf_glm_chunks_*'s arguments and partials, same bits.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_skip(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                          const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                          float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn_ovn::chunks<16, false, false>(qa, lc, nullptr, pos, tok, cnt, po, pm, pl, nullptr, 1, rows, heads, w, scale,
                                         sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_skip(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                          const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                          float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn_ovn::chunks<32, false, false>(qa, lc, nullptr, pos, tok, cnt, po, pm, pl, nullptr, 1, rows, heads, w, scale,
                                         sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_skip(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                              const int* tok, const int* cnt, float* po, float* pm, float* pl, int64_t rows,
                              int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn_ovn::chunks<16, true, false>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), scales, pos, tok, cnt, po, pm,
                                        pl, nullptr, 1, rows, heads, w, scale, sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_skip(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                              const int* tok, const int* cnt, float* po, float* pm, float* pl, int64_t rows,
                              int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn_ovn::chunks<32, true, false>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), scales, pos, tok, cnt, po, pm,
                                        pl, nullptr, 1, rows, heads, w, scale, sparse, pt, pstep);
}

// Chunks and merge in one launch (grid rows x head blocks): out bf16 [rows][heads][512] as tf_glm_merge writes it.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fused(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                           const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t w, float scale,
                           int sparse, const int* pt, int64_t pstep, int nch) {
  glm_attn_ovn::chunks<32, false, true>(qa, lc, nullptr, pos, tok, cnt, nullptr, nullptr, nullptr, out, nch, rows, heads,
                                        w, scale, sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_fused(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                               const int* tok, const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads,
                               int64_t w, float scale, int sparse, const int* pt, int64_t pstep, int nch) {
  glm_attn_ovn::chunks<32, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), scales, pos, tok, cnt, nullptr,
                                       nullptr, nullptr, out, nch, rows, heads, w, scale, sparse, pt, pstep);
}
