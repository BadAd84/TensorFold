// GLM-5.3-Flash DSA indexer, a decode window's top-K pools over chunk CTAs (TF_GLM_OVN_DECODE_INDEX): its tokens.
// Port of production's select_split (prod-src tensorfold/families/glm5_next/cuda/sparse.py _select_split, Apache-2.0).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "glm_index_pools.cuh"

namespace glm_split {

constexpr int T = 128, WARPS = T / 32, PER = 8, SPAN = T * PER;  // threads, warps, keys a thread a scan, a scan
constexpr long long PAGE = 512;                                   // rows a KV page holds (tf_glm_dsa_tokens' PAGE)

struct Sh {
  int hist[256];
  int wsum[2][WARPS];  // block_scan's warp totals, two slots taken in turn
  int red[2][WARPS];   // block_sum2's warp totals
  uint32_t prefix, fixed;
  int need, digit;
};

// a and b summed over the block (integers: any order); every thread calls it and gets the totals.
__device__ __forceinline__ void block_sum2(Sh& sh, int& a, int& b) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, o);
    b += __shfl_xor_sync(0xffffffffu, b, o);
  }
  if (lane == 0) {
    sh.red[0][w] = a;
    sh.red[1][w] = b;
  }
  __syncthreads();
  a = 0;
  b = 0;
  for (int i = 0; i < WARPS; ++i) {
    a += sh.red[0][i];
    b += sh.red[1][i];
  }
  __syncthreads();  // every thread has read sh.red before a later call writes it
}

// Exclusive sum of v in thread order and the block's total, via slot `slot` (two slots, used in turn per scan).
__device__ __forceinline__ int block_scan(Sh& sh, int v, int slot, int& total) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  int inc = v;
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += u;
  }
  if (lane == 31) sh.wsum[slot][w] = inc;
  __syncthreads();
  int before = inc - v;
  total = 0;
  for (int i = 0; i < WARPS; ++i) {
    if (i < w) before += sh.wsum[slot][i];
    total += sh.wsum[slot][i];
  }
  return before;
}

// sparse._split_digits from the row's totals tot [4][256] by warp 0, with tf_glm_select_rows' bin walk (the digit is
// the highest bin whose count from the top reaches need; none: 0 and need less every pool), into sh.
__device__ __forceinline__ void digits(Sh& sh, const int* __restrict__ tot, int steps, int K) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    uint32_t prefix = 0, fixed = 0;
    int need = K, digit = 0;
    for (int p = 0; p < steps; ++p) {
      int h[8], own = 0;
      for (int j = 0; j < 8; ++j) own += h[j] = tot[p * 256 + 255 - 8 * lane - j];
      int upto = own;
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, upto, o);
        if (lane >= o) upto += v;
      }
      const unsigned hit = __ballot_sync(0xffffffffu, upto >= need);
      const int src = hit ? __ffs(hit) - 1 : 31;
      int dg = 0, left = 0;
      if (lane == src) {
        int above = upto - own;
        for (int j = 0; j < 8; ++j) {
          if (above + h[j] >= need) {
            dg = 255 - 8 * lane - j;
            break;
          }
          above += h[j];
        }
        left = need - above;
      }
      digit = __shfl_sync(0xffffffffu, dg, src);
      need = __shfl_sync(0xffffffffu, left, src);
      prefix |= (uint32_t)digit << (24 - 8 * p);
      fixed |= 0xFFu << (24 - 8 * p);
    }
    if (lane == 0) {
      sh.prefix = prefix;
      sh.fixed = fixed;
      sh.need = need;
      sh.digit = digit;
    }
  }
  __syncthreads();
}

// The order keys of the chunk's scores i0 .. i0 + PER - 1 (0 past n).
__device__ __forceinline__ void keys(const float* __restrict__ s, int n, int i0, uint32_t (&u)[PER]) {
  if (i0 + PER <= n && ((uintptr_t)(s + i0) & 15) == 0) {
    const float4 a = *reinterpret_cast<const float4*>(s + i0), b = *reinterpret_cast<const float4*>(s + i0 + 4);
    const float v[PER] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < PER; ++j) u[j] = glm_index::order_key(v[j]);
  } else {
#pragma unroll
    for (int j = 0; j < PER; ++j) u[j] = i0 + j < n ? glm_index::order_key(s[i0 + j]) : 0u;
  }
}

// Token t's row: in order (pt null), or row t % PAGE of page pt[t / PAGE] (tf_glm_dsa_tokens' `at`).
__device__ __forceinline__ int token_at(const int* __restrict__ pt, long long page_rows, long long t) {
  return static_cast<int>(pt ? pt[t / PAGE] * page_rows + t % PAGE : t);
}

}  // namespace glm_split

// Program (row, chunk) of step 0..3 (pass `step` of the radix select over the chunk's chs scores: its count above the
// prefix into GT, its histogram into HIST and the row's totals TOT) or step 4 (the chunk's picks as token rows at the
// offset the earlier chunks give, chunk 0 the tail and count): tf_glm_select_rows then tf_glm_dsa_tokens' bytes.
// S [R][np] fp32, row r at pos[0] + r over its first min(np, max(pools, K)) scores; TOT [R][4][256] zeroed before
// step 0, HIST [R][cp][256], GT [R][cp] int32; TOK [R][4 K + 3], CNT [R] int32; grid (R, cp), glm_split::T threads.
extern "C" __global__ void __launch_bounds__(glm_split::T)
    tf_glm_index_split(const float* __restrict__ S, int64_t np, const int* __restrict__ pos, int K, int chs, int cp,
                       int step, int* __restrict__ TOT, int* __restrict__ HIST, int* __restrict__ GT,
                       int* __restrict__ TOK, int* __restrict__ CNT, const int* __restrict__ pt, int64_t page_rows) {
  using namespace glm_split;
  __shared__ Sh sh;
  const int64_t r = blockIdx.x;
  const int c = blockIdx.y;
  const int64_t q = (int64_t)pos[0] + r, npool = (q + 1) / 4;
  const int64_t NP = min(np, max(npool, (int64_t)K));  // sparse._visible_bound
  const int64_t i0 = (int64_t)c * chs;
  if (i0 >= NP) return;  // a chunk past the row's bound: the whole block leaves
  const int n = (int)min((int64_t)chs, NP - i0);
  const float* row = S + r * np + i0;
  int* own = HIST + (r * cp + c) * 256;
  int* tot = TOT + r * 1024;
  digits(sh, tot, step, K);
  const uint32_t prefix = sh.prefix, fixed = sh.fixed;
  const int need = sh.need, d = sh.digit;
  if (step < 4) {
    if (step > 0) {  // the chunk's scores above the prefix first at the byte just fixed (its last histogram)
      int a = 0, none = 0;
      for (int b = threadIdx.x; b < 256; b += T) a += b > d ? own[b] : 0;
      block_sum2(sh, a, none);
      if (threadIdx.x == 0) GT[r * cp + c] = a + (step > 1 ? GT[r * cp + c] : 0);
    }
    for (int b = threadIdx.x; b < 256; b += T) sh.hist[b] = 0;
    __syncthreads();  // also: every read of `own` above is done before it is rewritten below
    const int lane = threadIdx.x & 31, shift = 24 - 8 * step;
    const unsigned below = (1u << lane) - 1;
    for (int k0 = 0; k0 < n; k0 += SPAN) {
      const int ib = k0 + (int)threadIdx.x * PER;
      uint32_t u[PER];
      keys(row, n, ib, u);
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        const int b = ib + j < n && (u[j] & fixed) == prefix ? (int)((u[j] >> shift) & 0xFFu) : 256;
        if (__any_sync(0xffffffffu, b < 256)) {
          const unsigned same = __match_any_sync(0xffffffffu, b);
          if (b < 256 && (same & below) == 0) atomicAdd(&sh.hist[b], __popc(same));
        }
      }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += T) {
      const int v = sh.hist[b];
      own[b] = v;
      if (v != 0) atomicAdd(&tot[step * 256 + b], v);
    }
    return;
  }
  // step 4: the earlier chunks' scores above the K-th best's key (GT and their last histograms) and equal to it
  int above = 0, equal = 0;
  for (int i = threadIdx.x; i < c * 256; i += T) {
    const int b = i & 255, v = HIST[(r * cp + (i >> 8)) * 256 + b];
    above += b > d ? v : 0;
    equal += b == d ? v : 0;
  }
  for (int i = threadIdx.x; i < c; i += T) above += GT[r * cp + i];
  block_sum2(sh, above, equal);
  int* out = TOK + r * (4 * K + 3);
  int run_take = above + min(equal, need), run_eq = equal;
  for (int k0 = 0; k0 < n; k0 += SPAN) {  // in pool order: every key above the prefix, and the first `need` equal
    const int ib = k0 + (int)threadIdx.x * PER;
    uint32_t u[PER];
    keys(row, n, ib, u);
    bool eq[PER], gt[PER];
    int eqs = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const bool in = ib + j < n;
      eq[j] = in && u[j] == prefix;
      gt[j] = in && u[j] > prefix;
      eqs += eq[j];
    }
    int all_eq, all_take;
    int seen = run_eq + block_scan(sh, eqs, 0, all_eq);
    bool take[PER];
    int takes = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      take[j] = gt[j] || (eq[j] && seen < need);
      seen += eq[j];
      takes += take[j];
    }
    int at = run_take + block_scan(sh, takes, 1, all_take);
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if (take[j]) {
        const long long p = i0 + ib + j;
        for (int t = 0; t < 4; ++t) out[4 * at + t] = token_at(pt, page_rows, p * 4 + t);
        at++;
      }
    run_eq += all_eq;
    run_take += all_take;
  }
  if (c == 0 && threadIdx.x < 3) {  // the incomplete pool's visible tokens, as tf_glm_dsa_tokens writes them
    const long long t = npool * 4 + threadIdx.x;
    out[4 * K + threadIdx.x] = t <= q ? token_at(pt, page_rows, t) : -1;
  }
  if (c == 0 && threadIdx.x == 0) {
    int tail = 0;
    for (int t = 0; t < 3; ++t) tail += npool * 4 + t <= q;
    CNT[r] = npool > K ? 4 * K + tail : 0;
  }
}
