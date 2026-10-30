// GLM-5.3-Flash DSA indexer over the ranks (TF_GLM_OVN_DECODE_SHARD): a rank's K best pools of its share, the merge.
// Production's decode pool shard (prod-src tensorfold/families/glm5_next/cuda/shard.py, OTS 0107/0112, Apache-2.0).
#pragma once

namespace glm_shard {

constexpr int PICK = 1024, MERGE = 256;  // a pick's threads (tf_glm_select_rows_wide's), a merge's
constexpr long long PAGE = 512;          // rows a KV page holds (tf_glm_dsa_tokens' PAGE)

// Token t's row: in order (pt null), or row t % PAGE of page pt[t / PAGE] (tf_glm_dsa_tokens' `at`).
__device__ __forceinline__ int token_at(const int* pt, long long page_rows, long long t) {
  return static_cast<int>(pt ? pt[t / PAGE] * page_rows + t % PAGE : t);
}

// glm_index::select_row's K best of n keys in index order (ties to the lower index), each taken one handed to
// take(i, slot, key) with its slot in index order. keys(i0, u, ok) fills keys i0 .. i0 + 7 and whether each is one
// of the n (ok false past n or for a hole). With `all` (every present key is taken: at most K) the passes are skipped.
template <int T, class Keys, class Take>
__device__ __forceinline__ void top_k(Keys keys, int64_t n, int k, bool all, Take take) {
  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_need, wsum[2][T / 32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const unsigned below = (1u << lane) - 1;
  uint32_t prefix = 0, fixed = 0;
  int need = k;
  for (int shift = 24; !all && shift >= 0; shift -= 8) {
    for (int b = t; b < 256; b += T) hist[b] = 0;
    __syncthreads();
    for (int64_t c = 0; c < n; c += 8 * T) {
      uint32_t u[8];
      bool ok[8];
      keys(c + 8 * t, u, ok);
      for (int j = 0; j < 8; ++j) {
        const int b = ok[j] && (u[j] & fixed) == prefix ? (int)((u[j] >> shift) & 0xFF) : 256;
        if (__any_sync(0xffffffffu, b < 256)) {
          const unsigned same = __match_any_sync(0xffffffffu, b);
          if (b < 256 && (same & below) == 0) atomicAdd(&hist[b], __popc(same));
        }
      }
    }
    __syncthreads();
    if (warp == 0) {  // select_row's bin walk: the digit is in the first lane whose count reaches need, else 0
      int h[8], own = 0;
      for (int j = 0; j < 8; ++j) own += h[j] = hist[255 - 8 * lane - j];
      int upto = own;
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, upto, o);
        if (lane >= o) upto += v;
      }
      const unsigned hit = __ballot_sync(0xffffffffu, upto >= need);
      if (lane == (hit ? __ffs(hit) - 1 : 31)) {
        int above = upto - own, digit = 0;
        for (int j = 0; j < 8; ++j) {
          if (above + h[j] >= need) {
            digit = 255 - 8 * lane - j;
            break;
          }
          above += h[j];
        }
        sh_need = need - above;
        sh_prefix = prefix | ((uint32_t)digit << shift);
      }
    }
    __syncthreads();
    need = sh_need;
    prefix = sh_prefix;
    fixed |= 0xFFu << shift;
    __syncthreads();
  }
  int run_take = 0, run_eq = 0;
  for (int64_t c = 0; c < n; c += 8 * T) {
    uint32_t u[8];
    bool ok[8], eq[8], gt[8];
    keys(c + 8 * t, u, ok);
    int eqs = 0;
    for (int j = 0; j < 8; ++j) {
      eq[j] = ok[j] && u[j] == prefix;
      gt[j] = ok[j] && u[j] > prefix;
      eqs += eq[j];
    }
    int all_eq, all_take;
    int seen = run_eq + glm_index::block_scan<T / 32>(eqs, wsum, 0, &all_eq);
    bool tk[8];
    int takes = 0;
    for (int j = 0; j < 8; ++j) {
      tk[j] = gt[j] || (eq[j] && seen < need);
      seen += eq[j];
      takes += tk[j];
    }
    int at = run_take + glm_index::block_scan<T / 32>(takes, wsum, 1, &all_take);
    for (int j = 0; j < 8; ++j)
      if (tk[j]) take(c + 8 * t + j, at++, u[j]);
    run_eq += all_eq;
    run_take += all_take;
  }
}

}  // namespace glm_shard

// A rank's share [off, off + ext) of each row's selectable pools [0, min(np, max(visible, K))) (tf_glm_select_rows'
// bound): its K best as pairs into OUT [R][2 K] int32 (order keys, then global pools ascending; pool -1 past them).
// S [R][np] fp32 (the share's columns written), row r at pos[0] + r; grid (R), glm_shard::PICK threads.
extern "C" __global__ void __launch_bounds__(glm_shard::PICK)
    tf_glm_index_shard_pick(const float* __restrict__ S, int64_t np, const int* __restrict__ pos, int64_t off,
                            int64_t ext, int K, int* __restrict__ OUT) {
  const int64_t r = blockIdx.x, vis = ((int64_t)pos[0] + r + 1) / 4;
  const int64_t limit = min(np, max(vis, (int64_t)K)), n = max((int64_t)0, min(limit - off, ext));
  const float* row = S + r * np + off;
  int* out = OUT + r * 2 * K;
  const bool vec = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
  auto keys = [&](int64_t i0, uint32_t (&u)[8], bool (&ok)[8]) {
    if (vec && i0 + 8 <= n) {
      const float4 a = *reinterpret_cast<const float4*>(row + i0), b = *reinterpret_cast<const float4*>(row + i0 + 4);
      const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      for (int j = 0; j < 8; ++j) u[j] = glm_index::order_key(v[j]), ok[j] = true;
    } else {
      for (int j = 0; j < 8; ++j) ok[j] = i0 + j < n, u[j] = ok[j] ? glm_index::order_key(row[i0 + j]) : 0u;
    }
  };
  glm_shard::top_k<glm_shard::PICK>(keys, n, K, n <= K, [&](int64_t i, int slot, uint32_t u) {
    out[slot] = (int)u;
    out[K + slot] = (int)(off + i);
  });
  for (int64_t s = min(n, (int64_t)K) + threadIdx.x; s < K; s += blockDim.x) {
    out[s] = 0;
    out[K + s] = -1;
  }
}

// Every rank's pairs gathered (RECV: rank blocks of R rows x 2 K, rank order = pool order) to each row's K best as
// tf_glm_select_rows then tf_glm_dsa_tokens write them: TOK [R][4 K + 3] (on pages their rows), CNT [R] int32.
// grid (R), glm_shard::MERGE threads.
extern "C" __global__ void __launch_bounds__(glm_shard::MERGE)
    tf_glm_index_shard_merge(const int* __restrict__ RECV, int64_t rows, int world, const int* __restrict__ pos, int K,
                             int* __restrict__ TOK, int* __restrict__ CNT, const int* __restrict__ pt,
                             int64_t page_rows) {
  const int64_t r = blockIdx.x, q = (int64_t)pos[0] + r, npool = (q + 1) / 4;
  auto pair = [&](int64_t c) { return RECV + ((c / K) * rows + r) * 2 * K + c % K; };
  auto keys = [&](int64_t i0, uint32_t (&u)[8], bool (&ok)[8]) {
    for (int j = 0; j < 8; ++j) {
      const int64_t c = i0 + j;
      ok[j] = c < (int64_t)world * K && pair(c)[K] >= 0;
      u[j] = ok[j] ? (uint32_t)pair(c)[0] : 0u;
    }
  };
  int* out = TOK + r * (4 * K + 3);
  glm_shard::top_k<glm_shard::MERGE>(keys, (int64_t)world * K, K, false, [&](int64_t c, int slot, uint32_t) {
    const long long p = pair(c)[K];
    for (int t = 0; t < 4; ++t) out[4 * slot + t] = glm_shard::token_at(pt, page_rows, p * 4 + t);
  });
  if (threadIdx.x < 3) {  // the incomplete pool's visible tokens, as tf_glm_dsa_tokens writes them
    const long long t = npool * 4 + threadIdx.x;
    out[4 * K + threadIdx.x] = t <= q ? glm_shard::token_at(pt, page_rows, t) : -1;
  }
  if (threadIdx.x == 0) {
    int tail = 0;
    for (int t = 0; t < 3; ++t) tail += npool * 4 + t <= q;
    CNT[r] = npool > K ? 4 * K + tail : 0;
  }
}
