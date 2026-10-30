// TF_GLM_OVN_SEG_PASS: OTS 0108's warp-specialised chunk pass computing glm_attn::chunks_wide's partials, bit for bit.
#pragma once

#include "glm_glue_attn_ovn.cuh"

namespace glm_seg_pass {  // a CTA: one row, a head block of 8 NQ heads, one chunk; tiles past its last key skipped

using bf16 = __nv_bfloat16;
constexpr int LW = glm_attn::LW, KT = glm_attn::KT, CH = glm_attn::CH;  // latent width, keys a tile, keys a chunk
constexpr int U = LW * 2 / 16;                                            // 16-byte units of a bf16 key row (64)
constexpr int NB = 3, NV = 5, NPW = 4;                                   // key-tile buffers, PV and producer warps
constexpr int BAR_FULL = 1, BAR_EMPTY = 4, BAR_PREADY = 7, BAR_PFREE = 8;
constexpr int PV_MAX = 7;                                                // the most latent m16 tiles a PV warp takes
constexpr int NT2 = (NV + 2 + NPW) * 32, NT3 = (NV + 3 + NPW) * 32;     // threads of 16- and 24-head blocks

// PV warp v's share of the 32 latent m16 tiles (production's 6 / 6 / 6 / 7 / 7)
__device__ __forceinline__ constexpr int pv_tiles(int v) { return v < 3 ? 6 : 7; }

template <int NQ>
struct Smem {
  uint4 kt[NB][KT * U];      // key tiles, bf16 [key][64 units], swizzled
  uint16_t ps[8 * NQ * KT];  // P [head][key] bf16, 4 units a row, unit-swizzled
  float alpha[8 * NQ];
};
static_assert(sizeof(Smem<2>) == 99392 && sizeof(Smem<3>) == 99936, "cuda_glue.zig's seg_smem");

// chunks_wide's products in its k order, transposed: keys (QK) or latent columns (P.V) as A, q^T or P^T as B
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm(uint32_t (&r)[4], unsigned s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void ldsm_t(uint32_t (&r)[4], unsigned s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// a key tile's 16-byte unit c of key r (production's swizzle: ldmatrix's eight rows and the producers' stores apart)
__device__ __forceinline__ int swz(int r, int c) { return c ^ ((r ^ (c >> 3)) & 7); }
__device__ __forceinline__ int pswz(int r, int c) { return c ^ ((r >> 1) & 3); }

// chunks_wide's max over a key group (lane g: its key g): quad_max's tree, operands in its lane-0 order, from g = 0
__device__ __forceinline__ float group_max(float a, int g, int t4) {
  const float b = __shfl_xor_sync(0xffffffffu, a, 4);
  const float v = (g & 1) ? fmaxf(b, a) : fmaxf(a, b);
  const float x = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  const float y = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
  return __shfl_sync(0xffffffffu, y, t4);
}

// chunks_wide's p sum over key group kg: (u0 + u2) + (u1 + u3), u_q = key 2q + key 2q + 1 (adds commute: any lane)
__device__ __forceinline__ float group_sum(float p) {
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 4));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 16));
  return __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 8));
}

struct Raw {  // an E4M3 producer's tile share: 16 codes a key a lane, and the key's scale
  uint4 v[8];
  float s[8];
};

// chunks_wide's partials of chunk blockIdx.z for row blockIdx.x and heads 8 NQ blockIdx.y .., its arguments as is
template <int NQ, bool FP8>
__device__ __forceinline__ void pass(const bf16* __restrict__ qa, const unsigned char* __restrict__ lc,
                                     const float* __restrict__ scales, const int* pos, const int* tok, const int* cnt,
                                     float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                                     float scale, int sparse, const int* pt, int64_t pstep) {
  constexpr int N_FULL = (NPW + NQ) * 32, N_EMPTY = (NV + NPW) * 32, N_QV = (NQ + NV) * 32;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<NQ>& S = *reinterpret_cast<Smem<NQ>*>(smem_raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int64_t r = blockIdx.x, h0 = (int64_t)blockIdx.y * 8 * NQ, c = blockIdx.z;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r, start = c * CH;
  const bool live = sparse ? start < n : start <= limit;
  const int ntile = live ? glm_attn_ovn::live_tiles(start, n, limit, sparse) : 0;
  const bool pad = live && ntile < CH / KT;           // chunks_wide's masked tiles past the last: their +0, once
  const int64_t base = (c * rows + r) * heads + h0;  // the block's first head's partial
  auto valid = [&](int64_t ki) { return sparse ? ki < n : ki <= limit; };

  if (warp >= 5 + NQ) {
    // producers: keys pw + 4 i of each tile, a lane's 16 latent columns 16 lane .., two tiles ahead
    const int pw = warp - 5 - NQ;
    if constexpr (FP8) {
      const uint32_t page_rows = pt ? (uint32_t)(pstep / LW) : 1u;
      auto ids = [&](int tile, int (&id)[8]) {  // a key's slab or page row (chunks_wide's src), -1: masked
        const int64_t first = start + (int64_t)tile * KT;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int64_t ki = first + pw + 4 * i;
          id[i] = tile < ntile && valid(ki) ? (sparse ? tok[r * w + ki] : (int)ki) : -1;
        }
      };
      auto load = [&](const int (&id)[8], Raw& x) {  // chunks_wide's fetch: codes and scale on a slab or pages
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const bool ok = id[i] >= 0;
          const int64_t k = ok ? id[i] : 0;
          const unsigned char* codes = lc + k * LW;
          const float* sp = scales + k;
          if (pt && ok) {
            const uint32_t off = sparse ? (uint32_t)k % page_rows : (uint32_t)(k - start);
            const unsigned char* page = sparse ? codes - (int64_t)off * LW : lc + (int64_t)pt[c] * pstep;
            codes = page + (int64_t)off * LW;
            sp = reinterpret_cast<const float*>(page + CH * LW) + off;
          }
          x.v[i] = ok ? *reinterpret_cast<const uint4*>(codes + 16 * lane) : make_uint4(0u, 0u, 0u, 0u);
          x.s[i] = ok ? *sp : 1.0f;
        }
      };
      auto convert = [&](const Raw& x, uint4* dst) {  // chunks_wide's store: bf16(code * scale) a value
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int key = pw + 4 * i;
          const uint32_t in[4] = {x.v[i].x, x.v[i].y, x.v[i].z, x.v[i].w};
          uint32_t o[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {  // code pairs (2j, 2j + 1), little-endian
            const uint16_t bits = (uint16_t)((in[j >> 1] >> (16 * (j & 1))) & 0xffffu);
            const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
            const __nv_bfloat162 b = __floats2bfloat162_rn(__fmul_rn(v.x, x.s[i]), __fmul_rn(v.y, x.s[i]));
            o[j] = *reinterpret_cast<const uint32_t*>(&b);
          }
          dst[key * U + swz(key, 2 * lane)] = make_uint4(o[0], o[1], o[2], o[3]);
          dst[key * U + swz(key, 2 * lane + 1)] = make_uint4(o[4], o[5], o[6], o[7]);
        }
      };
      Raw A, B;
      int ia[8], ib[8];
      ids(0, ia);
      ids(1, ib);
      load(ia, A);
      load(ib, B);
      ids(2, ia);
      ids(3, ib);
      for (int t = 0; t < ntile; t += 2) {
        if (t >= NB) bar_sync(BAR_EMPTY + t % NB, N_EMPTY);
        convert(A, S.kt[t % NB]);
        bar_arrive(BAR_FULL + t % NB, N_FULL);
        if (t + 2 < ntile) {
          load(ia, A);
          ids(t + 4, ia);
        }
        if (t + 1 >= ntile) break;
        if (t + 1 >= NB) bar_sync(BAR_EMPTY + (t + 1) % NB, N_EMPTY);
        convert(B, S.kt[(t + 1) % NB]);
        bar_arrive(BAR_FULL + (t + 1) % NB, N_FULL);
        if (t + 3 < ntile) {
          load(ib, B);
          ids(t + 5, ib);
        }
      }
    } else {
      // BF16 rows by cp.async (masked keys zero-filled, as chunks_wide's load_kv), each tile its own group
      const bf16* keys_at = reinterpret_cast<const bf16*>(lc);
      if (pt && !sparse && live) keys_at += (int64_t)pt[c] * (pstep / 2) - start * LW;
      auto issue = [&](int tile) {
        uint4* dst = S.kt[tile % NB];
        const int64_t first = start + (int64_t)tile * KT;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int key = pw + 4 * i;
          const int64_t ki = first + key;
          const bool ok = valid(ki);
          const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
#pragma unroll
          for (int h = 0; h < 2; ++h)
            glm_attn::cp16(dst + key * U + swz(key, 2 * lane + h), keys_at + src * LW + (2 * lane + h) * 8, ok);
        }
      };
      if (ntile > 0) issue(0);
      glm_attn::commit();
      if (ntile > 1) issue(1);
      glm_attn::commit();
      for (int t = 0; t < ntile; ++t) {
        glm_attn::wait<1>();  // tile t's group landed (t + 1's may still fly)
        bar_arrive(BAR_FULL + t % NB, N_FULL);
        if (t + 2 < ntile) {
          if (t + 2 >= NB) bar_sync(BAR_EMPTY + (t + 2) % NB, N_EMPTY);
          issue(t + 2);
        }
        glm_attn::commit();
      }
    }
    return;
  }

  if (warp >= 4 && warp < 4 + NQ) {
    // QK warps: local heads 8 j .. 8 j + 7 (an n8 tile), all 32 keys of a tile, the online softmax
    const int j = warp - 4;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};  // local heads 8 j + 2 t4, + 1
    if (ntile > 0) {
      const int64_t hq = h0 + 8 * j + g;  // this lane's B column (head); heads past the last read as zeros
      const bool hok = hq < heads;
      const bf16* q = qa + (r * heads + (hok ? hq : 0)) * LW + 2 * t4;
      uint32_t qf[32][2];  // q^T B fragments: step s, k = 16 s + 2 t4 (+ 8)
#pragma unroll
      for (int s = 0; s < 32; ++s) {
        qf[s][0] = hok ? *reinterpret_cast<const uint32_t*>(q + 16 * s) : 0u;
        qf[s][1] = hok ? *reinterpret_cast<const uint32_t*>(q + 16 * s + 8) : 0u;
      }
      const int ar = lane & 15, ac = lane >> 4;
      uint32_t a[3][2][4];
      unsigned ktn;
      float acc[2][4], accn[2][4];
#define SEG_LOADA(slot, s)                                                          \
  {                                                                                 \
    ldsm(a[slot][0], ktn + 16 * (ar * U + swz(ar, 2 * (s) + ac)));                  \
    ldsm(a[slot][1], ktn + 16 * ((16 + ar) * U + swz(16 + ar, 2 * (s) + ac)));      \
  }
#define SEG_QKSTEP(s)                                                               \
  {                                                                                 \
    if ((s) + 2 < 32) SEG_LOADA(((s) + 2) % 3, (s) + 2)                             \
    mma(accn[0], a[(s) % 3][0], qf[s][0], qf[s][1]);                                \
    mma(accn[1], a[(s) % 3][1], qf[s][0], qf[s][1]);                                \
  }
#define SEG_QKBEGIN()                                                               \
  {                                                                                 \
    for (int i_ = 0; i_ < 2; ++i_)                                                  \
      for (int e_ = 0; e_ < 4; ++e_) accn[i_][e_] = 0.0f;                           \
    SEG_LOADA(0, 0)                                                                 \
    SEG_LOADA(1, 1)                                                                 \
  }
      bar_sync(BAR_FULL + 0, N_FULL);
      ktn = (unsigned)__cvta_generic_to_shared(S.kt[0]);
      SEG_QKBEGIN()
#pragma unroll
      for (int s = 0; s < 32; ++s) SEG_QKSTEP(s)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][e] = accn[i][e];
      for (int t = 0; t < ntile; ++t) {
        const int b = t % NB, bn = (t + 1) % NB;
        const int64_t first = start + (int64_t)t * KT;
        if (t + 1 < ntile) bar_sync(BAR_FULL + bn, N_FULL);
        // the last tile's "next" scores run on its own buffer and are dropped
        ktn = (unsigned)__cvta_generic_to_shared(S.kt[t + 1 < ntile ? bn : b]);
        SEG_QKBEGIN()
#pragma unroll
        for (int s = 0; s < 8; ++s) SEG_QKSTEP(s)
        // acc[i][e]: key 16 i + g (e < 2) or 16 i + 8 + g (e >= 2) of the tile, local head 8 j + 2 t4 + (e & 1)
        float sc[2][4];
        bool val[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int hi = 0; hi < 2; ++hi) {
            val[i][hi] = valid(first + 16 * i + 8 * hi + g);
#pragma unroll
            for (int e = 0; e < 2; ++e)
              sc[i][2 * hi + e] = val[i][hi] ? __fmul_rn(acc[i][2 * hi + e], scale) : -INFINITY;
          }
        float nm[2], alpha[2];
        bool act[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {  // key group kg = 2 i + hi
          const float m0 = group_max(sc[0][e], g, t4), m1 = group_max(sc[0][2 + e], g, t4);
          const float m2 = group_max(sc[1][e], g, t4), m3 = group_max(sc[1][2 + e], g, t4);
          const float tm = fmaxf(fmaxf(m0, m2), fmaxf(m1, m3));
          act[e] = tm != -INFINITY;
          nm[e] = act[e] ? fmaxf(m[e], tm) : m[e];
          alpha[e] = act[e] ? (m[e] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[e] - nm[e])) : 1.0f;
        }
#pragma unroll
        for (int s = 8; s < 16; ++s) SEG_QKSTEP(s)
        float p[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            const int e = x & 1;
            p[i][x] = val[i][x >> 1] && act[e] ? glm_triton::exp_approx(sc[i][x] - nm[e]) : 0.0f;
          }
#pragma unroll
        for (int s = 16; s < 24; ++s) SEG_QKSTEP(s)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float s0 = group_sum(p[0][e]), s1 = group_sum(p[0][2 + e]);
          const float s2 = group_sum(p[1][e]), s3 = group_sum(p[1][2 + e]);
          l[e] = __fmaf_rn(l[e], alpha[e], __fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3)));
          m[e] = nm[e];
        }
#pragma unroll
        for (int s = 24; s < 32; ++s) SEG_QKSTEP(s)
        if (t > 0) bar_sync(BAR_PFREE, N_QV);  // the PV warps have read the previous P and alphas
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            const int h = 8 * j + 2 * t4 + (x & 1), key = 16 * i + 8 * (x >> 1) + g;
            const __nv_bfloat16 v = __float2bfloat16_rn(p[i][x]);
            S.ps[h * KT + 8 * pswz(h, key >> 3) + (key & 7)] = *reinterpret_cast<const uint16_t*>(&v);
          }
        if (g == 0) {
          S.alpha[8 * j + 2 * t4] = alpha[0];
          S.alpha[8 * j + 2 * t4 + 1] = alpha[1];
        }
        bar_arrive(BAR_PREADY, N_QV);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[i][e] = accn[i][e];
      }
#undef SEG_QKSTEP
#undef SEG_QKBEGIN
#undef SEG_LOADA
    }
    if (pad)
      for (int e = 0; e < 2; ++e) l[e] = __fmaf_rn(l[e], 1.0f, 0.0f);
    if (g == 0)
      for (int e = 0; e < 2; ++e) {
        const int hl = 8 * j + 2 * t4 + e;
        if (h0 + hl >= heads) continue;
        pm[base + hl] = m[e];
        pl[base + hl] = l[e];
      }
    return;
  }

  // PV warps (0-3 and 4 + NQ): latent rows 16 tb .. 16 (tb + nt) - 1, every head of the block
  const int v = warp < 4 ? warp : 4;
  const int nt = pv_tiles(v);
  int tb = 0;
#pragma unroll
  for (int k = 0; k < NV - 1; ++k) tb += k < v ? pv_tiles(k) : 0;
  float o[PV_MAX][NQ][4];
#pragma unroll
  for (int i = 0; i < PV_MAX; ++i)
#pragma unroll
    for (int nh = 0; nh < NQ; ++nh)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[i][nh][e] = 0.0f;
  const unsigned psb = (unsigned)__cvta_generic_to_shared(S.ps);
  const int vk = (lane & 7) + 8 * (lane >> 4), vc = (lane >> 3) & 1;
  for (int t = 0; t < ntile; ++t) {
    const int b = t % NB;
    const unsigned ktb = (unsigned)__cvta_generic_to_shared(S.kt[b]);
    bar_sync(BAR_PREADY, N_QV);
    uint32_t pb[NQ][4];  // P^T B fragments: heads 8 nh .., keys 0-15 (0, 1), 16-31 (2, 3)
#pragma unroll
    for (int nh = 0; nh < NQ; ++nh) {
      const int h = 8 * nh + (lane & 7), u = lane >> 3;
      ldsm(pb[nh], psb + 2 * (h * KT + 8 * pswz(h, u)));
    }
    float al[NQ][2];
#pragma unroll
    for (int nh = 0; nh < NQ; ++nh) {
      al[nh][0] = S.alpha[8 * nh + 2 * t4];
      al[nh][1] = S.alpha[8 * nh + 2 * t4 + 1];
    }
    if (t + 1 < ntile) bar_arrive(BAR_PFREE, N_QV);  // the QK warps write the next P only after this
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i)
      if (i < nt)
#pragma unroll
        for (int nh = 0; nh < NQ; ++nh)
#pragma unroll
          for (int e = 0; e < 4; ++e) o[i][nh][e] = __fmul_rn(o[i][nh][e], al[nh][e & 1]);
    uint32_t a[2][2][4];  // K^T A fragments (keys 0-15, 16-31), two m16 tiles in flight
#define SEG_LOADV(slot, i)                                                          \
  {                                                                                 \
    ldsm_t(a[slot][0], ktb + 16 * (vk * U + swz(vk, 2 * (tb + (i)) + vc)));         \
    ldsm_t(a[slot][1], ktb + 16 * ((16 + vk) * U + swz(16 + vk, 2 * (tb + (i)) + vc))); \
  }
    SEG_LOADV(0, 0)
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i) {
      if (i < nt) {
        if (i + 1 < nt) SEG_LOADV((i + 1) & 1, i + 1)
#pragma unroll
        for (int nh = 0; nh < NQ; ++nh) mma(o[i][nh], a[i & 1][0], pb[nh][0], pb[nh][1]);
#pragma unroll
        for (int nh = 0; nh < NQ; ++nh) mma(o[i][nh], a[i & 1][1], pb[nh][2], pb[nh][3]);
      }
    }
#undef SEG_LOADV
    if (t + NB < ntile) bar_arrive(BAR_EMPTY + b, N_EMPTY);
  }
  if (pad) {  // the masked tile: o * 1, then its two k16 steps of zero P and zero keys
    const uint32_t z[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i)
      if (i < nt)
#pragma unroll
        for (int nh = 0; nh < NQ; ++nh) {
#pragma unroll
          for (int e = 0; e < 4; ++e) o[i][nh][e] = __fmul_rn(o[i][nh][e], 1.0f);
          mma(o[i][nh], z, 0u, 0u);
          mma(o[i][nh], z, 0u, 0u);
        }
  }
  float* out = po + base * LW;
#pragma unroll
  for (int i = 0; i < PV_MAX; ++i)
    if (i < nt)
#pragma unroll
      for (int nh = 0; nh < NQ; ++nh)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int hl = 8 * nh + 2 * t4 + (e & 1), lat = 16 * (tb + i) + g + 8 * (e >> 1);
          if (h0 + hl < heads) out[(int64_t)hl * LW + lat] = o[i][nh][e];
        }
}

}  // namespace glm_seg_pass

// tf_glm_chunks_16 / _fp8's arguments and partials: 16-head blocks (seg16, any head count) and 24-head (seg24, 17-24)
extern "C" __global__ void __launch_bounds__(glm_seg_pass::NT2, 1)
    tf_glm_chunks_seg16(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                        const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                        float scale, int sparse, const int* pt, int64_t pstep) {
  glm_seg_pass::pass<2, false>(qa, reinterpret_cast<const unsigned char*>(lc), nullptr, pos, tok, cnt, po, pm, pl,
                               rows, heads, w, scale, sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(glm_seg_pass::NT3, 1)
    tf_glm_chunks_seg24(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                        const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                        float scale, int sparse, const int* pt, int64_t pstep) {
  glm_seg_pass::pass<3, false>(qa, reinterpret_cast<const unsigned char*>(lc), nullptr, pos, tok, cnt, po, pm, pl,
                               rows, heads, w, scale, sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(glm_seg_pass::NT2, 1)
    tf_glm_chunks_seg16_fp8(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                            const int* tok, const int* cnt, float* po, float* pm, float* pl, int64_t rows,
                            int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  glm_seg_pass::pass<2, true>(qa, lc, scales, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}
extern "C" __global__ void __launch_bounds__(glm_seg_pass::NT3, 1)
    tf_glm_chunks_seg24_fp8(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                            const int* tok, const int* cnt, float* po, float* pm, float* pl, int64_t rows,
                            int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  glm_seg_pass::pass<3, true>(qa, lc, scales, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}
