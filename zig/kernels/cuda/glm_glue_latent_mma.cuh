// TF_GLM_OVN_LATENT_MMA (a numerics trade): prompt rows' absorb and expand on the tensor cores, production's
// _absorb_mma, _expand_mma, _absorb_mma8 and _expand_mma8 (prod-src families/glm5_next/cuda/latent.py:237-313).
#pragma once

#include "glm_glue_bmm.cuh"
#include "glm_glue_latent_ovn.cuh"

namespace glm_latent_mma {

// latent.MMA_TILE (64 rows, 128 columns, k 64) and MMA_LAUNCH (4 warps, 3 stages): fixed, so no chunk moves a bit.
constexpr int BM = 64, BN = 128, BK = 64, STAGES = 3, THREADS = 128;
constexpr int A_BYTES = BM * BK * 2, B_BYTES = BN * BK * 2, STAGE = A_BYTES + B_BYTES, SMEM = STAGES * STAGE;
constexpr int D = 256, LW = 512, KB = glm_latent_ovn::KB;  // head width (qk and v), latent, E4M3 scale block

__device__ __forceinline__ void ldm4t(uint32_t* r, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// 8 E4M3 codes (low byte first) as 8 bf16, two a word, low half first: bf16(fp32(code) x s), exact for s = 1.
__device__ __forceinline__ uint4 bf8_of(uint2 c, float s) {
  uint32_t out[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint16_t bits = (uint16_t)(((k < 2 ? c.x : c.y) >> (16 * (k & 1))) & 0xffffu);
    const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
    const __nv_bfloat162 b = __floats2bfloat162_rn(__fmul_rn(v.x, s), __fmul_rn(v.y, s));
    out[k] = *reinterpret_cast<const uint32_t*>(&b);
  }
  return make_uint4(out[0], out[1], out[2], out[3]);
}

// ABSORB: out [rows][heads][512] = x [rows][heads][256] . w [heads][256][512] (k rows of n columns). Expand:
// out [rows][heads][256] = x [rows][heads][512] . w [heads][256][512]^T (n rows of k). Block (128 columns, head, 64
// rows): production's tiles, its grid reordered so a row block's q / o is read once (order sets no bit); 4 warps of
// 32 x 64; each output one fp32 mma chain from zero over ascending 16-wide k, as tl.dot's.
// F8: w E4M3 codes; absorb scales s [heads][256][4] (a k row's block), expand s [heads][256][4] (an output's).
template <bool ABSORB, bool F8>
__device__ void run(const __nv_bfloat16* x, const void* w, const float* s, __nv_bfloat16* out, int64_t rows,
                    int64_t heads) {
  constexpr int K = ABSORB ? D : LW, N = ABSORB ? LW : D, STEPS = K / BK, MT = 2, NT = 8;
  constexpr int BI = 8;                                // B's 16-byte chunks a thread moves a step (A's: 4)
  constexpr int BW = ABSORB ? BN : BK;                 // B's shared row: absorb 64 k rows of 128, expand 128 of 64
  constexpr int BSTEP = ABSORB ? 8 : 16;               // rows between a thread's B chunks
  constexpr int64_t KSTEP = ABSORB ? (int64_t)BK * LW : BK;  // a k step along w's elements
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int64_t h = blockIdx.y, bm0 = (int64_t)blockIdx.z * BM, bn0 = (int64_t)blockIdx.x * BN;
  const int wm = (warp >> 1) * 32, wn = (warp & 1) * 64;
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  // A: rows ar + 16 i, 16-byte column ac, stored at column ac ^ (row & 7) (rows past `rows`: zeros, never stored)
  const int ar = tid >> 3, ac = tid & 7;
  const __nv_bfloat16* ga[4];
  unsigned oka = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t row = bm0 + ar + 16 * i;
    oka |= (row < rows ? 1u : 0u) << i;
    ga[i] = x + ((row < rows ? row : 0) * heads + h) * K + ac * 8;
  }
  const unsigned sa_off = (ar * BK + ((ac ^ (ar & 7)) * 8)) * 2;
  // B: rows br + BSTEP i, 16-byte column bc, stored at bc ^ (row & 7); gb: chunk 0's element in w at k step 0
  const int br = ABSORB ? tid >> 4 : tid >> 3, bc = ABSORB ? tid & 15 : tid & 7;
  const unsigned sb_off = A_BYTES + (br * BW + ((bc ^ (br & 7)) * 8)) * 2;
  const int64_t gb = ABSORB ? (h * D + br) * LW + bn0 + bc * 8 : (h * D + bn0 + br) * LW + bc * 8;
  auto load = [&](int st, int kt) {
    const unsigned base = sbase + st * STAGE;
#pragma unroll
    for (int i = 0; i < 4; ++i) glm_bmm::cp16s(base + sa_off + i * 16 * BK * 2, ga[i] + kt * BK, (oka >> i) & 1);
    if constexpr (!F8) {
      const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w);
#pragma unroll
      for (int i = 0; i < BI; ++i)
        glm_bmm::cp16s(base + sb_off + i * BSTEP * BW * 2, wb + gb + (int64_t)i * BSTEP * LW + kt * KSTEP, true);
    }
  };
  // F8: a step's codes (and absorb's scales) into registers ahead, converted into shared after the step before
  uint2 codes[F8 ? BI : 1];
  float scl[F8 && ABSORB ? BI : 1];
  auto fetch = [&](int kt) {
    if constexpr (F8) {
      const unsigned char* wc = static_cast<const unsigned char*>(w);
#pragma unroll
      for (int i = 0; i < BI; ++i) {
        const int64_t at = gb + (int64_t)i * BSTEP * LW + kt * KSTEP;
        codes[i] = __ldg(reinterpret_cast<const uint2*>(wc + at));
        if constexpr (ABSORB) scl[i] = __ldg(s + (at / LW) * (LW / KB) + bn0 / KB);
      }
    }
  };
  auto put = [&](int st) {
    if constexpr (F8) {
#pragma unroll
      for (int i = 0; i < BI; ++i)
        *reinterpret_cast<uint4*>(smem + st * STAGE + sb_off + i * BSTEP * BW * 2) =
            bf8_of(codes[i], ABSORB ? scl[ABSORB ? i : 0] : 1.0f);
    }
  };
  // ldmatrix rows: A's (lane & 15, chunk + 1 for lanes 16 on); expand's B rows as glm_bmm::bmm_tiles; absorb's B
  // transposed: k rows (lane & 7) + 8 for lanes 8-15 and 24-31, n chunk + 1 for lanes 16 on.
  const int ra = wm + (lane & 15), rb = wn + (lane >> 4) * 8 + (lane & 7);
  const unsigned fa = ra * BK * 2, fb = A_BYTES + rb * BK * 2;
  const unsigned ft = A_BYTES + ((lane & 7) + ((lane >> 3) & 1) * 8) * BN * 2;
  unsigned ca[BK / 16], cb[BK / 16], ct[NT / 2];
#pragma unroll
  for (int s16 = 0; s16 < BK / 16; ++s16) {
    ca[s16] = ((2 * s16 + (lane >> 4)) ^ (ra & 7)) * 16;
    cb[s16] = ((2 * s16 + ((lane >> 3) & 1)) ^ (rb & 7)) * 16;
  }
#pragma unroll
  for (int jp = 0; jp < NT / 2; ++jp) ct[jp] = ((wn / 8 + 2 * jp + (lane >> 4)) ^ (lane & 7)) * 16;
  float c[MT][NT][4] = {};
#pragma unroll
  for (int s0 = 0; s0 < STAGES - 1; ++s0) {
    load(s0, s0);
    glm_bmm::commit();
    fetch(s0);
    put(s0);
  }
  int rs = 0, ws = STAGES - 1;
  for (int kt = 0; kt < STEPS; ++kt) {
    glm_bmm::wait<STAGES - 2>();
    __syncthreads();
    const bool more = kt + STAGES - 1 < STEPS;
    if (more) {
      load(ws, kt + STAGES - 1);
      fetch(kt + STAGES - 1);
    }
    glm_bmm::commit();
    const unsigned st = sbase + rs * STAGE;
    float p[F8 && !ABSORB ? MT : 1][NT][4] = {};  // expand8: the step's own dot, from zero
#pragma unroll
    for (int s16 = 0; s16 < BK / 16; ++s16) {
      uint32_t a[MT][4], b[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) glm_bmm::ldm4s(a[i], st + fa + i * 16 * BK * 2 + ca[s16]);
#pragma unroll
      for (int jp = 0; jp < NT / 2; ++jp) {
        uint32_t r[4];
        if constexpr (ABSORB)
          ldm4t(r, st + ft + s16 * 16 * BN * 2 + ct[jp]);
        else
          glm_bmm::ldm4s(r, st + fb + jp * 16 * BK * 2 + cb[s16]);
        b[2 * jp][0] = r[0], b[2 * jp][1] = r[1], b[2 * jp + 1][0] = r[2], b[2 * jp + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          if constexpr (F8 && !ABSORB)
            glm_router::mma16816(p[i][j], a[i], b[j]);
          else
            glm_router::mma16816(c[i][j], a[i], b[j]);
        }
    }
    if constexpr (F8 && !ABSORB) {
      // _expand_mma8: acc + dot(o, w^T) x s, one scale an output and 128-column block, contracted as Triton fuses
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int64_t col = bn0 + wn + j * 8 + q * 2 + e;
          const float sv = __ldg(s + (h * D + col) * (LW / KB) + kt * BK / KB);
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            c[i][j][e] = __fmaf_rn(p[i][j][e], sv, c[i][j][e]);
            c[i][j][e + 2] = __fmaf_rn(p[i][j][e + 2], sv, c[i][j][e + 2]);
          }
        }
    }
    if (more) put(ws);
    ws = ws + 1 == STAGES ? 0 : ws + 1;
    rs = rs + 1 == STAGES ? 0 : rs + 1;
  }
  glm_bmm::wait<0>();
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int64_t row = bm0 + wm + i * 16 + g + hh * 8, col = bn0 + wn + j * 8 + q * 2;
        if (row < rows)
          *reinterpret_cast<__nv_bfloat162*>(out + (row * heads + h) * N + col) =
              __floats2bfloat162_rn(c[i][j][2 * hh], c[i][j][2 * hh + 1]);
      }
}

}  // namespace glm_latent_mma

// latent._absorb_mma: qa = q . wk on the tensor cores (bf16 kv_b), grid (512 / 128, heads, rows / 64).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_mma(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                      int64_t heads) {
  glm_latent_mma::run<true, false>(q, wk, nullptr, qa, rows, heads);
}

// latent._expand_mma: out = o_lat . wv^T on the tensor cores (bf16 kv_b), grid (256 / 128, heads, rows / 64).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_mma(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                      int64_t heads) {
  glm_latent_mma::run<false, false>(ol, wv, nullptr, out, rows, heads);
}

// latent._absorb_mma8 (TF_GLM_OVN_KVB8's kv_b): each weight bf16(code x its k row's scale), then _absorb_mma's chain.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_mma8(const __nv_bfloat16* q, const unsigned char* wk, const float* sk, __nv_bfloat16* qa,
                       int64_t rows, int64_t heads) {
  glm_latent_mma::run<true, true>(q, wk, sk, qa, rows, heads);
}

// latent._expand_mma8 (TF_GLM_OVN_KVB8's kv_b): each k step's dot on the exact codes, times its scale, added.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_mma8(const __nv_bfloat16* ol, const unsigned char* wv, const float* sv, __nv_bfloat16* out,
                       int64_t rows, int64_t heads) {
  glm_latent_mma::run<false, true>(ol, wv, sv, out, rows, heads);
}
