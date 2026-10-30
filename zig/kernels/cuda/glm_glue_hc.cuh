// glue._hc_partial and glue._hc_finish by hand (hyper-connections' pre side, 4 streams of 4096).
#pragma once

#include "glm_glue_rows.cuh"
#include "common/qmm_frag.cuh"
#include "glm_triton.cuh"

using namespace glm_triton;

// glue._hc_partial, block (4 rows, K block b) of 128 threads: a row's 24 dots and squares in Triton's fma order.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_hc_partial(
            const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {
  constexpr int R = 4;
  __shared__ float smem[R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R, b = blockIdx.y, kb = wide / nb;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, c = t & 15, m0 = (t >> 4) & 7;
  float acc[R][3] = {}, ss[R] = {};
  for (int64_t step = 0; step < kb / 128; ++step) {
    const int64_t base = b * kb + step * 128;
    float xv[R][8];
    for (int q = 0; q < R; ++q) {  // the 8 inputs in one 16-byte load (wide and kb are multiples of 128: aligned)
      uint4 u = make_uint4(0, 0, 0, 0);
      if (r0 + q < rows) u = *reinterpret_cast<const uint4*>(x + (r0 + q) * wide + base + c * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[q][j] = bf(h[j]);
    }
    for (int i = 0; i < 3; ++i) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
      const __nv_bfloat16* hw = reinterpret_cast<const __nv_bfloat16*>(&u);
      float w[8];
      for (int j = 0; j < 8; ++j) w[j] = bf(hw[j]);
      for (int q = 0; q < R; ++q) {
        float d = __fmul_rn(xv[q][1], w[1]);
        d = __fmaf_rn(xv[q][0], w[0], d);
        for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[q][j], w[j], d);
        for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
        acc[q][i] = acc[q][i] + d;
      }
    }
    for (int q = 0; q < R; ++q) {
      const float xt = r0 + q < rows ? bf(x[(r0 + q) * wide + base + t]) : 0.0f;
      ss[q] = __fmaf_rn(xt, xt, ss[q]);
    }
  }
  for (int q = 0; q < R; ++q) {
    for (int o = 16; o > 0; o >>= 1) ss[q] += __shfl_xor_sync(0xffffffffu, ss[q], o);
    if (lane == 0) smem[q][warp] = ss[q];
  }
  __syncthreads();
  for (int q = 0; q < R; ++q) {
    if (r0 + q >= rows) break;
    float* out = part + ((r0 + q) * nb + b) * 32;
    if (c == 0)
      for (int i = 0; i < 3; ++i) out[m0 + 8 * i] = acc[q][i];
    if (t == 0) out[24] = (smem[q][0] + smem[q][2]) + (smem[q][1] + smem[q][3]);
  }
}

namespace glm_hc {

// The same for decode windows (16 blocks a row): 8 steps of 128 at once, a 128-thread group each; same ops, same order.
template <bool PDL>
__device__ __forceinline__ void partial8(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide,
                                         int nb) {
  __shared__ float dots[8][24];
  __shared__ float smem[4];
  const int64_t r = blockIdx.x, b = blockIdx.y, kb = wide / nb;
  const int grp = threadIdx.x >> 7, t = threadIdx.x & 127, c = t & 15, m0 = (t >> 4) & 7;
  const int steps = static_cast<int>(kb / 128);
  [[maybe_unused]] uint4 wq[3];
  if constexpr (PDL) {  // TF_GLM_OVN_PDL_EARLY, launched dependent: the weights load first, x after the wait
    if (grp < steps)
      for (int i = 0; i < 3; ++i)
        wq[i] = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + b * kb + grp * 128 + c * 8));
    qmm_frag::grid_wait();
  }
  if (grp < steps) {
    const int64_t base = b * kb + grp * 128;
    float xv[8];
    {
      const uint4 u = *reinterpret_cast<const uint4*>(x + r * wide + base + c * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[j] = bf(h[j]);
    }
    for (int i = 0; i < 3; ++i) {
      uint4 u;
      if constexpr (PDL)
        u = wq[i];
      else
        u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
      const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&u);
      float d = __fmul_rn(xv[1], bf(w[1]));
      d = __fmaf_rn(xv[0], bf(w[0]), d);
      for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[j], bf(w[j]), d);
      for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
      if (c == 0) dots[grp][m0 + 8 * i] = d;
    }
  }
  if (grp == 0) {
    float ss = 0.0f;
    for (int s = 0; s < steps; ++s) {
      const float xt = bf(x[r * wide + b * kb + s * 128 + t]);
      ss = __fmaf_rn(xt, xt, ss);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) smem[t >> 5] = ss;
  }
  __syncthreads();
  float* out = part + (r * nb + b) * 32;
  if (threadIdx.x < 24) {
    float acc = 0.0f;
    for (int s = 0; s < steps; ++s) acc = acc + dots[s][threadIdx.x];
    out[threadIdx.x] = acc;
  } else if (threadIdx.x == 32) {
    out[24] = (smem[0] + smem[2]) + (smem[1] + smem[3]);
  }
}

// The Sinkhorn sums of 4: lanes (i, j) = (lane / 4 % 4, lane % 4), rows over j (xor 2, 1), columns over i (xor 8, 4).
__device__ __forceinline__ float row4(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float col4(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 8);
  return v + __shfl_xor_sync(0xffffffffu, v, 4);
}
__device__ __forceinline__ float sigmoid(float v) { return div_full(1.0f, exp_approx(0.0f - v) + 1.0f); }

// A thread's 8 norm weights of each half, loaded with its inputs: bf16 in one 16-byte load, float32 in two.
template <typename W>
struct Weights;
template <>
struct Weights<__nv_bfloat16> {
  uint4 u[2];
  __device__ __forceinline__ void load(const __nv_bfloat16* nw, int rep, int t) {
    u[rep] = __ldg(reinterpret_cast<const uint4*>(nw + rep * 2048 + t * 8));
  }
  __device__ __forceinline__ float at(int rep, int j) const {
    return bf(reinterpret_cast<const __nv_bfloat16*>(&u[rep])[j]);
  }
};
template <>
struct Weights<float> {
  float4 u[2][2];
  __device__ __forceinline__ void load(const float* nw, int rep, int t) {
    u[rep][0] = __ldg(reinterpret_cast<const float4*>(nw + rep * 2048 + t * 8));
    u[rep][1] = __ldg(reinterpret_cast<const float4*>(nw + rep * 2048 + t * 8 + 4));
  }
  __device__ __forceinline__ float at(int rep, int j) const {
    return reinterpret_cast<const float*>(&u[rep][0])[j];
  }
};

// A load of hc_finish's rows or partials: plain, or (CG: other blocks of the same kernel wrote them) from L2.
template <bool CG, typename T>
__device__ __forceinline__ T ldr(const T* p) {
  if constexpr (CG)
    return __ldcg(p);
  else
    return *p;
}

// glue._hc_finish, 256 threads a row (D 4096, 4 streams): warp 0 sums K blocks and Sinkhorn, then collapse, RMS-norm.
template <typename W, bool CG = false>
__device__ __forceinline__ void finish(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                                       const W* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                                       float hc_eps, int nb, int iters, int64_t r) {
  constexpr int D = 4096, S = 4;
  __shared__ float smem[8];
  __shared__ float pre[4];
  const int t = threadIdx.x, lane = t & 31;
  // A thread's 8 inputs of each half are contiguous: one 16-byte load per stream and half, issued before Sinkhorn.
  const __nv_bfloat16* row = x + r * S * D;
  uint4 xu[2][S];
  Weights<W> wu;
  for (int rep = 0; rep < 2; ++rep) {
    for (int st = 0; st < S; ++st)
      xu[rep][st] = ldr<CG>(reinterpret_cast<const uint4*>(row + st * D + rep * 2048 + t * 8));
    wu.load(nw, rep, t);
  }
  if (t < 32) {
    const float* p = part + r * nb * 32;
    float mix = 0.0f, ss = 0.0f;
    for (int b = 0; b < nb; ++b) {
      mix = mix + ldr<CG>(p + b * 32 + lane);
      ss = ss + ldr<CG>(p + b * 32 + 24);
    }
    const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(ss, (float)(S * D))));
    mix = __fmul_rn(mix, rinv);
    const int k = lane & 3, idx = lane & 15;
    const float bpre = base[k], bpost = base[4 + k], bcomb = base[8 + idx];
    const float mpre = __shfl_sync(0xffffffffu, mix, k), mpost = __shfl_sync(0xffffffffu, mix, 4 + k);
    const float mcomb = __shfl_sync(0xffffffffu, mix, 8 + idx);
    const float pr = hc_eps + glm_hc::sigmoid(__fmaf_rn(mpre, scale[0], bpre));
    const float sp = glm_hc::sigmoid(__fmaf_rn(mpost, scale[1], bpost));
    const float cl = __fmaf_rn(mcomb, scale[2], bcomb);
    float cmax = fmaxf(cl, __shfl_xor_sync(0xffffffffu, cl, 2));
    cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 1));
    const float ce = exp_approx(cl - cmax);
    float cm = fadd(hc_eps, div_full(ce, glm_hc::row4(ce)));  // ptxas: fma(rcp(sum), ce, hc_eps), as for Triton
    cm = div_full(cm, hc_eps + glm_hc::col4(cm));
    for (int it = 1; it < iters; ++it) {
      cm = div_full(cm, hc_eps + glm_hc::row4(cm));
      cm = div_full(cm, hc_eps + glm_hc::col4(cm));
    }
    if (lane < 4) {
      post[r * S + lane] = sp + sp;
      pre[lane] = pr;
    }
    if (lane < 16) comb[r * 16 + lane] = cm;
  }
  __syncthreads();
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  float c[16];
  for (int k = 0; k < 16; ++k) {
    const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][0]);
    const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][1]);
    const __nv_bfloat16* h2 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][2]);
    const __nv_bfloat16* h3 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][3]);
    const float x0 = bf(h0[k % 8]), x1 = bf(h1[k % 8]), x2 = bf(h2[k % 8]), x3 = bf(h3[k % 8]);
    float m = k % 2 == 0 ? __fmaf_rn(p1, x1, __fmul_rn(p0, x0)) : __fmaf_rn(p0, x0, __fmul_rn(p1, x1));
    m = __fmaf_rn(p2, x2, m);
    c[k] = rbf(__fmaf_rn(p3, x3, m));
  }
  float acc = __fmaf_rn(c[0], c[0], __fmul_rn(c[1], c[1]));
  for (int k = 2; k < 16; ++k) acc = k % 2 == 0 ? __fmaf_rn(c[k], c[k], acc) : fadd(fmul(c[k], c[k]), acc);
  const float total = block_sum<8>(acc, smem);
  const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(total, (float)D)));
  for (int rep = 0; rep < 2; ++rep) {
    float y[8];
    uint4 ou;
    __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ou);
    for (int j = 0; j < 8; ++j) {
      const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(wu.at(rep, j), rbf(__fmul_rn(rinv, c[rep * 8 + j]))));
      oh[j] = o;
      y[j] = bf(o);
    }
    *reinterpret_cast<uint4*>(out + r * D + rep * 2048 + t * 8) = ou;
    float g = y[0] + y[1];
    for (int j = 2; j < 8; ++j) g = y[j] + g;
    for (int o = 4; o > 0; o >>= 1) g = g + __shfl_xor_sync(0xffffffffu, g, o);
    if ((t & 7) == 0) xs[r * (D / 64) + rep * 32 + t / 8] = g;
  }
}

// TF_GLM_OVN_GLUE_FUSE: block (row, K block b) runs hc_post of its 1024 values into xout, hc_partial8, hc_finish.
template <typename W, bool LAUNCH>
__device__ __forceinline__ void glue(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g, int64_t rs, int world,
                                     const __nv_bfloat16* fn, float* part, unsigned* count, const float* base,
                                     const float* scale, const W* nw, __nv_bfloat16* out, float* xs, float* post,
                                     float* comb, float eps, float hc_eps, int iters) {
  constexpr int D = 4096, S = 4, NB = 16, KB = S * D / NB, STEPS = KB / 128, HALF = STEPS / 2;
  constexpr int64_t wide = S * D;
  __shared__ __align__(16) __nv_bfloat16 xn[KB];
  __shared__ float dots[STEPS][24];
  __shared__ float sq[4];
  __shared__ bool last;
  if constexpr (LAUNCH) qmm_frag::grid_launch();  // TF_GLM_OVN_PDL_EARLY: the next site's weights stream in meanwhile
  const int64_t r = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, grp = tid >> 7, t = tid & 127, c = t & 15, m0 = (t >> 4) & 7;
  uint4 wq[HALF][3];  // the weights of this group's steps grp * 4 .. + 3 (hc_partial8's loads), issued first
  for (int k = 0; k < HALF; ++k)
    for (int i = 0; i < 3; ++i)
      wq[k][i] = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + b * KB +
                                                       (grp * HALF + k) * 128 + c * 8));
  if (tid < KB / 8) {  // stream b / 4's new values read all 4 streams here: hence xout, never x in place
    const int s = static_cast<int>(b / 4);
    const int64_t col = b % 4 * KB + tid * 8;
    float branch[8], xv[S][8], o[8];
    glm_glue::ranks8(g, rs, r * D + col, world, branch);
    for (int u = 0; u < S; ++u) glm_glue::ld8(x + r * wide + u * D + col, xv[u]);
    glm_glue::post_s(xv, branch, comb, post, r, s, o);
    glm_glue::st8(xout + r * wide + s * D + col, o);
    glm_glue::st8(xn + tid * 8, o);  // the bf16 values hc_partial8 would read back from x
  }
  __syncthreads();
  for (int k = 0; k < HALF; ++k) {
    const int step = grp * HALF + k;
    float xv[8];
    glm_glue::ld8(xn + step * 128 + c * 8, xv);
    for (int i = 0; i < 3; ++i) {
      const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq[k][i]);
      float d = __fmul_rn(xv[1], bf(w[1]));
      d = __fmaf_rn(xv[0], bf(w[0]), d);
      for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[j], bf(w[j]), d);
      for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
      if (c == 0) dots[step][m0 + 8 * i] = d;
    }
  }
  if (grp == 0) {
    float ss = 0.0f;
    for (int s = 0; s < STEPS; ++s) {
      const float xt = bf(xn[s * 128 + t]);
      ss = __fmaf_rn(xt, xt, ss);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) sq[t >> 5] = ss;
  }
  __syncthreads();
  float* po = part + (r * NB + b) * 32;
  if (tid < 24) {
    float acc = 0.0f;
    for (int s = 0; s < STEPS; ++s) acc = acc + dots[s][tid];
    po[tid] = acc;
  } else if (tid == 32) {
    po[24] = (sq[0] + sq[2]) + (sq[1] + sq[3]);
  }
  __threadfence();  // this block's x and partials are visible before its count
  __syncthreads();
  if (tid == 0) last = atomicAdd(count + r, 1u) == NB - 1;  // the row's 16th block finishes it
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (tid == 0) count[r] = 0;  // the row's next launch counts from zero (the launch boundary orders it)
  finish<W, true>(xout, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, NB, iters, r);
}

// TF_GLM_OVN_GLUE_ROW: 4 rows a block; each quarter's hc_post in place, then hc_partial's chains; then hc_finish.
template <typename W>
__device__ __forceinline__ void rows4(__nv_bfloat16* x, const float* g, int64_t rs, int world, const __nv_bfloat16* fn,
                                      const float* base, const float* scale, const W* nw, __nv_bfloat16* out, float* xs,
                                      float* post, float* comb, float eps, float hc_eps, int iters, int64_t rows) {
  constexpr int R = 4, D = 4096, S = 4, NB = 16, KB = S * D / NB, QUARTERS = D / KB, CHUNKS = KB / 8;
  constexpr int64_t wide = S * D;
  __shared__ float part[R][NB][32];
  __shared__ float sq[2][R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R;
  const int tid = threadIdx.x, grp = tid >> 7, t = tid & 127, lane = t & 31, warp = t >> 5, c = t & 15;
  const int m0 = (t >> 4) & 7;
  for (int qq = 0; qq < QUARTERS; ++qq) {
    for (int u = tid; u < R * CHUNKS; u += 256) {
      const int64_t row = r0 + u / CHUNKS;
      if (row < rows) glm_glue::hc_post8(x, x, g, post, comb, rs, D, world, row, qq * KB + u % CHUNKS * 8);
    }
    __syncthreads();
    for (int h = 0; h < 2; ++h) {
      const int b = (grp + 2 * h) * QUARTERS + qq;
      float acc[R][3] = {}, ss[R] = {};
      for (int64_t step = 0; step < KB / 128; ++step) {
        const int64_t at = b * KB + step * 128;
        float xv[R][8];
        for (int q = 0; q < R; ++q) {
          uint4 v = make_uint4(0, 0, 0, 0);
          if (r0 + q < rows) v = *reinterpret_cast<const uint4*>(x + (r0 + q) * wide + at + c * 8);
          const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&v);
          for (int j = 0; j < 8; ++j) xv[q][j] = bf(hv[j]);
        }
        for (int i = 0; i < 3; ++i) {
          const uint4 v = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + at + c * 8));
          const __nv_bfloat16* hw = reinterpret_cast<const __nv_bfloat16*>(&v);
          float w[8];
          for (int j = 0; j < 8; ++j) w[j] = bf(hw[j]);
          for (int q = 0; q < R; ++q) {
            float d = __fmul_rn(xv[q][1], w[1]);
            d = __fmaf_rn(xv[q][0], w[0], d);
            for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[q][j], w[j], d);
            for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
            acc[q][i] = acc[q][i] + d;
          }
        }
        for (int q = 0; q < R; ++q) {
          const float xt = r0 + q < rows ? bf(x[(r0 + q) * wide + at + t]) : 0.0f;
          ss[q] = __fmaf_rn(xt, xt, ss[q]);
        }
      }
      for (int q = 0; q < R; ++q) {
        for (int o = 16; o > 0; o >>= 1) ss[q] += __shfl_xor_sync(0xffffffffu, ss[q], o);
        if (lane == 0) sq[grp][q][warp] = ss[q];
      }
      __syncthreads();
      for (int q = 0; q < R; ++q) {
        if (c == 0)
          for (int i = 0; i < 3; ++i) part[q][b][m0 + 8 * i] = acc[q][i];
        if (t == 0) part[q][b][24] = (sq[grp][q][0] + sq[grp][q][2]) + (sq[grp][q][1] + sq[grp][q][3]);
      }
      __syncthreads();
    }
  }
  for (int q = 0; q < R && r0 + q < rows; ++q)
    finish<W>(x + r0 * wide, &part[0][0][0], base, scale, nw, out + r0 * D, xs + r0 * (D / 64), post + r0 * S,
              comb + r0 * 16, eps, hc_eps, NB, iters, q);
}

}  // namespace glm_hc

// hcPre's decode windows (16 blocks a row of 1024 threads): glm_hc::partial8.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb) {
  glm_hc::partial8<false>(x, fn, part, wide, nb);
}

// TF_GLM_OVN_PDL_EARLY: hc_partial8 launched dependent, its weights loading before griddepcontrol.wait.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8_pdl(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb) {
  glm_hc::partial8<true>(x, fn, part, wide, nb);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                     const __nv_bfloat16* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                     float hc_eps, int nb, int iters) {
  glm_hc::finish(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, nb, iters, blockIdx.x);
}

// FP4-Local's float32 post-attention norm weight: hc_finish's arithmetic, the weight read as float32.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish_f32w(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                          const float* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                          float hc_eps, int nb, int iters) {
  glm_hc::finish(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, nb, iters, blockIdx.x);
}

// TF_GLM_OVN_PDL_EARLY: hc_finish releasing its dependent launch (the next site's first matmul) at entry.
#define TF_GLM_HC_FINISH_PDL(name, W)                                                                              \
  extern "C" __global__ void __launch_bounds__(256)                                                                \
      name(const __nv_bfloat16* x, const float* part, const float* base, const float* scale, const W* nw,         \
           __nv_bfloat16* out, float* xs, float* post, float* comb, float eps, float hc_eps, int nb, int iters) {  \
    qmm_frag::grid_launch();                                                                                       \
    glm_hc::finish(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, nb, iters, blockIdx.x);            \
  }
TF_GLM_HC_FINISH_PDL(tf_glm_hc_finish_pdl, __nv_bfloat16)
TF_GLM_HC_FINISH_PDL(tf_glm_hc_finish_f32w_pdl, float)
#undef TF_GLM_HC_FINISH_PDL

// TF_GLM_OVN_GLUE_FUSE's kernel (glm_hc::glue), bf16 or float32 norm weight; _pdl: TF_GLM_OVN_PDL_EARLY's trigger.
#define TF_GLM_HC_GLUE(name, W, LAUNCH)                                                                            \
  extern "C" __global__ void __launch_bounds__(256)                                                                \
      name(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g, int64_t rs, int world,                    \
           const __nv_bfloat16* fn, float* part, unsigned* count, const float* base, const float* scale,          \
           const W* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps, float hc_eps,          \
           int iters) {                                                                                            \
    glm_hc::glue<W, LAUNCH>(x, xout, g, rs, world, fn, part, count, base, scale, nw, out, xs, post, comb, eps,    \
                            hc_eps, iters);                                                                        \
  }
TF_GLM_HC_GLUE(tf_glm_hc_glue, __nv_bfloat16, false)
TF_GLM_HC_GLUE(tf_glm_hc_glue_f32w, float, false)
TF_GLM_HC_GLUE(tf_glm_hc_glue_pdl, __nv_bfloat16, true)
TF_GLM_HC_GLUE(tf_glm_hc_glue_f32w_pdl, float, true)
#undef TF_GLM_HC_GLUE

// TF_GLM_OVN_GLUE_ROW's kernel (glm_hc::rows4), bf16 or float32 norm weight.
#define TF_GLM_HC_ROW(name, W)                                                                                     \
  extern "C" __global__ void __launch_bounds__(256)                                                                \
      name(__nv_bfloat16* x, const float* g, int64_t rs, int world, const __nv_bfloat16* fn, const float* base,   \
           const float* scale, const W* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,    \
           float hc_eps, int iters, int64_t rows) {                                                                \
    glm_hc::rows4<W>(x, g, rs, world, fn, base, scale, nw, out, xs, post, comb, eps, hc_eps, iters, rows);         \
  }
TF_GLM_HC_ROW(tf_glm_hc_row, __nv_bfloat16)
TF_GLM_HC_ROW(tf_glm_hc_row_f32w, float)
#undef TF_GLM_HC_ROW
