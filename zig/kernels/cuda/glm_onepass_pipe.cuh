// TF_GLM_OVN_ONEPASS_PIPE: glm_onepass.cu's pipeline with switches reverting each byte difference; all on == fused.
#pragma once
#include "glm_onepass_common.cuh"
#include <cuda_fp8.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
namespace glm_onepass_pipe {

constexpr int CHUNK_TILES = 512 / KT; // 16 key tiles a 512-key chunk (the default's CHUNK)

struct SmemPipe {
    uint4 kt[NB][KT * CH];
    uint16_t ps[HQ * KT];
    float ks[NB][KT];
    float alpha[HQ];
    float lsum[HQ];
    float cm[HQ]; // CHUNK: each 512-chunk's running max a head, published by the QK warps for the PV warps' fold
};
static_assert(sizeof(SmemPipe) == 100512, "cuda_onepass.zig's pipe smem");

// DECODE on: default decode (scale folded into the value); raw_load loads the scale only in lane 0, so broadcast it.
template <bool DECODE>
__device__ __forceinline__ void convert_pipe(const Raw& r, uint4* kt, float* ks, int pw, int lane) {
    if constexpr (!DECODE) {
        convert(r, kt, ks, pw, lane);
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key = pw + 4 * i;
        const uint32_t in[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
        const float s = __shfl_sync(0xffffffffu, r.s[i], 0);
        uint32_t w[8];
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint16_t bits = (uint16_t)((in[m >> 1] >> (16 * (m & 1))) & 0xffffu);
            const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
            const __nv_bfloat162 bb = __floats2bfloat162_rn(__fmul_rn(v.x, s), __fmul_rn(v.y, s));
            w[m] = *reinterpret_cast<const uint32_t*>(&bb);
        }
        kt[key * CH + swz(key, 2 * lane)] = make_uint4(w[0], w[1], w[2], w[3]);
        kt[key * CH + swz(key, 2 * lane + 1)] = make_uint4(w[4], w[5], w[6], w[7]);
        if (lane == 0) ks[key] = s;
    }
}

// tf_glm_merge's factors for folding a chunk (cm) into the running merged max M: b = exp(cm - nm), a = exp(M - nm).
__device__ __forceinline__ void merge_ba(float& M, float cm, float& b, float& a) {
    const float nm = fmaxf(M, cm);
    b = ex2(__fmul_rn(__fsub_rn(cm, nm), LOG2E));
    a = M == -INFINITY ? 0.0f : ex2(__fmul_rn(__fsub_rn(M, nm), LOG2E));
    M = nm;
}

template <bool DECODE, bool SCALE, bool CHUNK>
__device__ __forceinline__ void onepass_pipe(const __nv_bfloat16* __restrict__ QA, const uint8_t* __restrict__ LC,
                                             const float* __restrict__ LS, uint32_t PR,
                                             const int32_t* __restrict__ TOK, int W, const int32_t* __restrict__ CNT,
                                             __nv_bfloat16* __restrict__ OUT, int H, float scale) {
    extern __shared__ __align__(16) uint8_t smem_raw[];
    SmemPipe& S = *reinterpret_cast<SmemPipe*>(smem_raw);
    const int r = blockIdx.x;
    const int n = CNT[r];
    if (n <= 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int ntile = (n + KT - 1) / KT;
    const int32_t* row = TOK + (size_t)r * W;

    if (warp >= 8) {
        const int pw = warp - 8;
        Raw A, B;
        int ia[8], ib[8];
        ids_load(row, n, 0, pw, ia);
        ids_load(row, n, 1, pw, ib);
        raw_load(LC, LS, PR, ia, lane, A);
        raw_load(LC, LS, PR, ib, lane, B);
        ids_load(row, n, 2, pw, ia);
        ids_load(row, n, 3, pw, ib);
        int b = 0;
        for (int t = 0; t < ntile; t += 2) {
            if (t >= NB) bar_sync(BAR_EMPTY + b, N_EMPTY);
            convert_pipe<DECODE>(A, S.kt[b], S.ks[b], pw, lane);
            bar_arrive(BAR_FULL + b, N_FULL);
            b = b + 1 == NB ? 0 : b + 1;
            if (t + 2 < ntile) {
                raw_load(LC, LS, PR, ia, lane, A);
                ids_load(row, n, t + 4, pw, ia);
            }
            if (t + 1 >= ntile) break;
            if (t + 1 >= NB) bar_sync(BAR_EMPTY + b, N_EMPTY);
            convert_pipe<DECODE>(B, S.kt[b], S.ks[b], pw, lane);
            bar_arrive(BAR_FULL + b, N_FULL);
            b = b + 1 == NB ? 0 : b + 1;
            if (t + 3 < ntile) {
                raw_load(LC, LS, PR, ib, lane, B);
                ids_load(row, n, t + 5, pw, ib);
            }
        }
        return;
    }

    if (warp >= 4 && warp < 7) {
        const int j = warp - 4;
        const int hq = 8 * j + g;
        uint32_t qf[32][2];
#pragma unroll
        for (int s = 0; s < 32; ++s) {
            const __nv_bfloat16* q = QA + ((size_t)r * H + (hq < H ? hq : 0)) * LW + 16 * s + 2 * t4;
            qf[s][0] = hq < H ? *reinterpret_cast<const uint32_t*>(q) : 0u;
            qf[s][1] = hq < H ? *reinterpret_cast<const uint32_t*>(q + 8) : 0u;
        }
        float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
        float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f}; // CHUNK: the merged max and sum across chunks
        const int ar = lane & 15, ac = lane >> 4;
        uint32_t a[3][2][4];
        unsigned ktn;
        float acc[2][4], accn[2][4];
#define LOADA(slot, s)                                                                                   \
    {                                                                                                    \
        ldsm_x4(a[slot][0], ktn + 16 * (ar * CH + swz(ar, 2 * (s) + ac)));                               \
        ldsm_x4(a[slot][1], ktn + 16 * ((16 + ar) * CH + swz(16 + ar, 2 * (s) + ac)));                   \
    }
#define QKSTEP(s)                                                                                        \
    {                                                                                                    \
        if ((s) + 2 < 32) LOADA(((s) + 2) % 3, (s) + 2)                                                  \
        mma16816(accn[0], a[(s) % 3][0], qf[s][0], qf[s][1]);                                            \
        mma16816(accn[1], a[(s) % 3][1], qf[s][0], qf[s][1]);                                            \
    }
#define QKBEGIN()                                                                                        \
    {                                                                                                    \
        for (int i_ = 0; i_ < 2; ++i_)                                                                   \
            for (int e_ = 0; e_ < 4; ++e_) accn[i_][e_] = 0.0f;                                          \
        LOADA(0, 0)                                                                                      \
        LOADA(1, 1)                                                                                      \
    }
        bar_sync(BAR_FULL + 0, N_FULL);
        ktn = (unsigned)__cvta_generic_to_shared(S.kt[0]);
        QKBEGIN()
#pragma unroll
        for (int s = 0; s < 32; ++s) QKSTEP(s)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][e] = accn[i][e];
        int b = 0;
        for (int t = 0; t < ntile; ++t) {
            const int t0 = t * KT, bn = b + 1 == NB ? 0 : b + 1;
            const bool boundary = CHUNK && ((t + 1) % CHUNK_TILES == 0 || t + 1 == ntile);
            if (t + 1 < ntile) bar_sync(BAR_FULL + bn, N_FULL);
            ktn = (unsigned)__cvta_generic_to_shared(S.kt[t + 1 < ntile ? bn : b]);
            QKBEGIN()
#pragma unroll
            for (int s = 0; s < 8; ++s) QKSTEP(s)
            float sc[2][4], kk[2][2];
            bool val[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int key = 16 * i + 8 * hi + g;
                    kk[i][hi] = S.ks[b][key];
                    val[i][hi] = t0 + key < n;
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float raw = acc[i][2 * hi + e];
                        const float sv = SCALE ? __fmul_rn(raw, scale) : __fmul_rn(__fmul_rn(raw, kk[i][hi]), scale);
                        sc[i][2 * hi + e] = val[i][hi] ? sv : -INFINITY;
                    }
                }
            float nm[2], alpha[2];
            bool act[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float x = fmaxf(fmaxf(sc[0][e], sc[0][2 + e]), fmaxf(sc[1][e], sc[1][2 + e]));
                x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
                x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
                x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 16));
                act[e] = x != -INFINITY;
                nm[e] = act[e] ? fmaxf(m[e], x) : m[e];
                const float a_ = m[e] == -INFINITY ? 0.0f : ex2(__fmul_rn(__fsub_rn(m[e], nm[e]), LOG2E));
                alpha[e] = act[e] ? a_ : 1.0f;
            }
#pragma unroll
            for (int s = 8; s < 16; ++s) QKSTEP(s)
            float p[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const int e = x & 1;
                    p[i][x] = val[i][x >> 1] && act[e] ? ex2(__fmul_rn(__fsub_rn(sc[i][x], nm[e]), LOG2E)) : 0.0f;
                }
#pragma unroll
            for (int s = 16; s < 24; ++s) QKSTEP(s)
            float sw[4][2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int hi = 0; hi < 2; ++hi)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        float y = p[i][2 * hi + e];
                        y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 4));
                        y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 16));
                        y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 8));
                        sw[2 * i + hi][e] = y;
                    }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float tot = __fadd_rn(__fadd_rn(sw[0][e], sw[2][e]), __fadd_rn(sw[1][e], sw[3][e]));
                l[e] = __fmaf_rn(l[e], alpha[e], tot);
                m[e] = nm[e];
            }
#pragma unroll
            for (int s = 24; s < 32; ++s) QKSTEP(s)
            if (t > 0) bar_sync(BAR_PFREE, N_QV);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const int h = 8 * j + 2 * t4 + (x & 1), key = 16 * i + 8 * (x >> 1) + g;
                    const float pv = SCALE ? p[i][x] : __fmul_rn(p[i][x], kk[i][x >> 1]);
                    S.ps[h * KT + 8 * pswz(h, key >> 3) + (key & 7)] = bf16(pv);
                }
            if (g == 0) {
                S.alpha[8 * j + 2 * t4] = alpha[0];
                S.alpha[8 * j + 2 * t4 + 1] = alpha[1];
                if (boundary) {
                    S.cm[8 * j + 2 * t4] = m[0];
                    S.cm[8 * j + 2 * t4 + 1] = m[1];
                }
            }
            bar_arrive(BAR_PREADY, N_QV);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][e] = accn[i][e];
            if (boundary) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float bb, aa;
                    merge_ba(M[e], m[e], bb, aa);
                    L[e] = __fmaf_rn(L[e], aa, __fmul_rn(bb, l[e]));
                    m[e] = -INFINITY;
                    l[e] = 0.0f;
                }
            }
            b = bn;
        }
#undef QKSTEP
#undef QKBEGIN
#undef LOADA
        if (g == 0) {
            S.lsum[8 * j + 2 * t4] = CHUNK ? L[0] : l[0];
            S.lsum[8 * j + 2 * t4 + 1] = CHUNK ? L[1] : l[1];
        }
        bar_arrive(BAR_FIN, N_QV);
        return;
    }

    const int v = warp < 4 ? warp : 4;
    const int nt = pv_tiles(v);
    int tb = 0;
#pragma unroll
    for (int k = 0; k < NV - 1; ++k) tb += k < v ? pv_tiles(k) : 0;
    float o[PV_MAX][3][4], O[PV_MAX][3][4];
    float Mpv[3][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i)
#pragma unroll
        for (int nh = 0; nh < 3; ++nh)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[i][nh][e] = O[i][nh][e] = 0.0f;
    const unsigned psb = (unsigned)__cvta_generic_to_shared(S.ps);
    const int vk = (lane & 7) + 8 * (lane >> 4), vc = (lane >> 3) & 1;
    int b = 0;
    for (int t = 0; t < ntile; ++t) {
        const unsigned ktb = (unsigned)__cvta_generic_to_shared(S.kt[b]);
        const bool boundary = CHUNK && ((t + 1) % CHUNK_TILES == 0 || t + 1 == ntile);
        bar_sync(BAR_PREADY, N_QV);
        uint32_t pb[3][4];
#pragma unroll
        for (int nh = 0; nh < 3; ++nh) {
            const int h = 8 * nh + (lane & 7), c = lane >> 3;
            ldsm_x4(pb[nh], psb + 2 * (h * KT + 8 * pswz(h, c)));
        }
        float al[3][2], cm[3][2];
#pragma unroll
        for (int nh = 0; nh < 3; ++nh) {
            al[nh][0] = S.alpha[8 * nh + 2 * t4];
            al[nh][1] = S.alpha[8 * nh + 2 * t4 + 1];
            if (boundary) {
                cm[nh][0] = S.cm[8 * nh + 2 * t4];
                cm[nh][1] = S.cm[8 * nh + 2 * t4 + 1];
            }
        }
        if (t + 1 < ntile) bar_arrive(BAR_PFREE, N_QV);
        bool one = true;
#pragma unroll
        for (int nh = 0; nh < 3; ++nh) one = one && al[nh][0] == 1.0f && al[nh][1] == 1.0f;
        if (!__all_sync(0xffffffffu, one)) {
#pragma unroll
            for (int i = 0; i < PV_MAX; ++i)
                if (i < nt)
#pragma unroll
                    for (int nh = 0; nh < 3; ++nh)
#pragma unroll
                        for (int e = 0; e < 4; ++e) o[i][nh][e] = __fmul_rn(o[i][nh][e], al[nh][e & 1]);
        }
        uint32_t a[2][2][4];
#define LOADV(slot, i)                                                                                    \
    {                                                                                                     \
        ldsm_x4_t(a[slot][0], ktb + 16 * (vk * CH + swz(vk, 2 * (tb + (i)) + vc)));                       \
        ldsm_x4_t(a[slot][1], ktb + 16 * ((16 + vk) * CH + swz(16 + vk, 2 * (tb + (i)) + vc)));           \
    }
        LOADV(0, 0)
#pragma unroll
        for (int i = 0; i < PV_MAX; ++i) {
            if (i < nt) {
                if (i + 1 < nt) LOADV((i + 1) & 1, i + 1)
#pragma unroll
                for (int nh = 0; nh < 3; ++nh) mma16816(o[i][nh], a[i & 1][0], pb[nh][0], pb[nh][1]);
#pragma unroll
                for (int nh = 0; nh < 3; ++nh) mma16816(o[i][nh], a[i & 1][1], pb[nh][2], pb[nh][3]);
            }
        }
#undef LOADV
        if (t + NB < ntile) bar_arrive(BAR_EMPTY + b, N_EMPTY);
        if (boundary) {
#pragma unroll
            for (int nh = 0; nh < 3; ++nh)
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    float bb, aa;
                    merge_ba(Mpv[nh][hf], cm[nh][hf], bb, aa);
#pragma unroll
                    for (int i = 0; i < PV_MAX; ++i)
                        if (i < nt)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                if ((e & 1) == hf) O[i][nh][e] = __fmaf_rn(O[i][nh][e], aa, __fmul_rn(bb, o[i][nh][e]));
                }
        }
        b = b + 1 == NB ? 0 : b + 1;
    }
    bar_sync(BAR_FIN, N_QV);
    float lv[3][2];
#pragma unroll
    for (int nh = 0; nh < 3; ++nh) {
        lv[nh][0] = S.lsum[8 * nh + 2 * t4];
        lv[nh][1] = S.lsum[8 * nh + 2 * t4 + 1];
    }
    uint16_t* st = reinterpret_cast<uint16_t*>(S.kt[ntile % NB]);
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i)
        if (i < nt)
#pragma unroll
            for (int nh = 0; nh < 3; ++nh)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int h = 8 * nh + 2 * t4 + (e & 1), lat = 16 * (tb + i) + g + 8 * (e >> 1);
                    const float ov = CHUNK ? O[i][nh][e] : o[i][nh][e];
                    st[h * LW + lat] = bf16(div_full(ov, lv[nh][e & 1]));
                }
    __syncwarp();
    for (int c = lane; c < H * 2 * nt; c += 32) {
        const int h = c / (2 * nt), q = c - h * 2 * nt;
        *reinterpret_cast<uint4*>(OUT + ((size_t)r * H + h) * LW + 16 * tb + 8 * q) =
            *reinterpret_cast<const uint4*>(st + h * LW + 16 * tb + 8 * q);
    }
}

} // namespace glm_onepass_pipe
#endif
