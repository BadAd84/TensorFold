// TensorFold glm5_next/cuda/sparse_onepass.cu (OTS 0085 + 0094) as is: production's prompt sparse attention, its bits.
// Only the cache addressing is ours: FP8 rows on a slab (scales apart) or on pages (a part's codes, then scales).
#include <cuda_bf16.h>
#include <stdint.h>

namespace glm_onepass {

constexpr int HQ = 24, LW = 512, KT = 32, NB = 3;    // heads kept, latent width, keys a tile, key-tile buffers
constexpr int NV = 5, NQ = 3, NPW = 4;               // PV, QK, producer warps (warp roles: see the kernel)
constexpr int NT = (NV + NQ + NPW) * 32;

}  // namespace glm_onepass

// mul.rn.bf16x2 needs sm_90 and cvt.rn.f16x2.e4m3x2 sm_89: older targets build a trapping kernel (never launched)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
namespace glm_onepass {

// PV warp v's share of the 32 latent m16 tiles: 6 / 6 / 6 / 7 / 7, so each SM scheduler carries ~100 mma a tile (0094)
__device__ __forceinline__ constexpr int pv_tiles(int v) { return v < 3 ? 6 : 7; }
constexpr int PV_MAX = 7;                            // the most tiles a PV warp takes
constexpr int CH = LW * 2 / 16;                      // 16-byte chunks of a bf16 latent row (64)
constexpr int BAR_FULL = 1, BAR_EMPTY = 4, BAR_PREADY = 7, BAR_PFREE = 8, BAR_FIN = 9;
constexpr int N_FULL = (NPW + NQ) * 32, N_EMPTY = (NV + NPW) * 32, N_QV = (NQ + NV) * 32;

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], unsigned s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], unsigned s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void bar_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint16_t bf16(float x) {
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

// Two e4m3 codes (low byte first) -> bf16x2 by Triton's own sequence (f16x2, then cvt.bf16.f16): NaN tiles only.
__device__ __forceinline__ uint32_t e4m3x2_bf16x2_tri(uint32_t two) {
    uint32_t r;
    const unsigned short v = (unsigned short)two;
    asm("{\n .reg .b32 a;\n .reg .f16 a<2>;\n .reg .b16 b<2>;\n cvt.rn.f16x2.e4m3x2 a, %1;\n mov.b32 {a0, a1}, a;\n"
        " cvt.bf16.f16 b0, a0;\n cvt.bf16.f16 b1, a1;\n mov.b32 %0, {b0, b1};\n}" : "=r"(r) : "h"(v));
    return r;
}

// Four e4m3 codes -> two bf16x2 exactly for every code but NaN: sign-extended bits shifted and masked, times 2^120.
__device__ __forceinline__ void e4m3x4_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
    uint32_t a, b;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(a) : "r"(w), "r"(0u), "r"(0x9180u));   // [sext b1 | sext b0]
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b) : "r"(w), "r"(0u), "r"(0xB3A2u));   // [sext b3 | sext b2]
    a = (a << 4) & 0x87F087F0u;
    b = (b << 4) & 0x87F087F0u;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(0x7B807B80u));   // 2^120
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(hi) : "r"(b), "r"(0x7B807B80u));
}

// a key tile's 16-byte chunk c of key r, bank-conflict free for the ldmatrix rows and the producers' stores (0094)
__device__ __forceinline__ int swz(int r, int c) { return c ^ ((r ^ (c >> 3)) & 7); }
__device__ __forceinline__ int pswz(int r, int c) { return c ^ ((r >> 1) & 3); }

constexpr float LOG2E = 1.44269502162933349609375f;   // 0f3FB8AA3B

struct Smem {
    uint4 kt[NB][KT * CH];      // 96 KB: key tiles, bf16 [key][64 chunks], swizzled
    uint16_t ps[HQ * KT];       // P [head][key] bf16, 4 chunks a row, chunk-swizzled
    float ks[NB][KT];
    float alpha[HQ];
    float lsum[HQ];
};
static_assert(sizeof(Smem) == 100416, "cuda_onepass.zig's smem");

struct Raw {
    uint4 v[8];
    float s[8];
};

__device__ __forceinline__ void ids_load(const int32_t* __restrict__ row, int n, int tile, int pw, int (&id)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = tile * KT + pw + 4 * i;
        id[i] = k < n ? row[k] : -1;
    }
}

// Token k's codes at LC + 512 k; its scale at LS[k] (slab) or, on pages of PR rows, after its part's 512 code rows.
__device__ __forceinline__ void raw_load(const uint8_t* __restrict__ LC, const float* __restrict__ LS, uint32_t PR,
                                         const int (&id)[8], int lane, Raw& r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t k = id[i] < 0 ? 0 : id[i];
        r.v[i] = id[i] < 0 ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4*>(LC + k * LW + 16 * lane);
        float s = 1.0f;
        if (id[i] >= 0 && lane == 0) {
            const uint32_t off = PR ? (uint32_t)k % PR : 0u;
            s = LS[PR ? (k - off) * (LW / 4) + off : k];
        }
        r.s[i] = s;
    }
}

__device__ __forceinline__ void convert(const Raw& r, uint4* kt, float* ks, int pw, int lane) {
    uint32_t nan = 0;                                     // bit 7 of a byte: some code of it was 0x7F / 0xFF
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t in[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) nan |= (in[q] & 0x7F7F7F7Fu) + 0x01010101u;
    }
    if (__any_sync(0xffffffffu, nan & 0x80808080u)) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int key = pw + 4 * i;
            const uint32_t in[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
            uint32_t w[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) w[m] = e4m3x2_bf16x2_tri(in[m >> 1] >> (16 * (m & 1)));
            kt[key * CH + swz(key, 2 * lane)] = make_uint4(w[0], w[1], w[2], w[3]);
            kt[key * CH + swz(key, 2 * lane + 1)] = make_uint4(w[4], w[5], w[6], w[7]);
            if (lane == 0) ks[key] = r.s[i];
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key = pw + 4 * i;
        const uint32_t in[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
        uint32_t w[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) e4m3x4_bf16(in[q], w[2 * q], w[2 * q + 1]);
        kt[key * CH + swz(key, 2 * lane)] = make_uint4(w[0], w[1], w[2], w[3]);
        kt[key * CH + swz(key, 2 * lane + 1)] = make_uint4(w[4], w[5], w[6], w[7]);
        if (lane == 0) ks[key] = r.s[i];
    }
}

// production's onepass_kernel body: one CTA a row, warps 0-3 PV 0-3, 4-6 QK, 7 PV 4, 8-11 producers
__device__ __forceinline__ void onepass(const __nv_bfloat16* __restrict__ QA, const uint8_t* __restrict__ LC,
                                        const float* __restrict__ LS, uint32_t PR, const int32_t* __restrict__ TOK,
                                        int W, const int32_t* __restrict__ CNT, __nv_bfloat16* __restrict__ OUT, int H,
                                        float scale) {
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int r = blockIdx.x;
    const int n = CNT[r];
    if (n <= 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int ntile = (n + KT - 1) / KT;
    const int32_t* row = TOK + (size_t)r * W;

    if (warp >= 8) {
        // producers: keys pw + 4 i, 16 B a lane, two tiles ahead
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
            convert(A, S.kt[b], S.ks[b], pw, lane);
            bar_arrive(BAR_FULL + b, N_FULL);
            b = b + 1 == NB ? 0 : b + 1;
            if (t + 2 < ntile) {
                raw_load(LC, LS, PR, ia, lane, A);
                ids_load(row, n, t + 4, pw, ia);
            }
            if (t + 1 >= ntile) break;
            if (t + 1 >= NB) bar_sync(BAR_EMPTY + b, N_EMPTY);
            convert(B, S.kt[b], S.ks[b], pw, lane);
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
        // QK warps: heads 8 j .. 8 j + 7 (an n8 tile), all 32 keys
        const int j = warp - 4;
        const int hq = 8 * j + g;                         // this lane's B column (head)
        uint32_t qf[32][2];                               // q^T B fragments: step s, k = 16 s + 2 t (+ 8)
#pragma unroll
        for (int s = 0; s < 32; ++s) {
            const __nv_bfloat16* q = QA + ((size_t)r * H + (hq < H ? hq : 0)) * LW + 16 * s + 2 * t4;
            qf[s][0] = hq < H ? *reinterpret_cast<const uint32_t*>(q) : 0u;
            qf[s][1] = hq < H ? *reinterpret_cast<const uint32_t*>(q + 8) : 0u;
        }
        float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};   // heads 8 j + 2 t, + 1
        const int ar = lane & 15, ac = lane >> 4;
        // QK(t + 1)'s 32 steps are issued between softmax(t)'s stages
        uint32_t a[3][2][4];
        unsigned ktn;
        float acc[2][4], accn[2][4];
#define LOADA(slot, s)                                                                                   \
    {                                                                                                    \
        ldsm_x4(a[slot][0], ktn + 16 * (ar * CH + swz(ar, 2 * (s) + ac)));                              \
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
            if (t + 1 < ntile) bar_sync(BAR_FULL + bn, N_FULL);
            // the last tile's "next" scores run on its own buffer and are dropped (no branch in the block below)
            ktn = (unsigned)__cvta_generic_to_shared(S.kt[t + 1 < ntile ? bn : b]);
            QKBEGIN()
#pragma unroll
            for (int s = 0; s < 8; ++s) QKSTEP(s)
            // acc[i][e]: key 16 i + g (e < 2) or 16 i + 8 + g (e >= 2), head 8 j + 2 t + (e & 1)
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
                    for (int e = 0; e < 2; ++e)
                        sc[i][2 * hi + e] =
                            val[i][hi] ? __fmul_rn(__fmul_rn(acc[i][2 * hi + e], kk[i][hi]), scale) : -INFINITY;
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
            // the p sums in Triton's tree: (p[2u] + p[2u + 1]) over u, then u ^ 2, u ^ 1, then (s0 + s2) + (s1 + s3)
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
            if (t > 0) bar_sync(BAR_PFREE, N_QV);         // the PV warps have read the previous P and alphas
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const int h = 8 * j + 2 * t4 + (x & 1), key = 16 * i + 8 * (x >> 1) + g;
                    S.ps[h * KT + 8 * pswz(h, key >> 3) + (key & 7)] = bf16(__fmul_rn(p[i][x], kk[i][x >> 1]));
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
            b = bn;
        }
#undef QKSTEP
#undef QKBEGIN
#undef LOADA
        if (g == 0) {
            S.lsum[8 * j + 2 * t4] = l[0];
            S.lsum[8 * j + 2 * t4 + 1] = l[1];
        }
        bar_arrive(BAR_FIN, N_QV);
        return;
    }

    // PV warps: latent rows 16 tb .. 16 (tb + nt) - 1 (nt m16 tiles)
    const int v = warp < 4 ? warp : 4;
    const int nt = pv_tiles(v);
    int tb = 0;
#pragma unroll
    for (int k = 0; k < NV - 1; ++k) tb += k < v ? pv_tiles(k) : 0;
    float o[PV_MAX][3][4];
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i)
#pragma unroll
        for (int nh = 0; nh < 3; ++nh)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[i][nh][e] = 0.0f;
    const unsigned psb = (unsigned)__cvta_generic_to_shared(S.ps);
    const int vk = (lane & 7) + 8 * (lane >> 4), vc = (lane >> 3) & 1;
    int b = 0;
    for (int t = 0; t < ntile; ++t) {
        const unsigned ktb = (unsigned)__cvta_generic_to_shared(S.kt[b]);
        bar_sync(BAR_PREADY, N_QV);
        uint32_t pb[3][4];                                // P^T B fragments: heads 8 nh .., keys 0-15 (0, 1), 16-31
#pragma unroll
        for (int nh = 0; nh < 3; ++nh) {
            const int h = 8 * nh + (lane & 7), c = lane >> 3;
            ldsm_x4(pb[nh], psb + 2 * (h * KT + 8 * pswz(h, c)));
        }
        float al[3][2];
#pragma unroll
        for (int nh = 0; nh < 3; ++nh) {
            al[nh][0] = S.alpha[8 * nh + 2 * t4];
            al[nh][1] = S.alpha[8 * nh + 2 * t4 + 1];
        }
        if (t + 1 < ntile) bar_arrive(BAR_PFREE, N_QV);   // the QK warps write the next P only after this
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
        uint32_t a[2][2][4];                              // K^T A fragments, two m16 tiles in flight
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
        b = b + 1 == NB ? 0 : b + 1;
    }
    bar_sync(BAR_FIN, N_QV);
    float lv[3][2];
#pragma unroll
    for (int nh = 0; nh < 3; ++nh) {
        lv[nh][0] = S.lsum[8 * nh + 2 * t4];
        lv[nh][1] = S.lsum[8 * nh + 2 * t4 + 1];
    }
    // out[r, h, 16 tb ..] = bf16(o / l): staged as [head][512] in the key-tile buffer no warp reads any more
    uint16_t* st = reinterpret_cast<uint16_t*>(S.kt[ntile % NB]);
#pragma unroll
    for (int i = 0; i < PV_MAX; ++i)
        if (i < nt)
#pragma unroll
            for (int nh = 0; nh < 3; ++nh)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int h = 8 * nh + 2 * t4 + (e & 1), lat = 16 * (tb + i) + g + 8 * (e >> 1);
                    st[h * LW + lat] = bf16(div_full(o[i][nh][e], lv[nh][e & 1]));
                }
    __syncwarp();
    for (int c = lane; c < H * 2 * nt; c += 32) {
        const int h = c / (2 * nt), q = c - h * 2 * nt;
        *reinterpret_cast<uint4*>(OUT + ((size_t)r * H + h) * LW + 16 * tb + 8 * q) =
            *reinterpret_cast<const uint4*>(st + h * LW + 16 * tb + 8 * q);
    }
}

}  // namespace glm_onepass
#endif

// QA / OUT [R][H][512] bf16 (17 <= H <= 24), TOK [R][W] i32 rows, CNT [R] (count-0 rows left alone), LC / LS / PR as
// raw_load reads them (PR 0: a slab); grid (R) of 384 threads, sizeof(glm_onepass::Smem) bytes of dynamic smem.
extern "C" __global__ void __launch_bounds__(glm_onepass::NT, 1)
    tf_glm_sparse_onepass(const __nv_bfloat16* __restrict__ QA, const uint8_t* __restrict__ LC,
                          const float* __restrict__ LS, unsigned PR, const int32_t* __restrict__ TOK, int W,
                          const int32_t* __restrict__ CNT, __nv_bfloat16* __restrict__ OUT, int H, float scale) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    glm_onepass::onepass(QA, LC, LS, PR, TOK, W, CNT, OUT, H, scale);
#else
    __trap();
#endif
}
