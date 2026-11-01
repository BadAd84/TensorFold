// Copied from glm_onepass.cu for the measurement pipeline (glm_onepass_pipe.cuh), separate namespace; keep in sync.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace glm_onepass_pipe {

constexpr int HQ = 24, LW = 512, KT = 32, NB = 3; // heads kept, latent width, keys a tile, key-tile buffers
constexpr int NV = 5, NQ = 3, NPW = 4; // PV, QK, producer warps (warp roles: see the kernel)
constexpr int NT = (NV + NQ + NPW) * 32;

} // namespace glm_onepass_pipe

// mul.rn.bf16x2 needs sm_90 and cvt.rn.f16x2.e4m3x2 sm_89: older targets build a trapping kernel (never launched)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
namespace glm_onepass_pipe {

__device__ __forceinline__ constexpr int pv_tiles(int v) { return v < 3 ? 6 : 7; }
constexpr int PV_MAX = 7; // the most tiles a PV warp takes
constexpr int CH = LW * 2 / 16; // 16-byte chunks of a bf16 latent row (64)
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
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(a) : "r"(w), "r"(0u), "r"(0x9180u));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b) : "r"(w), "r"(0u), "r"(0xB3A2u));
    a = (a << 4) & 0x87F087F0u;
    b = (b << 4) & 0x87F087F0u;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(0x7B807B80u));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(hi) : "r"(b), "r"(0x7B807B80u));
}

__device__ __forceinline__ int swz(int r, int c) { return c ^ ((r ^ (c >> 3)) & 7); }
__device__ __forceinline__ int pswz(int r, int c) { return c ^ ((r >> 1) & 3); }

constexpr float LOG2E = 1.44269502162933349609375f; // 0f3FB8AA3B

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

// The one-pass's FP8 decode: e4m3 -> bf16 unscaled (bit trick, NaN tiles via the Triton sequence), scale kept in ks.
__device__ __forceinline__ void convert(const Raw& r, uint4* kt, float* ks, int pw, int lane) {
    uint32_t nan = 0;
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

} // namespace glm_onepass_pipe
#endif
