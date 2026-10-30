// TF_GLM_OVN_* prompt experts, ported from TensorFold glm5_next/cuda/exl3.cu (Apache-2.0, ots0106 and ots0127).
#pragma once
namespace tf_exl3_ovn {
using namespace tf_exl3;
using namespace tf_exl3_prompt;

// ots0106 (exl3.cu:597-640): 64 k a stage, member rows unpadded with their 16-byte chunks XOR-swizzled by row & 7.
constexpr int KBK = 64, KKT = KBK / 16, KXS = KBK, KWORDS = 16 * KKT * 32;
template <int XR>
constexpr int kstages() { return XR == TPM ? 3 : 2; }   // one row copy a stage: 3 stages; gate and up apart: 2
template <int XR>
constexpr int kstage_halves() { return XR * KXS + 2 * KWORDS; }
constexpr int kmax(int a, int b) { return a > b ? a : b; }
constexpr int ksmem_bytes() {
    return kmax(kmax(kstages<TPM>() * kstage_halves<TPM>() * 2, kstages<2 * TPM>() * kstage_halves<2 * TPM>() * 2),
                kmax(2 * TPH * GU_ES * 4, TPH * DN_ES * 4));
}
static_assert(ksmem_bytes() == 48 * 1024, "exl3_prompt.zig's k64_shared: the 64-k kernels' dynamic shared bytes");
__device__ __forceinline__ int kswz(int row, int chunk) { return chunk ^ (row & 7); }

// decode_tile_p's states from the lane's own word and its left neighbour's by shuffle (exl3.cu:46-54): the same states.
__device__ __forceinline__ void kdecode(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = decode2<CB_MCG>(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);         // (s >> 8) & 0xffff
    b0[1] = decode2<CB_MCG>(s & 0xffffu, w >> 16);
    b1[0] = decode2<CB_MCG>(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));  // (w >> 12), (w >> 8) & 0xffff
    b1[1] = decode2<CB_MCG>(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                 // (w >> 4) & 0xffff
}

// tmain over 64-k stages (exl3.cu:648-722, FST: ots0127's addressing hoisted): the same mma chains in the same k order.
// XK: member rows `xk` halves apart, as tmain's.
template <int MATS, int XR, bool FST, bool XK = false>
__device__ __forceinline__ void kmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, const int* rows_sh, int cnt, int e, int K, int NTILES, int cb, int xdiv,
        half* stage, float (&acc)[TMT][TNB][4], int xk = 0) {
    constexpr int NST = kstages<XR>(), SH = kstage_halves<XR>();
    constexpr int XP = XR * (KBK / 8), WP = 16 * KKT * 8;   // a stage's 16-byte pieces: member rows, trellis words
    constexpr int RPT = (XP + TT - 1) / TT, WPT = (WP + TT - 1) / TT;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int chunks = K / KBK;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(tp1[e]);
    const int nblocks = (cnt + 15) >> 4;   // 16-member blocks in use
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < TNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    [[maybe_unused]] const half* xsrc[FST ? RPT : 1];   // FST: each thread's pieces, addressed once a block
    [[maybe_unused]] int xoff[FST ? RPT : 1];
    [[maybe_unused]] bool xon[FST ? RPT : 1], xok[FST ? RPT : 1];
    [[maybe_unused]] const uint32_t* wsrc[FST ? WPT : 1];
    [[maybe_unused]] int woff[FST ? WPT : 1];
    if constexpr (FST) {
#pragma unroll
        for (int p = 0; p < RPT; ++p) {
            const int idx = threadIdx.x + p * TT;
            const int m = idx / (TPM * (KBK / 8)), rem = idx % (TPM * (KBK / 8)), i = rem / (KBK / 8), j = rem % (KBK / 8);
            xon[p] = idx < XP && i < nblocks * 16;   // a 16-member block without a member: never read
            const int r = xon[p] ? rows_sh[i] : -1;
            xsrc[p] = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * (XK ? xk : K) + j * 8;
            xoff[p] = (m * TPM + i) * KXS + kswz(i, j) * 8;
            xok[p] = r >= 0;
        }
#pragma unroll
        for (int p = 0; p < WPT; ++p) {
            const int idx = threadIdx.x + p * TT;
            const int q = idx / (KKT * 8), kk = (idx / 8) % KKT, part = idx % 8;
            int wm, nt;
            ttile<MATS>(q < 16 ? q : 0, cb, wm, nt);
            wsrc[p] = (wm ? T1 : T0) + ((size_t)kk * NTILES + nt) * 32 + part * 4;
            woff[p] = (q * KKT + kk) * 32 + part * 4;
        }
    }
    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % NST) * SH;
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * KXS);
        const int k0 = chunk * KBK;
        if constexpr (FST) {
#pragma unroll
            for (int p = 0; p < RPT; ++p)
                if (xon[p]) cp_async16(dst + xoff[p], xsrc[p] + k0, xok[p]);
            const size_t wk = (size_t)chunk * KKT * NTILES * 32;
#pragma unroll
            for (int p = 0; p < WPT; ++p)
                if (WPT * TT == WP || (int)threadIdx.x + p * TT < WP) cp_async16(wdst + woff[p], wsrc[p] + wk, true);
        } else {
            for (int idx = threadIdx.x; idx < XP; idx += TT) {
                const int m = idx / (TPM * (KBK / 8)), rem = idx % (TPM * (KBK / 8)), i = rem / (KBK / 8), j = rem % (KBK / 8);
                if (i >= nblocks * 16) continue;   // a 16-member block without a member: never read
                const int r = rows_sh[i];
                const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * (XK ? xk : K) + k0 + j * 8;
                cp_async16(dst + ((size_t)m * TPM + i) * KXS + kswz(i, j) * 8, src, r >= 0);
            }
            for (int idx = threadIdx.x; idx < WP; idx += TT) {   // 16-byte pieces of the 128-byte tile words
                const int q = idx / (KKT * 8), kk = (idx / 8) % KKT, part = idx % 8;
                int wm, nt;
                ttile<MATS>(q, cb, wm, nt);
                const uint32_t* src = (wm ? T1 : T0) + (((size_t)chunk * KKT + kk) * NTILES + nt) * 32 + part * 4;
                cp_async16(wdst + ((size_t)q * KKT + kk) * 32 + part * 4, src, true);
            }
        }
    };
#pragma unroll
    for (int c = 0; c < NST - 1; ++c) {
        if (c < chunks) load(c);
        cp_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_wait<NST - 2>();
        __syncthreads();
        if (c + NST - 1 < chunks) load(c + NST - 1);
        cp_commit();
        const half* base = stage + (size_t)(c % NST) * SH;
        const half* xs = base + (size_t)(XR == TPM ? 0 : mat) * TPM * KXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)XR * KXS);
#pragma unroll
        for (int kk = 0; kk < KKT; ++kk) {
            uint32_t a[TMT][4];
#pragma unroll
            for (int i = 0; i < TMT; ++i) {
                uint32_t b0[2], b1[2];
                kdecode(ws[((size_t)(warp * TMT + i) * KKT + kk) * 32 + lane], lane, b0, b1);
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < TPM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldmatrix4(r, xs + (size_t)row * KXS + kswz(row, kk * 2 + ((lane >> 3) & 1)) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                    const bool upper = jb * 16 + 8 < cnt;   // members 8-15 of the block (else never stored)
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        if (upper) mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}

// The item a block runs: plan order, or `order`'s slot (tf_exl3_ovn_order); -1 when the block has none.
__device__ __forceinline__ int kitem(const int* __restrict__ counts, const int* __restrict__ order) {
    if ((int)blockIdx.y >= counts[0]) return -1;
    return order ? order[blockIdx.y] : (int)blockIdx.y;
}

// tgateup with V: 0 tmain and scalar stores, 1 ots0106's kmain and packed stores, 2 plus ots0127's hoisting.
// rm, rr, rwn, rwx: TF_GLM_OVN_X3_ROTEXPERT (exl3_ovn.cuh rot_share; rm 0: every expert N columns).
template <int XR, int V>
__device__ __forceinline__ void ogateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint64_t* __restrict__ Tg,
        const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd,
        int K, int N, int xdiv, float limit, int act_mode, const int* __restrict__ order, int rm, int rr, int rwn, int rwx,
        unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = kitem(counts, order);
    if (item < 0) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const int ncols = rot_share(e, N, rm, rr, rwn, rwx); // this expert's share; N its row stride in xd and the scales
    if (cb * 128 >= ncols) return; // a narrow share has no such column block (the down kernel never reads one)
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    if constexpr (V == 0) tmain<2, XR>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, ncols >> 4, cb, xdiv, reinterpret_cast<half*>(smem_raw), acc);
    else kmain<2, XR, V == 2>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, ncols >> 4, cb, xdiv, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    [[maybe_unused]] float sg[4], su[4], sd[4];   // V 2: the column's scales once a block (the same floats)
    if constexpr (V == 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sg[j] = __half2float(svh_g[(size_t)e * N + n + j]);
            su[j] = __half2float(svh_u[(size_t)e * N + n + j]);
            sd[j] = __half2float(suh_d[(size_t)e * N + n + j]);
        }
    }
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();   // the first half's rows are read
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float g_s, u_s, d_s;
                if constexpr (V == 2) {
                    g_s = sg[j]; u_s = su[j]; d_s = sd[j];
                } else {
                    g_s = __half2float(svh_g[(size_t)e * N + n + j]);
                    u_s = __half2float(svh_u[(size_t)e * N + n + j]);
                    d_s = __half2float(suh_d[(size_t)e * N + n + j]);
                }
                float gg = gv[j] * HAD_SCALE * g_s;
                float uu = uv[j] * HAD_SCALE * u_s;
                if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                float act = gg / (1.f + expf(-gg));
                if (act_mode == 0) act = bf16r(act);
                act *= uu;
                if (act_mode == 0) act = bf16r(act);
                v[j] = act * d_s;
            }
            fwht128(v, lane);
            half* o = xd + (size_t)p * N + n;
            if constexpr (V == 0) {
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            } else {   // exl3.cu:874-879
                const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
                const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
                uint2 packed;
                packed.x = *reinterpret_cast<const uint32_t*>(&o01);
                packed.y = *reinterpret_cast<const uint32_t*>(&o23);
                *reinterpret_cast<uint2*>(o) = packed;
            }
        }
    }
}

// The down epilogue: pair p to row p / slots, slot p % slots of `ys` slots a row (fp32, or bf16 when `bf16`).
// rm, rr, rwn, rwx: TF_GLM_OVN_X3_ROTEXPERT, the share's K of each Xd row K halves long (rm 0: every expert K).
template <int V>
__device__ __forceinline__ void odown(const half* __restrict__ Xd, const uint64_t* __restrict__ Td, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d, void* __restrict__ y,
        int K, int D, int slots, int ys, int bf16, const int* __restrict__ order, int rm, int rr, int rwn, int rwx,
        unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = kitem(counts, order);
    if (item < 0) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const int kcols = rot_share(e, K, rm, rr, rwn, rwx);
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    if constexpr (V == 0) tmain<1, TPM, true>(Xd, Xd, Td, Td, rows_sh, cnt, e, kcols, D >> 4, cb, 1, reinterpret_cast<half*>(smem_raw), acc, K);
    else kmain<1, TPM, V == 2, true>(Xd, Xd, Td, Td, rows_sh, cnt, e, kcols, D >> 4, cb, 1, reinterpret_cast<half*>(smem_raw), acc, K);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            const size_t row = (size_t)(p / slots) * ys + p % slots;
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                float o[4];
                if constexpr (V == 0) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
                } else {   // exl3.cu:914-922
                    const uint2 sv = *reinterpret_cast<const uint2*>(svh_d + (size_t)e * D + n);
                    const half2 s01 = *reinterpret_cast<const half2*>(&sv.x), s23 = *reinterpret_cast<const half2*>(&sv.y);
                    o[0] = v[0] * HAD_SCALE * __low2float(s01);
                    o[1] = v[1] * HAD_SCALE * __high2float(s01);
                    o[2] = v[2] * HAD_SCALE * __low2float(s23);
                    o[3] = v[3] * HAD_SCALE * __high2float(s23);
                }
                if (bf16) {   // TF_GLM_OVN_X3_YBF16 (a numerics trade): each slot value rounded once, as the NVFP4 rows
                    __nv_bfloat16* q = reinterpret_cast<__nv_bfloat16*>(y) + row * D + n;
                    if constexpr (V == 0) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) q[j] = __float2bfloat16_rn(o[j]);
                    } else {
                        const __nv_bfloat162 q01 = __floats2bfloat162_rn(o[0], o[1]), q23 = __floats2bfloat162_rn(o[2], o[3]);
                        uint2 packed;
                        packed.x = *reinterpret_cast<const uint32_t*>(&q01);
                        packed.y = *reinterpret_cast<const uint32_t*>(&q23);
                        *reinterpret_cast<uint2*>(q) = packed;
                    }
                } else {
                    float* q = reinterpret_cast<float*>(y) + row * D + n;
                    if constexpr (V == 0) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) q[j] = o[j];
                    } else {
                        *reinterpret_cast<float4*>(q) = make_float4(o[0], o[1], o[2], o[3]);
                    }
                }
            }
        }
    }
}

}  // namespace tf_exl3_ovn

// TF_GLM_OVN_X3_PROMPT_ORDER: production's prompt_order_kernel (exl3.cu:755-815); only which block runs which item.
extern "C" __global__ void __launch_bounds__(1024) tf_exl3_ovn_order(const int* __restrict__ items, const int* __restrict__ counts,
        int* __restrict__ order, int cap, int E) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)   // items are in expert order: the routed ones come first
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = items[3 * i];
        return (i > 0 && items[3 * (i - 1)] == e) || (i + 1 < n && items[3 * (i + 1)] == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += u;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const int nh = wsum[31], nl = n - nh;
    int rh = (warp ? wsum[warp - 1] : 0) + v - h;   // heavy items before lo
    for (int i = lo; i < hi; ++i) {
        if (heavy(i)) {   // heavy rank r sits at (r + 1/2) / nh; the light ones strictly before it
            const int r = rh++;
            const long long X = (2LL * r + 1) * nl;
            const int cl = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
            order[r + cl] = i;
        } else {   // light rank l at (l + 1/2) / nl; the heavy ones at or before it
            const int l = i - rh;
            const long long Y = (2LL * l + 1) * nh;
            const int ch = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
            order[l + ch] = i;
        }
    }
    for (int i = n + tid; i < all; i += 1024) order[i] = -1;
}

// Entries a variant: _0 the 32-k loop (static shared bytes), _1 and _2 the 64-k loops (dynamic, ksmem_bytes, opt-in).
#define TF_EXL3_OVN_GU_ARGS                                                                                                    \
    const half *__restrict__ Xg, const half *__restrict__ Xu, const uint64_t *__restrict__ Tg, const uint64_t *__restrict__ Tu, \
        const int *__restrict__ items, const int *__restrict__ counts, const int *__restrict__ members,                        \
        const half *__restrict__ svh_g, const half *__restrict__ svh_u, const half *__restrict__ suh_d, half *__restrict__ xd,  \
        int K, int N, int xdiv, float limit, int act_mode, const int *__restrict__ order, int rm, int rr, int rwn, int rwx
#define TF_EXL3_OVN_GU_CALL(V, smem)                                                                                    \
    if (xdiv > 1) tf_exl3_ovn::ogateup<TPM, V>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N,    \
                                              xdiv, limit, act_mode, order, rm, rr, rwn, rwx, smem, rows_sh);          \
    else tf_exl3_ovn::ogateup<2 * TPM, V>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, 1, limit, \
                                         act_mode, order, rm, rr, rwn, rwx, smem, rows_sh)
#define TF_EXL3_OVN_DN_ARGS                                                                                                \
    const half *__restrict__ Xd, const uint64_t *__restrict__ Td, const int *__restrict__ items,                          \
        const int *__restrict__ counts, const int *__restrict__ members, const half *__restrict__ svh_d, void *__restrict__ y, \
        int K, int D, int slots, int ys, int bf16, const int *__restrict__ order, int rm, int rr, int rwn, int rwx
#define TF_EXL3_OVN_DN_CALL(V, smem) \
    tf_exl3_ovn::odown<V>(Xd, Td, items, counts, members, svh_d, y, K, D, slots, ys, bf16, order, rm, rr, rwn, rwx, smem, rows_sh)

extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_ovn_gu_0(TF_EXL3_OVN_GU_ARGS) {
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    TF_EXL3_OVN_GU_CALL(0, smem_raw);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_ovn_gu_1(TF_EXL3_OVN_GU_ARGS) {
    extern __shared__ __align__(16) unsigned char ovn_smem[];
    __shared__ int rows_sh[TPM];
    TF_EXL3_OVN_GU_CALL(1, ovn_smem);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_ovn_gu_2(TF_EXL3_OVN_GU_ARGS) {
    extern __shared__ __align__(16) unsigned char ovn_smem[];
    __shared__ int rows_sh[TPM];
    TF_EXL3_OVN_GU_CALL(2, ovn_smem);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_ovn_down_0(TF_EXL3_OVN_DN_ARGS) {
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    TF_EXL3_OVN_DN_CALL(0, smem_raw);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_ovn_down_1(TF_EXL3_OVN_DN_ARGS) {
    extern __shared__ __align__(16) unsigned char ovn_smem[];
    __shared__ int rows_sh[TPM];
    TF_EXL3_OVN_DN_CALL(1, ovn_smem);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_ovn_down_2(TF_EXL3_OVN_DN_ARGS) {
    extern __shared__ __align__(16) unsigned char ovn_smem[];
    __shared__ int rows_sh[TPM];
    TF_EXL3_OVN_DN_CALL(2, ovn_smem);
}
#undef TF_EXL3_OVN_GU_ARGS
#undef TF_EXL3_OVN_GU_CALL
#undef TF_EXL3_OVN_DN_ARGS
#undef TF_EXL3_OVN_DN_CALL
