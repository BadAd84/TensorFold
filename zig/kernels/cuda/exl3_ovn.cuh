// Speed-hunt decode variants of the grouped EXL3 experts (TF_GLM_OVN_X3_*): the same chains and sums, so the same bits.
// The 16-byte staged loads follow TensorFold's glm5_next/cuda/exl3.cu dec_kernel LD 1 (Apache-2.0; jayleaton 0580).
#pragma once

namespace tf_exl3 {

// The variants' launch arguments after GroupedArgs; the fields must stay in exl3_experts.zig's runWith add order.
struct OvnTail {
    const uint32_t* tb0; // the gate (or down) trellis of every expert, expert e at tb0 + e * ew (TF_GLM_OVN_X3_NC)
    const uint32_t* tb1; // the up trellis
    long long ew; // 32-bit words an expert's trellis
    const half* svh; // fp16 [E, N], the fused down epilogue's output scales (TF_GLM_OVN_X3_FUSEDOWN)
    float* y; // fp32 [P, N], the fused down epilogue's slot rows
    int xrow; // X holds one rotated row a token: pair p reads row p / slots (TF_GLM_OVN_X3_XROW)
    int fuse; // the down launch writes y through tf_exl3_down_combine's per-slot arithmetic, no Z
    long long rcw; // 32-bit words a share column (TF_GLM_OVN_X3_ROTEXPERT: expert e at tb + rcw * rot_before(e))
    int rm, rr; // the experts e with e % rm == rr hold the wide share here (rm 0: every expert alike, `n` or `k`)
    int rwn, rwx; // a narrow share's columns; the wide share has rwx more
    int rk; // 1: the share is the K axis (down), 0: the N axis (gate and up)
};

struct OvnArgs {
    GroupedArgs g;
    OvnTail t;
};

// TF_GLM_OVN_X3_ROTEXPERT: expert e's share columns (exl3_experts.zig Rot.cols); `full` when off.
__device__ __forceinline__ int rot_share(int e, int full, int m, int r, int wn, int wx) {
    return m ? wn + (e % m == r ? wx : 0) : full;
}
__device__ __forceinline__ int rot_cols(const OvnTail& t, int e, int full) { return rot_share(e, full, t.rm, t.rr, t.rwn, t.rwx); }
// Share columns of the experts before e laid back to back by the loader (exl3_experts.zig Rot.before).
__device__ __forceinline__ long long rot_before(const OvnTail& t, int e) {
    return (long long)e * t.rwn + (long long)t.rwx * ((e + t.rm - 1 - t.rr) / t.rm);
}

// A narrow share's missing column block of gate/up: zero partials for its members, so the epilogue reads zeros.
template <int NT>
__device__ __forceinline__ void rot_zero(const GroupedArgs& a, const int* rows_sh, int mat, int split, int nt0) {
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += blockDim.x) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r >= 0) a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = 0.f;
    }
}

// 16 bytes of trellis words through the read-only path without L1 allocation (exl3.cu ldg_nc_v4).
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// A 4-bit lane's two words from one load: word lane + 31 by shuffle, word lane loaded (lane_start<8>: bit 20 on).
__device__ __forceinline__ void words_of(uint32_t word, int lane, uint32_t (&w)[2]) {
    static_assert(lane_words<8>() == 2 && tile_words<8>() == 32, "4-bit tiles: two words a lane, 32 a tile");
    w[0] = __shfl_sync(0xffffffffu, word, (lane + 31) & 31);
    w[1] = word;
}

// tf_exl3_down_combine's per-slot arithmetic on a block's warp sums (sk 1): 0 + sum, rotated, * svh, into y's slot row.
template <int NT, int W>
__device__ __forceinline__ void fused_down(const float (&red)[W][16][NT * 16], const int* rows_sh, int e, int nt0,
                                           int n_all, const half* svh, float* y, int warp, int lane) {
    static_assert(NT * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    const int n = nt0 * 16 + 4 * lane;
    for (int i = warp; i < 16; i += W) {
        const int p = rows_sh[i];
        if (p < 0) continue;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = red[0][i][4 * lane + j];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][i][4 * lane + j];
            float z = 0.f;
            z += s;
            v[j] = z;
        }
        fwht128(v, lane);
        float* o = y + (size_t)p * n_all + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh[(size_t)e * n_all + n + j]);
    }
}

// grouped_body<CB_MCG, 8, 4> (non-standard plans, 4-bit) with production's loads: the count, the expert and the member
// codes in one round trip, PD k steps of 16-byte trellis loads issued before the member barrier and kept PD ahead,
// staged through the warp's 1 KB of `red`. Every mma, warp sum and store is grouped_body's, in its order.
template <int PD>
__device__ __forceinline__ void nc_body(const OvnArgs o, int u, int nblock, int zz) {
    constexpr int NT = 8, W = 4, NV = NT / 4, TW = 32;
    static_assert(PD >= 1 && PD <= 2, "PD 1 or 2");
    const GroupedArgs& a = o.g;
    const int MT = (a.maxm + 15) / 16;
    const int mtile = zz % MT;
    const int split = (zz / MT) % a.sk;
    const int mat = zz / MT / a.sk;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nt0 = nblock * NT;
    const int total = a.ucount[0];
    if (u >= total) return;
    const int e = a.standard ? a.uids[3 * u] : a.uids[u];
    if (a.standard && e == a.skip) return; // the shared item: no routed output, no trellis past the last expert
    const int m = mtile * 16 + (int)threadIdx.x;
    const int count = a.standard ? a.uids[3 * u + 2] : a.maxm;
    const int first = a.standard ? a.uids[3 * u + 1] : u * a.maxm;
    int code = -1;
    if (threadIdx.x < 16 && m < count) code = a.members[first + m];
    const int ncols = o.t.rm && !o.t.rk ? rot_cols(o.t, e, a.n) : a.n; // TF_GLM_OVN_X3_ROTEXPERT: this expert's share
    const int NTILES = a.ns > 0 ? a.ns : (ncols >> 4);
    const int per_warp = ((o.t.rm && o.t.rk ? rot_cols(o.t, e, a.k) : a.k) >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const bool past = nt0 * 16 >= ncols; // a narrow share has no such column block: zeros, and no loads
    const size_t kstep = (size_t)NTILES * TW;
    const size_t at = o.t.rm ? (size_t)rot_before(o.t, e) * o.t.rcw : (size_t)e * o.t.ew;
    const uint32_t* tbase = (mat ? o.t.tb1 : o.t.tb0) + at + (size_t)kt0 * kstep + (size_t)nt0 * TW;
    uint4 rv[PD][NV];
    if (!past)
#pragma unroll
        for (int d = 0; d < PD; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = a.standard ? code : (code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1);
    __syncthreads();
    if (rows_sh[0] < 0) return; // members come first, so this tile is empty
    if (past) return rot_zero<NT>(a, rows_sh, mat, split, nt0);
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (o.t.xrow ? r0 / a.slots : r0), xr1 = r1 < 0 ? 0 : (o.t.xrow ? r1 / a.slots : r1);
    const half* X = mat ? a.x1 : a.x0;
    const half* x0 = X + (size_t)xr0 * a.k + 2 * t;
    const half* x1 = X + (size_t)xr1 * a.k + 2 * t;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    uint32_t ra[PD][4];
#pragma unroll
    for (int d = 0; d < PD; ++d) {
        const int k = (kt0 + d) * 16;
        ra[d][0] = r0 >= 0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
        ra[d][1] = r1 >= 0 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
        ra[d][2] = r0 >= 0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
        ra[d][3] = r1 >= 0 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
    }
    uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
    for (int kb = 0; kb < per_warp; kb += PD) {
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int s = kb + d;
            const bool more = s + PD < per_warp;
            uint32_t av[4], words[NT];
#pragma unroll
            for (int c = 0; c < 4; ++c) av[c] = ra[d][c];
            __syncwarp(); // every lane has read the previous step back
#pragma unroll
            for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
            if (more) { // the slot's registers are free again
#pragma unroll
                for (int v = 0; v < NV; ++v)
                    rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);
                const int k = (kt0 + s + PD) * 16;
                ra[d][0] = r0 >= 0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
                ra[d][1] = r1 >= 0 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
                ra[d][2] = r0 >= 0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
                ra[d][3] = r1 >= 0 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
            }
            __syncwarp();
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t ww[2], b0[2], b1[2];
                words_of(words[i], lane, ww);
                decode_lane<8, CB_MCG>(ww, lane, b0, b1);
                mma16816(acc[i][0], av, b0);
                mma16816(acc[i][1], av, b1);
            }
        }
    }
    __syncwarp(); // the staging area becomes red[warp] below

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    if (o.t.fuse) {
        fused_down<NT, W>(red, rows_sh, e, nt0, a.n, o.t.svh, o.t.y, warp, lane);
        return;
    }
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) { // grouped_body's store
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
    }
}

}  // namespace tf_exl3
