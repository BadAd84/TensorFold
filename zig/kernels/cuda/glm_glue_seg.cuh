// TF_GLM_OVN_SEG_DSA: the DSA latent write, index update, chunks, merge, scores and selection for every segment of a
// shared window in one launch each (production's latent.seg_* / sparse.seg_*, families/glm5_next/cuda/segments.py): a
// block reads its segment (glm_seg_table.cuh), returns past that segment's own grid, else runs the unchanged body.
#pragma once

#include "glm_glue_latent.cuh"
#include "glm_glue_index.cuh"
#include "glm_glue_attn.cuh"
#include "glm_seg_table.cuh"

namespace glm_seg {

using bf16 = __nv_bfloat16;

// tf_glm_chunks_* over the listed segments: blockIdx.z = list entry x nch + chunk; dense rows stop at their own nch.
template <int HB, bool FP8>
__device__ __forceinline__ void chunks(const Seg* segs, const int* list, const Cache* cache, const bf16* qa,
                                       const int* tok, const int* cnt, float* po, float* pm, float* pl,
                                       int64_t heads, int64_t w, float scale, int sparse, int64_t pn, int nch) {
  const int s = pick(list, (int)(blockIdx.z / nch));
  const int64_t c = blockIdx.z % nch;
  const Seg g = segs[s];
  if ((int64_t)blockIdx.x >= g.rows || c >= (sparse ? (int64_t)nch : g.nch)) return;
  const Cache k = cache[s];
  const int64_t p = g.row0 * pn * heads;  // the segment's partials: pn chunk slots a row from its first row's
  glm_attn::chunks_wide<HB, FP8>(c, qa + g.row0 * heads * glm_attn::LW, reinterpret_cast<const bf16*>(k.lc), g.pos,
                                 sparse ? tok + g.row0 * w : tok, sparse ? cnt + g.row0 : cnt, po + p * glm_attn::LW,
                                 pm + p, pl + p, g.rows, heads, w, scale, sparse, k.pt, k.ps,
                                 FP8 ? reinterpret_cast<const float*>(k.ls) : nullptr);
}

}  // namespace glm_seg

// tf_glm_lat_write a segment a grid y: row blockIdx.x of segment blockIdx.y into its cache at its position.
extern "C" __global__ void tf_glm_lat_write_seg(const glm_seg::Seg* segs, const glm_seg::Cache* cache,
                                                const __nv_bfloat16* lat, int64_t lat_stride, int64_t lw) {
  const glm_seg::Seg g = segs[blockIdx.y];
  if ((int64_t)blockIdx.x >= g.rows) return;
  const glm_seg::Cache k = cache[blockIdx.y];
  glm_latent::lat_write(lat + g.row0 * lat_stride, lat_stride, reinterpret_cast<__nv_bfloat16*>(k.lc), g.pos, lw, k.pt,
                        k.ps);
}

// tf_glm_lat_write_fp8 a segment a grid y.
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_lat_write_fp8_seg(const glm_seg::Seg* segs, const glm_seg::Cache* cache, const __nv_bfloat16* lat,
                             int64_t stride) {
  const glm_seg::Seg g = segs[blockIdx.y];
  if ((int64_t)blockIdx.x >= g.rows) return;
  const glm_seg::Cache k = cache[blockIdx.y];
  glm_latent::lat_write_fp8(lat + g.row0 * stride, stride, reinterpret_cast<unsigned char*>(k.lc),
                            reinterpret_cast<float*>(k.ls), g.pos, k.pt, k.ps);
}

// tf_glm_index_write a segment a grid y (D 128: the gate rows' stride).
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_index_write_seg(const glm_seg::Seg* segs, const glm_seg::Cache* cache, const __nv_bfloat16* kr,
                           int64_t k_stride, const float* gr, const __nv_bfloat16* lnw, const __nv_bfloat16* lnb,
                           float eps) {
  const glm_seg::Seg g = segs[blockIdx.y];
  if ((int64_t)blockIdx.x >= g.rows) return;
  const glm_seg::Cache k = cache[blockIdx.y];
  glm_index::index_write_body(kr + g.row0 * k_stride, k_stride, gr + g.row0 * 128, lnw, lnb,
                              reinterpret_cast<__nv_bfloat16*>(k.ik), reinterpret_cast<__nv_bfloat16*>(k.ig), g.pos,
                              eps);
}

// tf_glm_pool_keys a segment a grid y: a segment's grid is its rows / 4 + 2 blocks.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_pool_keys_seg(const glm_seg::Seg* segs, const glm_seg::Cache* cache, const __nv_bfloat16* ape) {
  const glm_seg::Seg g = segs[blockIdx.y];
  if ((int64_t)blockIdx.x >= g.rows / 4 + 2) return;
  const glm_seg::Cache k = cache[blockIdx.y];
  glm_index::pool_keys_body(reinterpret_cast<const __nv_bfloat16*>(k.ik), reinterpret_cast<const __nv_bfloat16*>(k.ig),
                            ape, reinterpret_cast<__nv_bfloat16*>(k.pk), g.pos, g.rows, k.pt, k.ps);
}

// The chunk kernels over the listed segments, in glm_attn::wide_smem<HB>() bytes of shared memory.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_seg(const glm_seg::Seg* segs, const int* list, const glm_seg::Cache* cache,
                         const __nv_bfloat16* qa, const int* tok, const int* cnt, float* po, float* pm, float* pl,
                         int64_t heads, int64_t w, float scale, int sparse, int64_t pn, int nch) {
  glm_seg::chunks<16, false>(segs, list, cache, qa, tok, cnt, po, pm, pl, heads, w, scale, sparse, pn, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_seg(const glm_seg::Seg* segs, const int* list, const glm_seg::Cache* cache,
                         const __nv_bfloat16* qa, const int* tok, const int* cnt, float* po, float* pm, float* pl,
                         int64_t heads, int64_t w, float scale, int sparse, int64_t pn, int nch) {
  glm_seg::chunks<32, false>(segs, list, cache, qa, tok, cnt, po, pm, pl, heads, w, scale, sparse, pn, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_seg(const glm_seg::Seg* segs, const int* list, const glm_seg::Cache* cache,
                             const __nv_bfloat16* qa, const int* tok, const int* cnt, float* po, float* pm, float* pl,
                             int64_t heads, int64_t w, float scale, int sparse, int64_t pn, int nch) {
  glm_seg::chunks<16, true>(segs, list, cache, qa, tok, cnt, po, pm, pl, heads, w, scale, sparse, pn, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_seg(const glm_seg::Seg* segs, const int* list, const glm_seg::Cache* cache,
                             const __nv_bfloat16* qa, const int* tok, const int* cnt, float* po, float* pm, float* pl,
                             int64_t heads, int64_t w, float scale, int sparse, int64_t pn, int nch) {
  glm_seg::chunks<32, true>(segs, list, cache, qa, tok, cnt, po, pm, pl, heads, w, scale, sparse, pn, nch);
}

// tf_glm_merge a listed segment a grid z: dense rows merge their own nch chunks, sparse rows `nch` (count 0 skipped).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_merge_seg(const glm_seg::Seg* segs, const int* list, const float* po, const float* pm, const float* pl,
                     __nv_bfloat16* out, const int* cnt, int64_t heads, int64_t pn, int nch, int sparse) {
  const glm_seg::Seg g = segs[glm_seg::pick(list, blockIdx.z)];
  if ((int64_t)blockIdx.x >= g.rows) return;
  const int64_t p = g.row0 * pn * heads;
  glm_attn::merge_body(po + p * glm_attn::LW, pm + p, pl + p, out + g.row0 * heads * glm_attn::LW,
                       sparse ? cnt + g.row0 : cnt, g.rows, heads, sparse ? nch : (int)g.nch, sparse);
}

// tf_glm_scores (one pool block a CTA, a decode window's loop of 1) a listed segment a grid z, into its scores rows.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_scores_seg(const glm_seg::Seg* segs, const int* list, const glm_seg::Cache* cache,
                      const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, float* scores, float scale,
                      float wscale, int64_t heads) {
  const int s = glm_seg::pick(list, blockIdx.z);
  const glm_seg::Seg g = segs[s];
  if ((int64_t)blockIdx.x >= g.rows || (int64_t)blockIdx.y >= g.nb) return;
  const glm_seg::Cache k = cache[s];
  glm_attn::scores_body(qi + g.row0 * heads * 128, wts + g.row0 * w_stride, w_stride,
                        reinterpret_cast<const __nv_bfloat16*>(k.pk), scores + g.soff, g.pos, g.np, g.nb, scale,
                        wscale, heads, 1, k.pt, k.ps);
}

// tf_glm_select_rows and _wide a listed segment a grid y: each row's K best pools into its pools row.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_select_rows_seg(const glm_seg::Seg* segs, const int* list, const float* scores, int64_t* pools, int k) {
  const glm_seg::Seg g = segs[glm_seg::pick(list, blockIdx.y)];
  if ((int64_t)blockIdx.x >= g.rows) return;
  glm_index::select_row<128>(scores + g.soff, pools + g.row0 * k, g.np, g.pos, k);
}
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_select_rows_wide_seg(const glm_seg::Seg* segs, const int* list, const float* scores, int64_t* pools,
                                int k) {
  const glm_seg::Seg g = segs[glm_seg::pick(list, blockIdx.y)];
  if ((int64_t)blockIdx.x >= g.rows) return;
  glm_index::select_row<1024>(scores + g.soff, pools + g.row0 * k, g.np, g.pos, k);
}
