// TF_GLM_OVN_SEG_DSA: tf_glm_dsa_tokens a listed segment a grid y (glm_ops.cu includes this after dsa_tokens).
#pragma once

#include "glm_seg_table.cuh"

// Each segment's rows' tokens and counts at its own rows, from its pools rows, through its page table.
extern "C" __global__ void tf_glm_dsa_tokens_seg(const glm_seg::Seg* __restrict__ segs, const int* __restrict__ list,
                                                 const glm_seg::Cache* __restrict__ cache,
                                                 const long long* __restrict__ pools, int K, int* __restrict__ tokens,
                                                 int* __restrict__ counts) {
    const int s = glm_seg::pick(list, blockIdx.y);
    const glm_seg::Seg g = segs[s];
    if (static_cast<long long>(blockIdx.x) >= g.rows) return;
    const glm_seg::Cache c = cache[s];
    dsa_tokens(pools + g.row0 * K, g.pos, K, tokens + g.row0 * (4 * K + 3), counts + g.row0, c.pt, c.page_rows);
}
