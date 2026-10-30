// TF_GLM_OVN_SEG_DSA: a shared decode window's segment table (cuda_dsa_seg.zig's Seg and Cache, 64 bytes each).
#pragma once

namespace glm_seg {

// A stream's rows of the window: first row, rows, position (i32 on device), dense chunks, pools, score blocks, and its
// scores' first float in the segmented scores buffer.
struct Seg {
  long long row0, rows;
  const int* pos;
  long long nch, np, nb, soff, pad;
};

// That stream's caches at one DSA layer (dsa_block.Caches): latent rows, scales, index keys, gates, pooled keys, the
// page table (null: a slab), bytes from page to page and latent rows a page part holds.
struct Cache {
  unsigned long long lc, ls, ik, ig, pk;
  const int* pt;
  long long ps, page_rows;
};

// Entry i of a launch's segment list (null: every segment in order).
__device__ __forceinline__ int pick(const int* list, int i) { return list ? list[i] : i; }

}  // namespace glm_seg
