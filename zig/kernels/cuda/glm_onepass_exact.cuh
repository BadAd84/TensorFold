// TF_GLM_OVN_ONEPASS_EXACT: prompt rows' sparse latent attention in one pass a row, bit-identical to chunks + merge.
#pragma once
#include "glm_glue_attn_ovn.cuh"

// tf_glm_onepass_exact runs glm_attn_ovn::chunks<32,true,true>, proven equal to chunks_32_fp8 + merge by the check.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_onepass_exact(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* tok,
                         const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t w, float scale,
                         const int* pt, int64_t pstep, int nch) {
  // one block a row (grid rows x 1), 256 threads, wide_smem<32>(); sparse prompt rows, so pos is unused and null
  glm_attn_ovn::chunks<32, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), scales, nullptr, tok, cnt,
                                       nullptr, nullptr, nullptr, out, nch, rows, heads, w, scale, 1, pt, pstep);
}
