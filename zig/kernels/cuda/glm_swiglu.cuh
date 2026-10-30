// glue._swiglu's value of one column, shared by glm_glue's swiglu kernels and glm_mm's fused split-K reduce.
#pragma once

#include "glm_triton.cuh"

namespace glm_moe {

__device__ __forceinline__ float swiglu1(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, 0.0f - limit), limit);
  const float s = glm_triton::rbf(glm_triton::div_full(g, glm_triton::exp_approx(0.0f - g) + 1.0f));
  return glm_triton::rbf(__fmul_rn(u, s));
}

}  // namespace glm_moe
