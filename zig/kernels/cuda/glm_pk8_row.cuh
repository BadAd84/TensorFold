// TF_GLM_OVN_PK8: a pooled key as production's kv8 row (prod-src families/glm5_next/cuda/kv8.py store_row, SHIFT 8):
// 128 E4M3 codes then the row's power-of-two FP32 scale, 132 bytes a pool; a page part holds 128 such rows in order.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace glm_pk8 {

constexpr int D = 128, ROW = D + 4;  // values a pooled key, bytes a kv8 row

// Byte offset of pool p's row: in order (pt null), or row p % 128 of page pt[p / 128], `ps` bytes a page.
__device__ __forceinline__ int64_t row_at(const int* pt, int64_t ps, int64_t p) {
  return pt ? (int64_t)pt[p / 128] * ps + (p % 128) * ROW : p * ROW;
}

// kv8.scale_of: amax -> 2^(ceil(log2 amax) - 8) and its reciprocal, the biased exponent held within 1 .. 254.
__device__ __forceinline__ void scale_of(float amax, float& s, float& inv) {
  const int bits = __float_as_int(amax);
  int e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
  e = min(max(e - 8, 1), 254);
  s = __int_as_float(e << 23);
  inv = __int_as_float((254 - e) << 23);
}

// Two E4M3 codes (low byte first) -> their bf16 values (low first), exactly: e4m3 -> f16 -> f32 -> bf16.
__device__ __forceinline__ uint32_t bf16x2_of(uint32_t two) {
  const __nv_fp8x2_storage_t c = (__nv_fp8x2_storage_t)(two & 0xffffu);
  const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(c, __NV_E4M3)));
  const __nv_bfloat162 b = __floats2bfloat162_rn(v.x, v.y);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// Codes c .. c + 7 of a row (4-byte aligned: rows are 132 bytes) as two words; zeros where !ok.
__device__ __forceinline__ uint2 codes8(const unsigned char* row, int c, bool ok) {
  if (!ok) return make_uint2(0u, 0u);
  return make_uint2(*reinterpret_cast<const uint32_t*>(row + c), *reinterpret_cast<const uint32_t*>(row + c + 4));
}

// Eight codes as the 16 bytes of their 8 bf16 values (zero codes: zeros).
__device__ __forceinline__ uint4 bf16x8_of(uint2 w) {
  return make_uint4(bf16x2_of(w.x), bf16x2_of(w.x >> 16), bf16x2_of(w.y), bf16x2_of(w.y >> 16));
}

// A row's FP32 scale (1 where !ok: never folded into a score that is kept).
__device__ __forceinline__ float row_scale(const unsigned char* row, bool ok) {
  return ok ? *reinterpret_cast<const float*>(row + D) : 1.0f;
}

}  // namespace glm_pk8
