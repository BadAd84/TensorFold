// TF_GLM_OVN_ONEPASS_PIPE measurement variants: exact (all switches on) equals the fused kernel; singles time each.
#include "glm_onepass_pipe.cuh"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#define PIPE_CALL(DEC, SCA, CHU) glm_onepass_pipe::onepass_pipe<DEC, SCA, CHU>(QA, LC, LS, PR, TOK, W, CNT, OUT, H, scale)
#else
#define PIPE_CALL(DEC, SCA, CHU) __trap()
#endif

#define PIPE_VARIANT(NAME, DEC, SCA, CHU)                                                                            \
    extern "C" __global__ void __launch_bounds__(glm_onepass_pipe::NT, 1)                                           \
        NAME(const __nv_bfloat16* __restrict__ QA, const uint8_t* __restrict__ LC, const float* __restrict__ LS,     \
             unsigned PR, const int32_t* __restrict__ TOK, int W, const int32_t* __restrict__ CNT,                   \
             __nv_bfloat16* __restrict__ OUT, int H, float scale) {                                                 \
        PIPE_CALL(DEC, SCA, CHU);                                                                                    \
    }

PIPE_VARIANT(tf_glm_onepass_pipe_off, false, false, false)
PIPE_VARIANT(tf_glm_onepass_pipe_decode, true, false, false)
PIPE_VARIANT(tf_glm_onepass_pipe_scale, false, true, false)
PIPE_VARIANT(tf_glm_onepass_pipe_chunk, false, false, true)
PIPE_VARIANT(tf_glm_onepass_pipe_exact, true, true, true)
