// Ours: TF_GLM_OVN_DRAW1's candidate rows for sample.cu's draw (included after tf_sample::draw).
#pragma once

// Row m's top RULE[m].top_k of this rank's V columns of row SRC[m] in the draw's order (logits / T, key desc, id asc);
// OUT row m holds K (bf16 bits, LO + column) word pairs, the slots past them id 0xFFFFFFFF.
extern "C" __global__ void __launch_bounds__(1024) tf_draw_cands(const __nv_bfloat16* L, unsigned V, unsigned LO,
                                                                 const tf_sample::Rule* rule, const unsigned* SRC,
                                                                 unsigned K, unsigned* OUT) {
    __shared__ tf_sample::Shared s;
    const unsigned m = blockIdx.x, t = threadIdx.x;
    const tf_sample::Rule rl = rule[m];
    const tf_sample::Row r{L + static_cast<size_t>(SRC[m]) * V, V, rl.inv_t, t, t & 31u, t >> 5};
    const tf_sample::Top top = tf_sample::top_keys(r, rl.top_k, s);
    if (t == 0) s.fill[0] = 0u;
    __syncthreads();
    unsigned* o = OUT + static_cast<size_t>(m) * K * 2u;
    for (unsigned i = t; i < V; i += tf_sample::TG) {
        const unsigned k = tf_sample::key(r.v(i));
        if (k > top.key || (k == top.key && i <= top.idcut)) {
            const unsigned at = atomicAdd(&s.fill[0], 1u);
            o[2u * at] = __bfloat16_as_ushort(r.L[i]);
            o[2u * at + 1u] = LO + i;
        }
    }
    __syncthreads();
    for (unsigned j = s.fill[0] + t; j < K; j += tf_sample::TG) {
        o[2u * j] = 0xFF80u;
        o[2u * j + 1u] = 0xFFFFFFFFu;
    }
}

// Row m of FULL (bf16 [rows][V]) all -inf, then every rank's candidates of row m (CAND: [world][rows][K] word pairs)
// at their columns, so the draw reads the gathered top k at their own bits.
extern "C" __global__ void __launch_bounds__(1024) tf_draw_place(const unsigned* CAND, unsigned world, unsigned rows,
                                                                 unsigned K, __nv_bfloat16* FULL, unsigned V) {
    const unsigned m = blockIdx.x;
    const __nv_bfloat16 none = __ushort_as_bfloat16(static_cast<unsigned short>(0xFF80u));
    __nv_bfloat16* row = FULL + static_cast<size_t>(m) * V;
    for (unsigned i = threadIdx.x; i < V; i += blockDim.x) row[i] = none;
    __syncthreads();
    for (unsigned j = threadIdx.x; j < world * K; j += blockDim.x) {
        const unsigned* w = CAND + ((static_cast<size_t>(j / K) * rows + m) * K + j % K) * 2u;
        if (w[1] < V) row[w[1]] = __ushort_as_bfloat16(static_cast<unsigned short>(w[0]));
    }
}
