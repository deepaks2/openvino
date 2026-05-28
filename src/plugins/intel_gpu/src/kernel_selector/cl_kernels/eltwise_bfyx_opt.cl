// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimized eltwise kernel for bfyx layout with arbitrary pitches.
// Vectorizes in X dimension with tail handling.
// Supports inputs with different pitches from output (e.g., crop/slice views).

#include "include/batch_headers/fetch_data.cl"

KERNEL(eltwise_bfyx_opt)(
    INPUTS_DECLS
    __global OUTPUT_TYPE* output)
{
    const uint x_block = (uint)get_global_id(0);
    const uint y = (uint)get_global_id(1);
    const uint fb = (uint)get_global_id(2);
    const uint f = fb % F_SIZE;
    const uint b = fb / F_SIZE;

    const uint x = x_block * VEC_SIZE;

    // Compute base offsets using per-tensor pitches
    const uint out_base = OUT_OFFSET + b * OUT_B_PITCH + f * OUT_F_PITCH + y * OUT_Y_PITCH + x;
    const uint in0_base = IN0_OFFSET + b * IN0_B_PITCH + f * IN0_F_PITCH + y * IN0_Y_PITCH + x;
#if INPUTS_COUNT > 1
    const uint in1_base = IN1_OFFSET + b * IN1_B_PITCH + f * IN1_F_PITCH + y * IN1_Y_PITCH + x;
#endif

    if (x + VEC_SIZE <= X_SIZE) {
        // Full vector path - use vload for safe unaligned access
        MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) val0 = vload8(0, input0 + in0_base);
#if INPUTS_COUNT > 1
        MAKE_VECTOR_TYPE(INPUT1_TYPE, VEC_SIZE) val1 = vload8(0, input1 + in1_base);
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE) res = ELTWISE_OP(val0, val1);
#else
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE) res = val0;
#endif
        vstore8(res, 0, output + out_base);
    }
#if X_TAIL > 0
    else if (x < X_SIZE) {
        // Tail path: process remaining elements scalar
        for (uint i = 0; i < X_TAIL; i++) {
            INPUT0_TYPE val0 = input0[in0_base + i];
#if INPUTS_COUNT > 1
            INPUT1_TYPE val1 = input1[in1_base + i];
            output[out_base + i] = ELTWISE_OP(val0, val1);
#else
            output[out_base + i] = val0;
#endif
        }
    }
#endif
}
