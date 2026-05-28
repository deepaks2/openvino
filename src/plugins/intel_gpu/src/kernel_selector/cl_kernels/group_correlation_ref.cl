// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimized GroupCorrelation kernel for stereo matching cost-volume computation.
//
// Dispatch: gws = (NUM_GROUPS, H, W)
// Each work item computes all MAX_DISPARITY outputs for one (g, h, w) position.
// The K left-feature values are loaded once into private float registers and
// reused across all MAX_DISPARITY inner iterations, eliminating the MAX_DISPARITY-fold
// redundant global memory reads of the original per-disparity dispatch.
//
// Inputs (bfzyx, z=1 — promoted by graph optimizer):
//   INPUT0: left  feature map (N, C, 1, H, W)
//   INPUT1: right feature map (N, C, 1, H, W)
// Output (bfzyx):
//   OUTPUT: cost volume (N, NUM_GROUPS, MAX_DISPARITY, H, W)
//
// out[0, g, d, h, w] = mean_k( left[0, g*CPG+k, 0, h, w]
//                                * right[0, g*CPG+k, 0, h, w-d] )  if w >= d
//                    = 0  otherwise
//
// JIT constants: NUM_GROUPS, CHANNELS_PER_GROUP, MAX_DISPARITY

KERNEL(group_correlation_ref)(
    const __global INPUT0_TYPE* left_feat,
    const __global INPUT1_TYPE* right_feat,
    __global OUTPUT_TYPE* output)
{
    const int g = (int)get_global_id(0);  // group index
    const int y = (int)get_global_id(1);  // spatial height
    const int x = (int)get_global_id(2);  // spatial width
    const int n = 0;

    // Load left features for this (g, h, w) into private float registers.
    // These are reused across all MAX_DISPARITY iterations, so we only hit
    // global memory once regardless of how large MAX_DISPARITY is.
    float left_cache[CHANNELS_PER_GROUP];
    __attribute__((opencl_unroll_hint(CHANNELS_PER_GROUP)))
    for (int k = 0; k < CHANNELS_PER_GROUP; ++k) {
        left_cache[k] = (float)left_feat[INPUT0_GET_INDEX(n, g * CHANNELS_PER_GROUP + k, 0, y, x)];
    }

    const float inv_cpg = 1.f / (float)CHANNELS_PER_GROUP;

    // Split on the boundary: for x >= MAX_DISPARITY-1 every disparity is valid,
    // so we can eliminate the per-disparity branch entirely for ~70% of threads.
    if (x >= MAX_DISPARITY - 1) {
        // All disparities valid — branch-free inner loops.
        for (int d = 0; d < MAX_DISPARITY; ++d) {
            float sum = 0.f;
            __attribute__((opencl_unroll_hint(CHANNELS_PER_GROUP)))
            for (int k = 0; k < CHANNELS_PER_GROUP; ++k) {
                sum = fma(left_cache[k],
                          (float)right_feat[INPUT1_GET_INDEX(n, g * CHANNELS_PER_GROUP + k, 0, y, x - d)],
                          sum);
            }
            output[OUTPUT_GET_INDEX(n, g, d, y, x)] = (OUTPUT_TYPE)(sum * inv_cpg);
        }
    } else {
        // Boundary region: disparities where x - d < 0 produce zero.
        for (int d = 0; d < MAX_DISPARITY; ++d) {
            const int x_r = x - d;
            float sum = 0.f;
            if (x_r >= 0) {
                __attribute__((opencl_unroll_hint(CHANNELS_PER_GROUP)))
                for (int k = 0; k < CHANNELS_PER_GROUP; ++k) {
                    sum = fma(left_cache[k],
                              (float)right_feat[INPUT1_GET_INDEX(n, g * CHANNELS_PER_GROUP + k, 0, y, x_r)],
                              sum);
                }
                sum *= inv_cpg;
            }
            output[OUTPUT_GET_INDEX(n, g, d, y, x)] = (OUTPUT_TYPE)sum;
        }
    }
}
