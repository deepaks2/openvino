// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimized GroupCorrelation kernel with SLM tiling for stereo matching.
//
// Key optimization: right features are loaded into SLM once per work-group
// tile, then reused across all disparities. This reduces global memory reads
// by ~MAX_DISPARITY/TILE_W fold compared to the reference kernel.
//
// Dispatch: gws = (NUM_GROUPS, H, ceil(W / TILE_W) * TILE_W)
//           lws = (1, 1, TILE_W)
// where TILE_W is the subgroup size (typically 16).
//
// SLM layout: right_slm[k][TILE_W + MAX_DISPARITY - 1]
//   stores right features for x range [x_tile_start - MAX_DISPARITY + 1, x_tile_start + TILE_W - 1]
//
// JIT constants: NUM_GROUPS, CHANNELS_PER_GROUP, MAX_DISPARITY, TILE_W, OUTPUT_WIDTH

KERNEL(group_correlation_opt)(
    const __global INPUT0_TYPE* left_feat,
    const __global INPUT1_TYPE* right_feat,
    __global OUTPUT_TYPE* output)
{
    const int g = (int)get_global_id(0);          // group index
    const int y = (int)get_global_id(1);          // spatial height
    const int x = (int)get_global_id(2);          // spatial width (may be OOB)
    const int lid_x = (int)get_local_id(2);       // local x within tile
    const int n = 0;

    // Tile start in x dimension
    const int x_tile_start = x - lid_x;

    // Early exit for out-of-bounds work-items (last tile may be partial)
    const bool valid = (x < OUTPUT_WIDTH);

    // SLM buffer for right features: CPG channels × (TILE_W + MAX_DISPARITY - 1) elements
    // We load the extended window of right features needed by all work-items in the tile.
#define SLM_STRIDE (TILE_W + MAX_DISPARITY - 1)
    __local INPUT1_TYPE right_slm[CHANNELS_PER_GROUP * SLM_STRIDE];

    // Load left features for this work-item into private registers
    float left_cache[CHANNELS_PER_GROUP];
    if (valid) {
        __attribute__((opencl_unroll_hint(CHANNELS_PER_GROUP)))
        for (int k = 0; k < CHANNELS_PER_GROUP; ++k) {
            left_cache[k] = (float)left_feat[INPUT0_GET_INDEX(n, g * CHANNELS_PER_GROUP + k, 0, y, x)];
        }
    }

    // Cooperatively load right features into SLM.
    // The window spans [x_tile_start - MAX_DISPARITY + 1, x_tile_start + TILE_W - 1]
    // Total elements per channel = TILE_W + MAX_DISPARITY - 1 = SLM_STRIDE
    // Each of TILE_W work-items loads ceil(SLM_STRIDE / TILE_W) elements per channel.
    for (int k = 0; k < CHANNELS_PER_GROUP; ++k) {
        for (int i = lid_x; i < SLM_STRIDE; i += TILE_W) {
            int src_x = x_tile_start - (MAX_DISPARITY - 1) + i;
            INPUT1_TYPE val = (INPUT1_TYPE)0;
            if (src_x >= 0 && src_x < OUTPUT_WIDTH) {
                val = right_feat[INPUT1_GET_INDEX(n, g * CHANNELS_PER_GROUP + k, 0, y, src_x)];
            }
            right_slm[k * SLM_STRIDE + i] = val;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (!valid) return;

    const float inv_cpg = 1.0f / (float)CHANNELS_PER_GROUP;

    // The position of this work-item's x=0 disparity in SLM is at offset:
    // (x - x_tile_start) + (MAX_DISPARITY - 1) = lid_x + MAX_DISPARITY - 1
    const int base_slm_offset = lid_x + (MAX_DISPARITY - 1);

    for (int d = 0; d < MAX_DISPARITY; ++d) {
        float sum = 0.0f;
        const int slm_x = base_slm_offset - d;  // always >= 0 and < SLM_STRIDE by construction
        __attribute__((opencl_unroll_hint(CHANNELS_PER_GROUP)))
        for (int k = 0; k < CHANNELS_PER_GROUP; ++k) {
            sum = fma(left_cache[k], (float)right_slm[k * SLM_STRIDE + slm_x], sum);
        }
        output[OUTPUT_GET_INDEX(n, g, d, y, x)] = (OUTPUT_TYPE)(sum * inv_cpg);
    }
}
