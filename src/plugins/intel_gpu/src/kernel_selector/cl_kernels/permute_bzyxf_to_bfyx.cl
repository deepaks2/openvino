// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Optimized permute kernel for 5D bzyxf -> 4D bfyx with dimension merge.
// Pattern: output[b][y_in][f_in + x_in * F_SIZE][z_in] = input[b][f_in][z_in][y_in][x_in]
// Uses SLM-based tiling: coalesced reads along x, coalesced writes along z.

#include "include/batch_headers/fetch_data.cl"

#define INPUTVTYPE CAT(INPUT0_TYPE, VEC_SIZE)
#define VLOAD CAT(vload, VEC_SIZE)
#define VSTORE CAT(vstore, VEC_SIZE)

REQD_SUB_GROUP_SIZE(X_TILE)
__attribute__((reqd_work_group_size(X_TILE, Z_TILE, 1)))
KERNEL(permute_bzyxf_to_bfyx)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
    // SLM for tile transpose
    // Layout: slm[z_local][x_local * F_SIZE + f], with padding to avoid bank conflicts
    __local INPUT0_TYPE slm[SLM_TOTAL_SIZE];

    const uint lid_x = get_local_id(0);  // 0..X_TILE-1 (fast dim, subgroup lane)
    const uint lid_z = get_local_id(1);  // 0..Z_TILE-1

    const uint x_tile_id = get_group_id(0);  // which x tile
    const uint z_tile_id = get_group_id(1);  // which z tile
    const uint y = get_global_id(2);         // y index (and batch combined)

    const uint x = x_tile_id * X_TILE + lid_x;
    const uint z = z_tile_id * Z_TILE + lid_z;

    // ---- PHASE 1: Read from input (coalesced along x within subgroup) ----
    // Input bzyxf physical offset: z * Y_SIZE * X_SIZE * F_SIZE + y * X_SIZE * F_SIZE + x * F_SIZE
    const uint in_offset = z * (Y_SIZE * X_SIZE * F_SIZE) + y * (X_SIZE * F_SIZE) + x * F_SIZE;

    // Vectorized load of all F_SIZE feature values at this (z, y, x) position
    INPUTVTYPE vals = VLOAD(0, &input[in_offset]);

    // Store to SLM: row = lid_z, column = lid_x * F_SIZE
    const uint slm_wr_offset = lid_z * SLM_ROW_STRIDE + lid_x * F_SIZE;
    VSTORE(vals, 0, &slm[slm_wr_offset]);

    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- PHASE 2: Write to output (coalesced along z within subgroup) ----
    // Transpose: lid_x now serves as the z-local index (consecutive → coalesced output writes)
    //            lid_z now serves as the x-local index
    const uint write_z_local = lid_x;  // subgroup lane → z for coalesced writes
    const uint write_x_local = lid_z;

    const uint out_z = z_tile_id * Z_TILE + write_z_local;
    const uint out_x_base = x_tile_id * X_TILE + write_x_local;

    // Read from SLM (transposed access)
    // Need data for: z_local = write_z_local, x_local = write_x_local
    const uint slm_rd_offset = write_z_local * SLM_ROW_STRIDE + write_x_local * F_SIZE;
    INPUTVTYPE out_vals = VLOAD(0, &slm[slm_rd_offset]);

    // Output bfyx: offset = f_out * Y_OUT * X_OUT + y_out * X_OUT + x_out
    // Where: f_out = y, y_out = f + out_x_base * F_SIZE, x_out = out_z
    // Base output offset for f=0: y * OUTPUT_Y_SIZE * OUTPUT_X_SIZE + out_x_base * F_SIZE * OUTPUT_X_SIZE + out_z
    const uint out_base = y * (OUTPUT_Y_SIZE * OUTPUT_X_SIZE) + out_x_base * F_SIZE * OUTPUT_X_SIZE + out_z;

    // Write each feature value. Stride between f values in output = OUTPUT_X_SIZE (=Z_SIZE=48)
#if VEC_SIZE == 8
    output[out_base + 0 * OUTPUT_X_SIZE] = out_vals.s0;
    output[out_base + 1 * OUTPUT_X_SIZE] = out_vals.s1;
    output[out_base + 2 * OUTPUT_X_SIZE] = out_vals.s2;
    output[out_base + 3 * OUTPUT_X_SIZE] = out_vals.s3;
    output[out_base + 4 * OUTPUT_X_SIZE] = out_vals.s4;
    output[out_base + 5 * OUTPUT_X_SIZE] = out_vals.s5;
    output[out_base + 6 * OUTPUT_X_SIZE] = out_vals.s6;
    output[out_base + 7 * OUTPUT_X_SIZE] = out_vals.s7;
#elif VEC_SIZE == 4
    output[out_base + 0 * OUTPUT_X_SIZE] = out_vals.s0;
    output[out_base + 1 * OUTPUT_X_SIZE] = out_vals.s1;
    output[out_base + 2 * OUTPUT_X_SIZE] = out_vals.s2;
    output[out_base + 3 * OUTPUT_X_SIZE] = out_vals.s3;
#elif VEC_SIZE == 2
    output[out_base + 0 * OUTPUT_X_SIZE] = out_vals.s0;
    output[out_base + 1 * OUTPUT_X_SIZE] = out_vals.s1;
#elif VEC_SIZE == 16
    output[out_base + 0 * OUTPUT_X_SIZE] = out_vals.s0;
    output[out_base + 1 * OUTPUT_X_SIZE] = out_vals.s1;
    output[out_base + 2 * OUTPUT_X_SIZE] = out_vals.s2;
    output[out_base + 3 * OUTPUT_X_SIZE] = out_vals.s3;
    output[out_base + 4 * OUTPUT_X_SIZE] = out_vals.s4;
    output[out_base + 5 * OUTPUT_X_SIZE] = out_vals.s5;
    output[out_base + 6 * OUTPUT_X_SIZE] = out_vals.s6;
    output[out_base + 7 * OUTPUT_X_SIZE] = out_vals.s7;
    output[out_base + 8 * OUTPUT_X_SIZE] = out_vals.s8;
    output[out_base + 9 * OUTPUT_X_SIZE] = out_vals.s9;
    output[out_base + 10 * OUTPUT_X_SIZE] = out_vals.sa;
    output[out_base + 11 * OUTPUT_X_SIZE] = out_vals.sb;
    output[out_base + 12 * OUTPUT_X_SIZE] = out_vals.sc;
    output[out_base + 13 * OUTPUT_X_SIZE] = out_vals.sd;
    output[out_base + 14 * OUTPUT_X_SIZE] = out_vals.se;
    output[out_base + 15 * OUTPUT_X_SIZE] = out_vals.sf;
#endif
}
