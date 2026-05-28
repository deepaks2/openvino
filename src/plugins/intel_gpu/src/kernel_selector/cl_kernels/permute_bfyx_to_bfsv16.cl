// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Optimized permute kernel for bfyx → b_fs_yx_fsv16 with order [0,3,1,2].
// Pattern: output[b][x_in][f_in][y_in] = input[b][f_in][y_in][x_in]
// where output is in b_fs_yx_fsv16 format and x_in (output's feature) is divisible by 16.
//
// The key insight: 16 consecutive channel values in the output's fsv16 block correspond
// to 16 consecutive values in the input's X (innermost) dimension. This allows vectorized
// vload16/vstore16 for near-bandwidth-limited performance.
//
// Dispatch: gws = (X_IN / 16, F_IN, Y_IN) where X_IN = output features
//           lws = auto
// JIT: INPUT_F, INPUT_Y, INPUT_X, OUTPUT_FS_PITCH, OUTPUT_Y_PITCH

KERNEL(permute_bfyx_to_bfsv16)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
    const int f_block = (int)get_global_id(0);  // output feature block (0..X_IN/16-1)
    const int y_out = (int)get_global_id(1);    // output Y = input F
    const int x_out = (int)get_global_id(2);    // output X = input Y

    // Input: bfyx with shape [1, INPUT_F, INPUT_Y, INPUT_X]
    // input[0][y_out][x_out][f_block*16 .. f_block*16+15]
    // offset = y_out * (INPUT_Y * INPUT_X) + x_out * INPUT_X + f_block * 16
    const int in_offset = y_out * (INPUT_Y * INPUT_X) + x_out * INPUT_X + f_block * 16;

    // Output: b_fs_yx_fsv16 with shape [1, INPUT_X, INPUT_F, INPUT_Y]
    // output[0][f_block][y_out][x_out][0..15]
    // offset = f_block * OUTPUT_FS_PITCH + y_out * OUTPUT_Y_PITCH + x_out * 16
    const int out_offset = f_block * OUTPUT_FS_PITCH + y_out * OUTPUT_Y_PITCH + x_out * 16;

    // Read 16 consecutive elements (contiguous in input's X dimension)
    // and write them contiguously into the fsv16 block
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 8) val0 = vload8(0, input + in_offset);
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 8) val1 = vload8(1, input + in_offset);
    vstore8(val0, 0, output + out_offset);
    vstore8(val1, 1, output + out_offset);
}
