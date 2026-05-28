// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_bfyx_to_bfsv16.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

ParamsKey PermuteKernel_bfyx_to_bfsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool PermuteKernel_bfyx_to_bfsv16::Validate(const Params& p) const {
    if (!PermuteKernelBase::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const permute_params& params = static_cast<const permute_params&>(p);

    // Only static shapes
    if (params.has_dynamic_tensors())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Must be 4D→4D
    if (params.inputs[0].GetDims().size() != 4 || params.outputs[0].GetDims().size() != 4)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Layout check
    if (params.inputs[0].GetLayout() != DataLayout::bfyx)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (params.outputs[0].GetLayout() != DataLayout::b_fs_yx_fsv16)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Permute order in cldnn format (converted from OV [0,3,1,2]):
    // OV [0,3,1,2] → cldnn [0,2,3,1]
    // This corresponds to output[b][x_in][f_in][y_in] = input[b][f_in][y_in][x_in]
    const std::vector<uint16_t> expected_order = {0, 2, 3, 1};
    if (params.order != expected_order)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const auto& in = params.inputs[0];

    // Output feature dimension (= input X) must be divisible by 16 for vectorized access
    if (in.X().v % 16 != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // No fused ops for safety
    if (!params.fused_ops.empty())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

CommonDispatchData PermuteKernel_bfyx_to_bfsv16::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in = params.inputs[0];

    // Dispatch: (INPUT_X / 16, INPUT_F, INPUT_Y * BATCH)
    // Each work-item copies 16 consecutive values using vector operations.
    dispatchData.gws = {in.X().v / 16, in.Feature().v, in.Y().v * in.Batch().v};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants PermuteKernel_bfyx_to_bfsv16::GetJitConstants(const permute_params& params,
                                                            const CommonDispatchData& /*dispatchData*/) const {
    auto jit = MakeBaseParamsJitConstants(params);

    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];

    // Input shape in bfyx
    jit.AddConstant(MakeJitConstant("INPUT_F", in.Feature().v));
    jit.AddConstant(MakeJitConstant("INPUT_Y", in.Y().v));
    jit.AddConstant(MakeJitConstant("INPUT_X", in.X().v));

    // Output pitches for b_fs_yx_fsv16
    // offset = (f/16) * fs_pitch + y * y_pitch + x * 16 + (f%16)
    size_t output_y_dim = out.Y().v;   // = input_f
    size_t output_x_dim = out.X().v;   // = input_y
    size_t fs_pitch = output_y_dim * output_x_dim * 16;
    size_t y_pitch = output_x_dim * 16;

    jit.AddConstant(MakeJitConstant("OUTPUT_FS_PITCH", fs_pitch));
    jit.AddConstant(MakeJitConstant("OUTPUT_Y_PITCH", y_pitch));

    return jit;
}

KernelsPriority PermuteKernel_bfyx_to_bfsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
