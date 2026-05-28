// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_bzyxf_to_bfyx.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

static constexpr size_t X_TILE = 16;
static constexpr size_t Z_TILE = 16;
// SLM row padding (in half elements) to avoid bank conflicts.
// Stride per z-row = X_TILE*F + PAD. In DWORDs: (X_TILE*F + PAD)/2.
// For F=8: (128+2)/2 = 65 DW, gcd(65,32)=1 → conflict-free.
static constexpr size_t SLM_PAD = 2;

ParamsKey PermuteKernel_bzyxf_to_bfyx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bzyxf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool PermuteKernel_bzyxf_to_bfyx::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const permute_params& params = static_cast<const permute_params&>(p);

    // Only static shapes
    if (params.has_dynamic_tensors())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Must be 5D input → 4D output
    if (params.inputs[0].GetDims().size() != 5 || params.outputs[0].GetDims().size() != 4)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Layout check
    if (params.inputs[0].GetLayout() != DataLayout::bzyxf)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (params.outputs[0].GetLayout() != DataLayout::bfyx)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Validate permute order: [0, 3, 4, 1, 2] in kernel_selector 5D convention
    // This produces: output[b][y][f+x*F][z] = input[b][f][z][y][x]
    const std::vector<uint16_t> expected_order = {0, 3, 4, 1, 2};
    if (params.order != expected_order)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const auto& in = params.inputs[0];

    // Feature must be power of 2 and <= 16 for vectorization
    size_t f = in.Feature().v;
    if (f != 2 && f != 4 && f != 8 && f != 16)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // X and Z must be divisible by tile size for simplicity
    if (in.X().v % X_TILE != 0 || in.Z().v % Z_TILE != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Verify output shape matches: F_out=Y, Y_out=F*X, X_out=Z
    const auto& out = params.outputs[0];
    if (out.Feature().v != in.Y().v)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (out.Y().v != in.Feature().v * in.X().v)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (out.X().v != in.Z().v)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // No fused ops for safety
    if (!params.fused_ops.empty())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants PermuteKernel_bzyxf_to_bfyx::GetJitConstants(const permute_params& params,
                                                           const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    const auto& in = params.inputs[0];
    size_t f_size = in.Feature().v;
    size_t x_size = in.X().v;
    size_t y_size = in.Y().v;
    size_t z_size = in.Z().v;

    size_t slm_row_stride = X_TILE * f_size + SLM_PAD;

    jit.AddConstant(MakeJitConstant("X_TILE", X_TILE));
    jit.AddConstant(MakeJitConstant("Z_TILE", Z_TILE));
    jit.AddConstant(MakeJitConstant("F_SIZE", f_size));
    jit.AddConstant(MakeJitConstant("X_SIZE", x_size));
    jit.AddConstant(MakeJitConstant("Y_SIZE", y_size));
    jit.AddConstant(MakeJitConstant("Z_SIZE", z_size));
    jit.AddConstant(MakeJitConstant("SLM_ROW_STRIDE", slm_row_stride));
    jit.AddConstant(MakeJitConstant("SLM_TOTAL_SIZE", Z_TILE * slm_row_stride));
    jit.AddConstant(MakeJitConstant("OUTPUT_Y_SIZE", in.Feature().v * in.X().v));
    jit.AddConstant(MakeJitConstant("OUTPUT_X_SIZE", z_size));

    // Vector width for f-dimension loads
    jit.AddConstant(MakeJitConstant("VEC_SIZE", f_size));

    return jit;
}

CommonDispatchData PermuteKernel_bzyxf_to_bfyx::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in = params.inputs[0];

    // GWS: (X, Z, Y*B)
    // LWS: (X_TILE, Z_TILE, 1)
    // get_local_id(0) = lid_x (fast dim for coalesced input reads)
    // get_local_id(1) = lid_z
    // get_global_id(2) = y * batch_idx (batch=1 for our case)
    dispatchData.gws = {in.X().v, in.Z().v, in.Y().v * in.Batch().v};
    dispatchData.lws = {X_TILE, Z_TILE, 1};

    return dispatchData;
}

KernelsPriority PermuteKernel_bzyxf_to_bfyx::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
