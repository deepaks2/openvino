// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_correlation_kernel_opt.hpp"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr size_t TILE_W = 16;  // subgroup width / tile width

CommonDispatchData GroupCorrelationKernelOpt::CalcDispatch(const group_correlation_params& kernel_params) const {
    CommonDispatchData dispatch_data;
    const auto& output = kernel_params.outputs.front();

    const size_t width = output.X().v;
    const size_t height = output.Y().v;
    const size_t num_groups = output.Feature().v;

    // Round up width to multiple of TILE_W
    const size_t width_aligned = ((width + TILE_W - 1) / TILE_W) * TILE_W;

    dispatch_data.gws = {
        num_groups,       // groups
        height,           // H
        width_aligned     // W (rounded up)
    };
    dispatch_data.lws = {1, 1, TILE_W};

    return dispatch_data;
}

ParamsKey GroupCorrelationKernelOpt::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableDifferentTypes();
    key.EnableInputLayout(DataLayout::bfzyx);
    key.EnableInputLayout(DataLayout::bzyxf);
    key.EnableOutputLayout(DataLayout::bfzyx);
    key.EnableOutputLayout(DataLayout::bzyxf);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

JitConstants GroupCorrelationKernelOpt::GetJitConstants(const group_correlation_params& kernel_params) const {
    auto jit = GroupCorrelationKernelBase::GetJitConstants(kernel_params);

    const auto& output = kernel_params.outputs.front();
    jit.AddConstants({
        MakeJitConstant("TILE_W", TILE_W),
        MakeJitConstant("OUTPUT_WIDTH", output.X().v),
    });

    return jit;
}

}  // namespace kernel_selector
