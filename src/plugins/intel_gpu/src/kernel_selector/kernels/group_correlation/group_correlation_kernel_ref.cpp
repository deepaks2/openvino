// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_correlation_kernel_ref.hpp"
#include "kernel_selector_utils.h"

namespace kernel_selector {

CommonDispatchData GroupCorrelationKernelRef::CalcDispatch(const group_correlation_params& kernel_params) const {
    CommonDispatchData dispatch_data;
    const auto& output = kernel_params.outputs.front();

    // Dispatch (NUM_GROUPS, H, W) — each work item computes all MAX_DISPARITY
    // outputs for one (g, y, x) position, with left features cached in registers.
    dispatch_data.gws = {
        output.Feature().v,  // NUM_GROUPS (G)
        output.Y().v,         // H
        output.X().v          // W
    };
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo);

    return dispatch_data;
}

ParamsKey GroupCorrelationKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableDifferentTypes();
    key.EnableInputLayout(DataLayout::bfzyx);  // inputs are promoted to bfzyx (z=1) by graph optimizer
    key.EnableOutputLayout(DataLayout::bfzyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

}  // namespace kernel_selector
