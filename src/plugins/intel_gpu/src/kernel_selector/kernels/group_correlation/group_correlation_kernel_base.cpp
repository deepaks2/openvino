// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_correlation_kernel_base.hpp"
#include "kernel_selector_utils.h"

namespace kernel_selector {

KernelsData GroupCorrelationKernelBase::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<group_correlation_params>(params);
    const auto& kernel_params = dynamic_cast<const group_correlation_params&>(*kernel_data.params);
    const auto dispatch_data = CalcDispatch(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    // 2 inputs: left_feat and right_feat
    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point, {}, false, false, 2);

    return {kernel_data};
}

bool GroupCorrelationKernelBase::Validate(const Params& params) const {
    if (params.GetType() != KernelType::GROUP_CORRELATION) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& kernel_params = dynamic_cast<const group_correlation_params&>(params);
    if (kernel_params.inputs.size() != 2) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

JitConstants GroupCorrelationKernelBase::GetJitConstants(const group_correlation_params& kernel_params) const {
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("NUM_GROUPS", kernel_params.num_groups),
        MakeJitConstant("CHANNELS_PER_GROUP", kernel_params.channels_per_group),
        MakeJitConstant("MAX_DISPARITY", kernel_params.max_disparity),
    });

    return jit_constants;
}

}  // namespace kernel_selector
