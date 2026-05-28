// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/// @brief Kernel parameters for the GroupCorrelation kernel.
struct group_correlation_params : public base_params {
    group_correlation_params() : base_params(KernelType::GROUP_CORRELATION) {}
    int64_t num_groups = 8;
    int64_t channels_per_group = 12;
    int64_t max_disparity = 48;
};

/// @brief Base class for GroupCorrelation kernels.
class GroupCorrelationKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    KernelsData GetKernelsData(const Params& params) const override;

protected:
    virtual CommonDispatchData CalcDispatch(const group_correlation_params& kernel_params) const = 0;
    virtual JitConstants GetJitConstants(const group_correlation_params& kernel_params) const;
    bool Validate(const Params& params) const override;
};

}  // namespace kernel_selector
