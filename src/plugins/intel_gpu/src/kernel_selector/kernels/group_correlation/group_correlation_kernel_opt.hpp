// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "group_correlation_kernel_base.hpp"

namespace kernel_selector {

/// @brief Optimized GroupCorrelation kernel with SLM tiling along X dimension.
/// Reduces global memory reads by sharing right-feature loads across work-items in a tile.
class GroupCorrelationKernelOpt : public GroupCorrelationKernelBase {
public:
    GroupCorrelationKernelOpt() : GroupCorrelationKernelBase("group_correlation_opt") {}

protected:
    ParamsKey GetSupportedKey() const override;
    CommonDispatchData CalcDispatch(const group_correlation_params& kernel_params) const override;
    JitConstants GetJitConstants(const group_correlation_params& kernel_params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override {
        return FORCE_PRIORITY_2;
    }
};

}  // namespace kernel_selector
