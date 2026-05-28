// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "group_correlation_kernel_base.hpp"

namespace kernel_selector {

/// @brief Reference kernel for GroupCorrelation.
class GroupCorrelationKernelRef : public GroupCorrelationKernelBase {
public:
    GroupCorrelationKernelRef() : GroupCorrelationKernelBase("group_correlation_ref") {}

protected:
    ParamsKey GetSupportedKey() const override;
    CommonDispatchData CalcDispatch(const group_correlation_params& kernel_params) const override;
};

}  // namespace kernel_selector
