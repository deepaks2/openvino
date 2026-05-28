// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_correlation_kernel_selector.hpp"
#include "group_correlation_kernel_ref.hpp"
#include "group_correlation_kernel_opt.hpp"

namespace kernel_selector {

group_correlation_kernel_selector::group_correlation_kernel_selector() {
    Attach<GroupCorrelationKernelOpt>();
    Attach<GroupCorrelationKernelRef>();
}

KernelsData group_correlation_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GROUP_CORRELATION);
}

group_correlation_kernel_selector& group_correlation_kernel_selector::Instance() {
    static group_correlation_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
