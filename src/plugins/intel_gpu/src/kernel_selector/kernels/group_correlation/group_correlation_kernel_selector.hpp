// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

/// @brief GroupCorrelation kernel selector.
class group_correlation_kernel_selector : public kernel_selector_base {
public:
    group_correlation_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static group_correlation_kernel_selector& Instance();
};

}  // namespace kernel_selector
