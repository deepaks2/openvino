// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
// Optimized eltwise kernel for bfyx layout with arbitrary pitches.
// Vectorizes in X dimension (8 elements per work item) with scalar tail handling.
// Supports inputs with different pitches from output (e.g., crop/slice views).
class EltwiseKernel_bfyx_opt : public EltwiseKernelBase {
public:
    EltwiseKernel_bfyx_opt() : EltwiseKernelBase("eltwise_bfyx_opt") {}
    virtual ~EltwiseKernel_bfyx_opt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
    DispatchData SetDefault(const eltwise_params& params) const;
};
}  // namespace kernel_selector
