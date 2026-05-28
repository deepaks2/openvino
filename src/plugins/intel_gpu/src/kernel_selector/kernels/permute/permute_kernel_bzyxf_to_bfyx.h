// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {

// Optimized permute kernel for 5D bzyxf -> 4D bfyx transpose+reshape patterns.
// Uses SLM-based tiling for coalesced reads (along x) and coalesced writes (along z).
// Targets the specific pattern: output[b][y][f+x*F][z] = input[b][f][z][y][x]
class PermuteKernel_bzyxf_to_bfyx : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernel_bzyxf_to_bfyx() : PermuteKernelBase("permute_bzyxf_to_bfyx") {}
    ~PermuteKernel_bzyxf_to_bfyx() override = default;

    bool Validate(const Params& p) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }
};
}  // namespace kernel_selector
