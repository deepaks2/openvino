// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {

/// @brief Optimized permute for bfyx → b_fs_yx_fsv16 with order [0,3,1,2].
/// Uses vectorized vload16/vstore16 when output features are aligned to 16.
class PermuteKernel_bfyx_to_bfsv16 : public PermuteKernelBase {
public:
    PermuteKernel_bfyx_to_bfsv16() : PermuteKernelBase("permute_bfyx_to_bfsv16") {}
    ~PermuteKernel_bfyx_to_bfsv16() override = default;

    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const override;
};

}  // namespace kernel_selector
