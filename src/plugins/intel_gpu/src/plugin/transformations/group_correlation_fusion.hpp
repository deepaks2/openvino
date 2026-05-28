// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/// @brief Fuses the stereo cost-volume construction subgraph into a single
///        GroupCorrelation op.
///
/// Matches the following repeated pattern for each disparity d=0..D-1:
///   left_feat --(StridedSlice_L[d])--> Multiply[d] --> Reshape[d] --> ReduceMean[d]
///   right_feat-(StridedSlice_R[d])-/                                         |
///                                                                  ScatterNDUpdate[d]
///
/// The D ScatterNDUpdate ops form a serial chain: each one reads from the
/// previous one's output and appends a new depth-slice.
///
/// All of them are replaced by:
///   GroupCorrelation(left_feat, right_feat, num_groups, channels_per_group, D)
class GroupCorrelationFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GroupCorrelationFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
