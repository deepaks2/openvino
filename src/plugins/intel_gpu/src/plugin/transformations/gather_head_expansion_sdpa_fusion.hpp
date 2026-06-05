// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// @brief Fuses Gather-based GQA head expansion into SDPA native broadcasting.
///
/// Matches pattern: K/V -> Gather(repeat_indices, axis=heads_axis) -> SDPA
/// where the Gather implements repeat_interleave for GQA head expansion
/// (e.g., [B,S,4,D] -> [B,S,8,D] with indices=[0,0,1,1,2,2,3,3]).
///
/// Removes the Gather and connects K/V directly to SDPA, which handles
/// GQA broadcasting natively via kv_group_size when Q.num_heads > K.num_heads.
class GatherHeadExpansionSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatherHeadExpansionSDPAFusion");
    GatherHeadExpansionSDPAFusion();
};

}   // namespace ov::intel_gpu
