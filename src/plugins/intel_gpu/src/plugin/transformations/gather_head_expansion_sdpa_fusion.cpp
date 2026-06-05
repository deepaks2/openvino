// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_head_expansion_sdpa_fusion.hpp"

#include "intel_gpu/op/sdpa.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

#include <algorithm>
#include <numeric>

namespace ov::intel_gpu {

// Check if indices represent a simple head repeat pattern like [0,0,1,1,2,2,3,3]
// for kv_group_size=2, or [0,0,0,1,1,1,2,2,2,3,3,3] for kv_group_size=3, etc.
static bool is_repeat_interleave_indices(const std::shared_ptr<ov::op::v0::Constant>& indices_const,
                                         int64_t input_heads,
                                         int64_t output_heads,
                                         int64_t& kv_group_size) {
    if (output_heads % input_heads != 0)
        return false;

    kv_group_size = output_heads / input_heads;
    if (kv_group_size < 2)
        return false;

    auto indices = indices_const->cast_vector<int64_t>();
    if (static_cast<int64_t>(indices.size()) != output_heads)
        return false;

    // Verify pattern: [0,0,...,0, 1,1,...,1, ..., N-1,N-1,...,N-1]
    // Each input head index should repeat kv_group_size times
    for (int64_t i = 0; i < input_heads; i++) {
        for (int64_t j = 0; j < kv_group_size; j++) {
            if (indices[i * kv_group_size + j] != i)
                return false;
        }
    }
    return true;
}

GatherHeadExpansionSDPAFusion::GatherHeadExpansionSDPAFusion() {
    using namespace ov::pass::pattern;

    auto sdpa_m = wrap_type<ov::intel_gpu::op::SDPA>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa_node = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(
            pattern_map.at(sdpa_m).get_node_shared_ptr());

        if (!sdpa_node || transformation_callback(sdpa_node))
            return false;

        // Check inputs 1 (K) and 2 (V) for Gather pattern
        bool changed = false;

        for (size_t input_idx : {1, 2}) {
            if (input_idx >= sdpa_node->get_input_size())
                continue;

            auto gather_node = ov::as_type_ptr<ov::op::v8::Gather>(
                sdpa_node->get_input_node_shared_ptr(input_idx));
            if (!gather_node)
                continue;

            // Gather must have exactly one consumer (this SDPA)
            if (gather_node->get_output_target_inputs(0).size() != 1)
                continue;

            // Get the indices (must be constant)
            auto indices_const = ov::as_type_ptr<ov::op::v0::Constant>(
                gather_node->get_input_node_shared_ptr(1));
            if (!indices_const)
                continue;

            // Get the axis (must be constant)
            auto axis_const = ov::as_type_ptr<ov::op::v0::Constant>(
                gather_node->get_input_node_shared_ptr(2));
            if (!axis_const)
                continue;

            int64_t axis = axis_const->cast_vector<int64_t>()[0];

            // Get shapes
            auto input_shape = gather_node->get_input_partial_shape(0);
            auto output_shape = gather_node->get_output_partial_shape(0);

            if (input_shape.rank().is_dynamic() || output_shape.rank().is_dynamic())
                continue;

            auto rank = input_shape.rank().get_length();
            if (axis < 0)
                axis += rank;

            // The axis should be the heads dimension.
            // For typical shapes [B,S,H,D], heads axis is 2.
            // Verify axis dimension is static in both input and output
            if (input_shape[axis].is_dynamic() || output_shape[axis].is_dynamic())
                continue;

            int64_t input_heads = input_shape[axis].get_length();
            int64_t output_heads = output_shape[axis].get_length();

            // Must be an expansion (output > input)
            if (output_heads <= input_heads)
                continue;

            // Verify Q has the matching number of heads (after transpose order)
            auto q_shape = sdpa_node->get_input_partial_shape(0);
            auto order_q = sdpa_node->get_input0_transpose_order();
            auto order_k = sdpa_node->get_input1_transpose_order();

            // Determine which dimension in the original K layout corresponds to num_heads
            // In the transposed layout, num_heads is at position 1.
            // So in the original layout, num_heads is at order_k[1] (or axis if no transpose)
            size_t k_heads_orig_dim = order_k.empty() ? 1 : static_cast<size_t>(order_k[1]);

            // The gather axis should match the heads dimension
            if (static_cast<size_t>(axis) != k_heads_orig_dim && input_idx == 1)
                continue;
            if (input_idx == 2) {
                auto order_v = sdpa_node->get_input2_transpose_order();
                size_t v_heads_orig_dim = order_v.empty() ? 1 : static_cast<size_t>(order_v[1]);
                if (static_cast<size_t>(axis) != v_heads_orig_dim)
                    continue;
            }

            // Check if indices form a valid repeat_interleave pattern
            int64_t kv_group_size = 0;
            if (!is_repeat_interleave_indices(indices_const, input_heads, output_heads, kv_group_size))
                continue;

            // Verify Q num_heads matches output_heads (expanded)
            size_t q_heads_orig_dim = order_q.empty() ? 1 : static_cast<size_t>(order_q[1]);
            if (q_shape.rank().is_dynamic() || q_shape[q_heads_orig_dim].is_dynamic())
                continue;
            if (q_shape[q_heads_orig_dim].get_length() != output_heads)
                continue;

            // All checks passed — bypass the Gather
            sdpa_node->input(input_idx).replace_source_output(
                gather_node->input_value(0));  // Connect directly to Gather's data input
            changed = true;
        }

        if (changed) {
            sdpa_node->validate_and_infer_types();
        }

        return changed;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "GatherHeadExpansionSDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
