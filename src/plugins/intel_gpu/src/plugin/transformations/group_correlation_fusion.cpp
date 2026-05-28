// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_correlation_fusion.hpp"
#include "intel_gpu/op/group_correlation.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov::intel_gpu {

namespace {

/// Returns the StridedSlice begin value for the last (x/width) dimension, or -1 on failure.
int64_t get_slice_x_begin(const std::shared_ptr<ov::op::v1::StridedSlice>& slice) {
    auto begin_const = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_input_node_shared_ptr(1));
    if (!begin_const)
        return -1;

    const auto begin_vals = begin_const->cast_vector<int64_t>();
    if (begin_vals.empty())
        return -1;

    return begin_vals.back();  // last dimension = width (x) for NCHW input
}

}  // namespace

bool GroupCorrelationFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Collect all ScatterNDUpdate v3 nodes (the version used by IGEV-Stereo model).
    // We find the TERMINAL scatter in a chain: its output is not consumed by another
    // ScatterNDUpdate.
    for (const auto& op : model->get_ordered_ops()) {
        auto terminal_scatter = ov::as_type_ptr<ov::op::v3::ScatterNDUpdate>(op);
        if (!terminal_scatter)
            continue;

        // Check if this is a terminal (no ScatterNDUpdate consumer on output 0).
        bool is_terminal = true;
        for (const auto& in : terminal_scatter->output(0).get_target_inputs()) {
            if (ov::is_type<ov::op::v3::ScatterNDUpdate>(in.get_node())) {
                is_terminal = false;
                break;
            }
        }
        if (!is_terminal)
            continue;

        // Walk backward through the chain: chain[0] is the first scatter,
        // chain.back() == terminal_scatter.
        std::vector<std::shared_ptr<ov::op::v3::ScatterNDUpdate>> chain;
        {
            auto cur = terminal_scatter;
            while (cur) {
                chain.push_back(cur);
                // Port 0 is "data"; if it's another ScatterND, keep walking.
                auto data_input = cur->input_value(0).get_node_shared_ptr();
                auto prev = ov::as_type_ptr<ov::op::v3::ScatterNDUpdate>(data_input);
                cur = prev;
            }
        }
        std::reverse(chain.begin(), chain.end());  // chain[0] first, chain[D-1] terminal

        if (chain.size() < 2)
            continue;

        const int64_t max_disparity = static_cast<int64_t>(chain.size());

        // Validate the pattern for every scatter in the chain:
        //   ScatterND.input(2) [updates] = ReduceMean <- Reshape <- Multiply
        //                                <- (Slice_L, Slice_R) [absent at di==0]
        //                                <- (left_feat, right_feat)
        ov::Output<ov::Node> left_feat_output, right_feat_output;
        int64_t num_groups = -1;
        int64_t channels_per_group = -1;
        bool pattern_ok = true;

        for (size_t di = 0; di < chain.size(); ++di) {
            const auto& scatter = chain[di];

            // Port 2 is "updates" for ScatterNDUpdate
            auto updates_node = scatter->input_value(2).get_node_shared_ptr();

            // updates <- ReduceMean
            auto reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(updates_node);
            if (!reduce) {
                pattern_ok = false;
                break;
            }

            // ReduceMean input 0 <- Reshape
            auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(
                reduce->input_value(0).get_node_shared_ptr());
            if (!reshape) {
                pattern_ok = false;
                break;
            }

            // Reshape input 0 <- Multiply
            auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(
                reshape->input_value(0).get_node_shared_ptr());
            if (!mul) {
                pattern_ok = false;
                break;
            }

            // Multiply inputs are (Slice, Slice) for di>0, or direct features for di==0.
            // The IGEV model omits Slice for d=0 (identity slices are folded away).
            auto slice_0 = ov::as_type_ptr<ov::op::v1::StridedSlice>(
                mul->input_value(0).get_node_shared_ptr());
            auto slice_1 = ov::as_type_ptr<ov::op::v1::StridedSlice>(
                mul->input_value(1).get_node_shared_ptr());

            if (di == 0) {
                // For d=0 the full-width feature maps feed Multiply directly
                // (identity Slices are removed by constant folding).
                if (slice_0 && slice_1) {
                    // Trivial slices survived; use their sources.
                    left_feat_output  = slice_0->input_value(0);  // tentative
                    right_feat_output = slice_1->input_value(0);
                } else if (!slice_0 && !slice_1) {
                    // Direct feature inputs (most common after optimization).
                    left_feat_output  = mul->input_value(0);  // tentative
                    right_feat_output = mul->input_value(1);
                } else {
                    pattern_ok = false;
                    break;
                }

                // Extract group dimensions from Reshape output shape:
                // Reshape: (N, C, H, W) -> (N, G, K, H, W)
                const auto& reshape_out = reshape->get_output_partial_shape(0);
                if (reshape_out.rank().is_static() &&
                    reshape_out.rank().get_length() == 5 &&
                    reshape_out[1].is_static() && reshape_out[2].is_static()) {
                    num_groups         = reshape_out[1].get_length();
                    channels_per_group = reshape_out[2].get_length();
                } else {
                    pattern_ok = false;
                    break;
                }
            } else {
                // For di > 0, Slice nodes are required.
                if (!slice_0 || !slice_1) {
                    pattern_ok = false;
                    break;
                }

                const auto src0 = slice_0->input_value(0);
                const auto src1 = slice_1->input_value(0);

                // Verify slices draw from the same two feature tensors recorded at di==0.
                bool match_direct  = (src0 == left_feat_output  && src1 == right_feat_output);
                bool match_swapped = (src0 == right_feat_output && src1 == left_feat_output);
                if (!match_direct && !match_swapped) {
                    pattern_ok = false;
                    break;
                }

                // At di==1 determine the shift convention:
                //   IGEV convention: left feature is sliced [d:W]  (start=d),
                //                   right feature is sliced [0:W-d] (start=0).
                //   Kernel convention: output[g,d,h,w] = mean_k( left[w] * right[w-d] )
                if (di == 1) {
                    const int64_t x_begin_0 = get_slice_x_begin(slice_0);
                    const int64_t x_begin_1 = get_slice_x_begin(slice_1);
                    if (x_begin_0 < 0 || x_begin_1 < 0) {
                        pattern_ok = false;
                        break;
                    }
                    if (x_begin_0 == 1 && x_begin_1 == 0) {
                        // src0 is LEFT (begin=1 at d=1), src1 is RIGHT (begin=0).
                        left_feat_output  = src0;
                        right_feat_output = src1;
                    } else if (x_begin_0 == 0 && x_begin_1 == 1) {
                        // src1 is LEFT (begin=1 at d=1), src0 is RIGHT (begin=0).
                        left_feat_output  = src1;
                        right_feat_output = src0;
                    } else {
                        pattern_ok = false;
                        break;
                    }
                }
            }
        }

        if (!pattern_ok || num_groups < 0 || channels_per_group < 0) {
            continue;
        }

        // All checks passed – replace with GroupCorrelation.
        auto gc_node = std::make_shared<op::GroupCorrelation>(left_feat_output,
                                                               right_feat_output,
                                                               num_groups,
                                                               channels_per_group,
                                                               max_disparity);

        gc_node->set_friendly_name(terminal_scatter->get_friendly_name() +
                                   "/group_correlation");

        // Collect original layer names for rt_info
        ov::NodeVector orig_nodes(chain.begin(), chain.end());
        ov::copy_runtime_info(orig_nodes, gc_node);

        // Replace the terminal scatter's consumers with GroupCorrelation.
        terminal_scatter->output(0).replace(gc_node->output(0));

        changed = true;
        // There is typically only one such subgraph; break after first replacement
        // so that model->get_ordered_ops() is not invalidated during iteration.
        break;
    }

    return changed;
}

}  // namespace ov::intel_gpu
