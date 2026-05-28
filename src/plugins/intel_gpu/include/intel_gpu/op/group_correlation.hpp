// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief GPU-internal op that computes a grouped cross-correlation cost volume
///        used in stereo matching networks (e.g., IGEV-Stereo).
///
/// Inputs:
///   0: left_feat  (N, C, H, W)  – left feature map
///   1: right_feat (N, C, H, W)  – right feature map
///
/// Output:
///   (N, num_groups, max_disparity, H, W) cost volume
///
/// For each disparity d, group g, and pixel (h, w):
///   if (w + d) < W:
///     out[0, g, d, h, w] = mean over k in [0, channels_per_group) of
///                            left[0, g*CPG+k, h, w] * right[0, g*CPG+k, h, w+d]
///   else:
///     out[0, g, d, h, w] = 0
///
/// C = num_groups * channels_per_group
class GroupCorrelation : public ov::op::Op {
public:
    OPENVINO_OP("GroupCorrelation", "gpu_opset");

    GroupCorrelation() = default;

    GroupCorrelation(const ov::Output<Node>& left_feat,
                     const ov::Output<Node>& right_feat,
                     int64_t num_groups,
                     int64_t channels_per_group,
                     int64_t max_disparity,
                     const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_num_groups() const { return m_num_groups; }
    int64_t get_channels_per_group() const { return m_channels_per_group; }
    int64_t get_max_disparity() const { return m_max_disparity; }
    ov::element::Type get_output_type() const { return m_output_type; }

private:
    int64_t m_num_groups = 8;
    int64_t m_channels_per_group = 12;
    int64_t m_max_disparity = 48;
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
