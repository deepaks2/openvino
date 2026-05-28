// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/group_correlation.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov::intel_gpu::op {

GroupCorrelation::GroupCorrelation(const ov::Output<Node>& left_feat,
                                   const ov::Output<Node>& right_feat,
                                   int64_t num_groups,
                                   int64_t channels_per_group,
                                   int64_t max_disparity,
                                   const ov::element::Type output_type)
    : Op({left_feat, right_feat}),
      m_num_groups(num_groups),
      m_channels_per_group(channels_per_group),
      m_max_disparity(max_disparity),
      m_output_type(output_type) {
    validate_and_infer_types();
}

bool GroupCorrelation::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("num_groups", m_num_groups);
    visitor.on_attribute("channels_per_group", m_channels_per_group);
    visitor.on_attribute("max_disparity", m_max_disparity);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void GroupCorrelation::validate_and_infer_types() {
    const auto& left_shape = get_input_partial_shape(0);

    auto out_type = m_output_type;
    if (out_type == ov::element::dynamic) {
        out_type = get_input_element_type(0);
    }

    // Output shape: (N, num_groups, max_disparity, H, W)
    ov::PartialShape output_shape = ov::PartialShape::dynamic(5);
    if (left_shape.rank().is_static() && left_shape.rank().get_length() == 4) {
        output_shape = {left_shape[0],
                        m_num_groups,
                        m_max_disparity,
                        left_shape[2],
                        left_shape[3]};
    }

    set_output_type(0, out_type, output_shape);
}

std::shared_ptr<Node> GroupCorrelation::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GroupCorrelation>(new_args[0],
                                              new_args[1],
                                              m_num_groups,
                                              m_channels_per_group,
                                              m_max_disparity,
                                              m_output_type);
}

}  // namespace ov::intel_gpu::op
