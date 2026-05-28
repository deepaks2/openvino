// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <sstream>

#include "group_correlation_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(group_correlation)

layout group_correlation_inst::calc_output_layout(const group_correlation_node& node,
                                                   const kernel_impl_params& impl_param) {
    const auto prim = impl_param.typed_desc<group_correlation>();
    const auto left_layout = impl_param.get_input_layout(0);
    const auto& dims = left_layout.get_dims();  // N, C, [Z,] H, W
    const auto rank = dims.size();

    const int64_t N = dims[0];
    const int64_t H = dims[rank - 2];  // Y/H — second-to-last, works for 4D and 5D
    const int64_t W = dims[rank - 1];  // X/W — last, works for 4D and 5D

    // Output: (N, num_groups, max_disparity, H, W) in bfzyx
    // tensor(b, f, x, y, z) => batch=N, feature=G, x=W, y=H, z=D
    const tensor out_tensor(static_cast<int>(N),
                            static_cast<int>(prim->num_groups),
                            static_cast<int>(W),
                            static_cast<int>(H),
                            static_cast<int>(prim->max_disparity));

    return {left_layout.data_type, format::bzyxf, out_tensor};
}

template <typename ShapeType>
std::vector<layout> group_correlation_inst::calc_output_layouts(group_correlation_node const& /*node*/,
                                                                  const kernel_impl_params& impl_param) {
    const auto prim = impl_param.typed_desc<group_correlation>();
    const auto left_layout = impl_param.get_input_layout(0);

    // Compute (N, num_groups, max_disparity, H, W) as a PartialShape
    auto in_shape = left_layout.get<ov::PartialShape>();  // (N, C, H, W) or (N, C, Z, H, W) if bfzyx
    const auto in_rank = in_shape.size();
    ov::PartialShape out_shape = {in_shape[0],
                                  prim->num_groups,
                                  prim->max_disparity,
                                  in_shape[in_rank - 2],  // H — second-to-last
                                  in_shape[in_rank - 1]}; // W — last

    return {layout{out_shape, left_layout.data_type, format::bzyxf}};
}

template std::vector<layout> group_correlation_inst::calc_output_layouts<ov::PartialShape>(
    group_correlation_node const& node,
    const kernel_impl_params& impl_param);

std::string group_correlation_inst::to_string(const group_correlation_node& node) {
    auto primitive = node.get_primitive();
    json_composite gc_info;
    gc_info.add("left_feat id", node.input(0).id());
    gc_info.add("right_feat id", node.input(1).id());
    gc_info.add("num_groups", primitive->num_groups);
    gc_info.add("channels_per_group", primitive->channels_per_group);
    gc_info.add("max_disparity", primitive->max_disparity);

    auto node_info = node.desc_to_json();
    node_info->add("group_correlation info", gc_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

group_correlation_inst::typed_primitive_inst(network& network, group_correlation_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
