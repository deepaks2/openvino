// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/group_correlation.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<group_correlation> : public typed_program_node_base<group_correlation> {
    using parent = typed_program_node_base<group_correlation>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using group_correlation_node = typed_program_node<group_correlation>;

template <>
class typed_primitive_inst<group_correlation> : public typed_primitive_inst_base<group_correlation> {
    using parent = typed_primitive_inst_base<group_correlation>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(group_correlation_node const& /*node*/,
                                                    const kernel_impl_params& impl_param);
    static layout calc_output_layout(const group_correlation_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const group_correlation_node& node);

    typed_primitive_inst(network& network, group_correlation_node const& node);
};

using group_correlation_inst = typed_primitive_inst<group_correlation>;

}  // namespace cldnn
