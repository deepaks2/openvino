// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "group_correlation_inst.hpp"
#include "group_correlation/group_correlation_kernel_ref.hpp"
#include "group_correlation/group_correlation_kernel_selector.hpp"

namespace cldnn {
namespace ocl {

struct group_correlation_impl : public typed_primitive_impl_ocl<group_correlation> {
    using parent = typed_primitive_impl_ocl<group_correlation>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::group_correlation_kernel_selector;
    using kernel_params_t = kernel_selector::group_correlation_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::group_correlation_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<group_correlation_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<group_correlation>();
        auto params = get_default_params<kernel_selector::group_correlation_params>(impl_param);

        // Second input: right feature map
        const auto right_layout = impl_param.get_input_layout(1);
        params.inputs.push_back(convert_data_tensor(right_layout));

        params.num_groups = primitive->num_groups;
        params.channels_per_group = primitive->channels_per_group;
        params.max_disparity = primitive->max_disparity;

        return params;
    }
};

namespace detail {

attach_group_correlation_impl::attach_group_correlation_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {format::bfzyx};  // output is bfzyx (N, G, D, H, W)

    implementation_map<group_correlation>::add(impl_types::ocl,
                                                typed_primitive_impl_ocl<group_correlation>::create<group_correlation_impl>,
                                                types,
                                                formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::group_correlation_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_correlation)
