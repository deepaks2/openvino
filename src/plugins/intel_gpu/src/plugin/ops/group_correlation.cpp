// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/group_correlation.hpp"
#include "intel_gpu/primitives/group_correlation.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

namespace ov {
namespace op {
namespace internal {
using GroupCorrelation = ov::intel_gpu::op::GroupCorrelation;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

namespace {

void CreateGroupCorrelationOp(ProgramBuilder& p,
                               const std::shared_ptr<ov::op::internal::GroupCorrelation>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);

    const cldnn::group_correlation prim(layer_type_name_ID(op),
                                        inputs,
                                        op->get_num_groups(),
                                        op->get_channels_per_group(),
                                        op->get_max_disparity());
    p.add_primitive(*op, prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(internal, GroupCorrelation);

}  // namespace ov::intel_gpu
