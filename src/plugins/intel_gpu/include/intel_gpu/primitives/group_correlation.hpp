// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "primitive.hpp"

namespace cldnn {

/// @brief GroupCorrelation primitive – computes grouped cross-correlation cost
///        volume for stereo matching networks.
///
/// Given left and right feature maps (N, C, H, W), produces a 5D cost volume
/// (N, num_groups, max_disparity, H, W) where for each disparity d:
///   out[0, g, d, h, w] = mean_k( left[0, g*CPG+k, h, w] *
///                                  right[0, g*CPG+k, h, w+d] )
/// and pixels with w+d >= W are zero-padded.
struct group_correlation : primitive_base<group_correlation> {
    CLDNN_DECLARE_PRIMITIVE(group_correlation)

    group_correlation() : primitive_base("", {}) {}

    /// @brief Constructs group_correlation primitive.
    /// @param id              Primitive id.
    /// @param inputs          {left_feat, right_feat}.
    /// @param num_groups      Number of correlation groups (G).
    /// @param channels_per_group  Channels per group (K); total C = G*K.
    /// @param max_disparity   Maximum disparity (D).
    group_correlation(const primitive_id& id,
                      const std::vector<input_info>& inputs,
                      int64_t num_groups,
                      int64_t channels_per_group,
                      int64_t max_disparity)
        : primitive_base(id, inputs),
          num_groups(num_groups),
          channels_per_group(channels_per_group),
          max_disparity(max_disparity) {}

    int64_t num_groups = 8;
    int64_t channels_per_group = 12;
    int64_t max_disparity = 48;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_groups);
        seed = hash_combine(seed, channels_per_group);
        seed = hash_combine(seed, max_disparity);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const group_correlation>(rhs);
        return num_groups == rhs_casted.num_groups &&
               channels_per_group == rhs_casted.channels_per_group &&
               max_disparity == rhs_casted.max_disparity;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<group_correlation>::save(ob);
        ob << num_groups;
        ob << channels_per_group;
        ob << max_disparity;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<group_correlation>::load(ib);
        ib >> num_groups;
        ib >> channels_per_group;
        ib >> max_disparity;
    }
};

}  // namespace cldnn
