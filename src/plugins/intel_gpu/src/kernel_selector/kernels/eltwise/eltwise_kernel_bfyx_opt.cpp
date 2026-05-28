// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>

namespace kernel_selector {

static constexpr size_t VEC_SIZE = 8;

ParamsKey EltwiseKernel_bfyx_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool EltwiseKernel_bfyx_opt::Validate(const Params& params) const {
    if (!EltwiseKernelBase::Validate(params)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    // Only support bfyx layout for all inputs and output
    if (ewParams.outputs[0].GetLayout() != DataLayout::bfyx)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        if (ewParams.inputs[i].GetLayout() != DataLayout::bfyx)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Reject comparison/logic modes that produce boolean results (vector code issue)
    if (IsUnsupportedModeForVecCode(ewParams))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Require same logical dimensions for all inputs and output (no broadcast)
    const auto& output = ewParams.outputs[0];
    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        const auto& input = ewParams.inputs[i];
        if (input.Batch().v != output.Batch().v ||
            input.Feature().v != output.Feature().v ||
            input.Y().v != output.Y().v ||
            input.X().v != output.X().v)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Need at least VEC_SIZE elements in X for vectorization benefit
    if (output.X().v < VEC_SIZE)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // No fused ops support for simplicity (keeps kernel safe)
    if (!ewParams.fused_ops.empty())
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // No updateInput support
    if (!ewParams.updateInputIds.empty())
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // No output buffer reads
    for (size_t op = 0; op < ewParams.operations.size(); op++) {
        for (size_t input_idx = 0; input_idx < ewParams.operations[op].inputs.size(); input_idx++) {
            if (ewParams.operations[op].inputs[input_idx].mode == EltwiseInputMode::OUTPUT_BUFFER) {
                DO_NOT_USE_THIS_KERNEL(params.layerID);
            }
        }
    }

    // Only single operation supported
    if (ewParams.operations.size() != 1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    return true;
}

JitConstants EltwiseKernel_bfyx_opt::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    // Add INPUTS_DECLS for kernel signature
    jit.Merge(MakeInputDeclsJitConstants(params, false));

    const auto& output = params.outputs[0];
    const size_t x_size = output.X().v;
    const size_t x_blocks = x_size / VEC_SIZE;
    const size_t x_tail = x_size % VEC_SIZE;

    jit.AddConstant(MakeJitConstant("VEC_SIZE", VEC_SIZE));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", x_blocks));
    jit.AddConstant(MakeJitConstant("X_TAIL", x_tail));
    jit.AddConstant(MakeJitConstant("X_SIZE", x_size));
    jit.AddConstant(MakeJitConstant("Y_SIZE", output.Y().v));
    jit.AddConstant(MakeJitConstant("F_SIZE", output.Feature().v));
    jit.AddConstant(MakeJitConstant("B_SIZE", output.Batch().v));

    // Number of inputs
    jit.AddConstant(MakeJitConstant("INPUTS_COUNT", params.inputs.size()));

    // Output pitches
    jit.AddConstant(MakeJitConstant("OUT_Y_PITCH", output.Y().pitch));
    jit.AddConstant(MakeJitConstant("OUT_F_PITCH", output.Feature().pitch));
    jit.AddConstant(MakeJitConstant("OUT_B_PITCH", output.Batch().pitch));
    jit.AddConstant(MakeJitConstant("OUT_OFFSET", output.GetFirstElementOffset()));

    // Input pitches for each input
    for (size_t i = 0; i < params.inputs.size(); i++) {
        const auto& input = params.inputs[i];
        std::string idx = std::to_string(i);
        jit.AddConstant(MakeJitConstant("IN" + idx + "_Y_PITCH", input.Y().pitch));
        jit.AddConstant(MakeJitConstant("IN" + idx + "_F_PITCH", input.Feature().pitch));
        jit.AddConstant(MakeJitConstant("IN" + idx + "_B_PITCH", input.Batch().pitch));
        jit.AddConstant(MakeJitConstant("IN" + idx + "_OFFSET", input.GetFirstElementOffset()));
    }

    // Operation type
    const auto& op = params.operations[0];
    switch (op.mode) {
        case EltwiseMode::ADD:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "(a) + (b)")); break;
        case EltwiseMode::SUB:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "(a) - (b)")); break;
        case EltwiseMode::MUL:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "(a) * (b)")); break;
        case EltwiseMode::DIV:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "(a) / (b)")); break;
        case EltwiseMode::MAX:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "fmax((a), (b))")); break;
        case EltwiseMode::MIN:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "fmin((a), (b))")); break;
        case EltwiseMode::POW:  jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "pow((a), (b))")); break;
        case EltwiseMode::SQUARED_DIFF: jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "((a) - (b)) * ((a) - (b))")); break;
        default:
            // Unsupported mode - should not reach here due to Validate
            jit.AddConstant(MakeJitConstant("ELTWISE_OP(a, b)", "(a) * (b)"));
            break;
    }

    // Activation support
    jit.Merge(MakeActivationJitConstants(params.activations, params.outputs[0].GetDType(), "_TYPED"));

    return jit;
}

EltwiseKernelBase::DispatchData EltwiseKernel_bfyx_opt::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.outputs[0];
    const size_t x_blocks = output.X().v / VEC_SIZE;
    // If there's a tail, we need one extra work item in X
    const size_t x_work_items = x_blocks + (output.X().v % VEC_SIZE > 0 ? 1 : 0);

    dispatchData.gws = {x_work_items, output.Y().v, output.Feature().v * output.Batch().v};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData EltwiseKernel_bfyx_opt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    return {kd};
}

KernelsPriority EltwiseKernel_bfyx_opt::GetKernelsPriority(const Params& /*params*/) const {
    // Higher priority than ref (DONT_USE_IF_HAVE_SOMETHING_ELSE) but lower than vload8 (FORCE_PRIORITY_8)
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
