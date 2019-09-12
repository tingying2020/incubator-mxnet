/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation of basic functions for elementwise numpy binary broadcast operator.
 */

#if MXNET_USE_TVM_OP
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

bool NumpyBinaryScalarType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return in_attrs->at(0) != -1;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(name)              \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr_parser([](NodeAttrs* attrs) {                           \
      attrs->parsed = std::stod(attrs->dict["scalar"]);             \
    })                                                              \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>) \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)  \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "source input")        \
  .add_argument("scalar", "float", "scalar input")


MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_add)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_add"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_subtract)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_sub"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_multiply)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mul"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_mod)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_power)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_power"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_add_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rminus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mul_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_mod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rmod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rmod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rmod_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_power_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rpower_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rpower>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_rpower_scalar"});

#if MXNET_USE_TVM_OP
static constexpr char func_floor_divide_cpu[] = "floor_divide";
static constexpr char func_floor_divide_gpu[] = "cuda_floor_divide";
static constexpr char func_floor_divide_scalar_cpu[] = "floor_divide_scalar";
static constexpr char func_floor_divide_scalar_gpu[] = "cuda_floor_divide_scalar";
static constexpr char func_rfloor_divide_scalar_cpu[] = "rfloor_divide_scalar";
static constexpr char func_rfloor_divide_scalar_gpu[] = "cuda_rfloor_divide_scalar";

TBlob PrependAxes(const TBlob& src, const int dst_ndim) {
  CHECK_LE(src.shape_.ndim(), dst_ndim);
  const int src_ndim = src.shape_.ndim();
  if (src_ndim == dst_ndim) return src;
  mxnet::TShape dst_shape(dst_ndim, 1);
  for (int i = dst_ndim - src_ndim; i < dst_ndim; ++i) {
    dst_shape[i] = src.shape_[i - dst_ndim + src_ndim];
  }
  return src.reshape(dst_shape);
}

template<const char* func>
void TVMBinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                               const mxnet::OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

  // prepare tblobs and TVMArgs
  std::vector<TBlob> tblobs = {inputs[0], inputs[1], outputs[0]};
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  const int ondim = outputs[0].shape_.ndim();
  const size_t num_args = inputs.size() + outputs.size();
  type_codes.resize(num_args);
  values.resize(num_args);
  for (size_t i = 0; i < num_args; ++i) {
    tblobs[i] = PrependAxes(tblobs[i], ondim);
    type_codes[i] = kArrayHandle;
    values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
  }
  tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], tblobs.size());
  tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
}

template<const char* func>
void TVMBinaryBroadcastScalarCompute(const nnvm::NodeAttrs& attrs,
                                     const mxnet::OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

  // prepare tblobs and TVMArgs
  std::vector<TBlob> tblobs = {inputs[0], outputs[0]};
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  const size_t num_args = 3;  // one input tensor, one scalar param, and one output
  type_codes.resize(num_args);
  values.resize(num_args);

  // input tensor setup
  type_codes[0] = kArrayHandle;
  values[0].v_handle = const_cast<DLTensor*>(&(tblobs[0].dltensor()));

  // scalar param
  type_codes[1] = kDLFloat;
  values[1].v_float64 = nnvm::get<double>(attrs.parsed);

  // output tensor
  type_codes[2] = kArrayHandle;
  values[2].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

  tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], 3);
  tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
}

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_floor_divide)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>", mxnet::op::TVMBinaryBroadcastCompute<func_floor_divide_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>", mxnet::op::TVMBinaryBroadcastCompute<func_floor_divide_cpu>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_npi_floor_divide_scalar)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>", mxnet::op::TVMBinaryBroadcastScalarCompute<func_floor_divide_scalar_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>", mxnet::op::TVMBinaryBroadcastScalarCompute<func_floor_divide_scalar_cpu>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_npi_rfloor_divide_scalar)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>", mxnet::op::TVMBinaryBroadcastScalarCompute<func_rfloor_divide_scalar_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>", mxnet::op::TVMBinaryBroadcastScalarCompute<func_rfloor_divide_scalar_cpu>);

#endif  // MXNET_USE_TVM_OP
}  // namespace op
}  // namespace mxnet
