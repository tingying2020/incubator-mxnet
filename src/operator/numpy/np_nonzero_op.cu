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
 * Copyright (c) 2018 by Contributors
 * \file boolean_mask.cu
*/

#include "np_nonzero_op-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

struct PrefixSumInit{
  template<typename DTypeIn, typename DTypeOut>
  MSHADOW_XINLINE static void Map(int i,
                  DTypeOut* out,
                  DTypeIn* in) {
    if(in[i]){
    out[i] = DTypeOut(1);
    }
  }
};

#define MAXDIM 10

inline void NonzeroForward(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NDArray &in = inputs[0];
  const NDArray &out = outputs[0];
  CHECK_LE(in.shape().ndim(), MAXDIM) << "ndim of input cannot larger than " << MAXDIM;
  // 0-dim
  if(0 == in.shape().ndim()){
    MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
      DType* in_dptr = in.data().dptr<DType>();
      if(*in_dptr) {
        mxnet::TShape s(1, 1);
        const_cast<NDArray &>(out).Init(s);
        *(out.data().dptr<int64_t>()) = 0;
      } else {
        mxnet::TShape s(1, 0);
        const_cast<NDArray &>(out).Init(s);
      }
    });
    return ;
  }
  size_t in_size = in.shape().Size();
  int32_t valid_num = 0;
  Stream<gpu>* stream = ctx.get_stream<gpu>();
  int32_t* prefix_sum = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // Calculate total temporary memory size
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                in_size,
                                Stream<gpu>::GetStream(stream));
  size_t buffer_size = in_size * sizeof(int32_t);
  temp_storage_bytes += buffer_size;
  // Allocate memory on GPU and allocate pointer
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), stream);
  prefix_sum = reinterpret_cast<int32_t*>(workspace.dptr_);
  d_temp_storage = workspace.dptr_ + buffer_size;
  MSHADOW_TYPE_SWITCH(in.dtype(), IType, {
    mxnet_op::Kernel<PrefixSumInit, gpu>::Launch(
      stream, in_size, prefix_sum, in.data().dptr<IType>());
  });
  // Calculate prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                in_size,
                                Stream<gpu>::GetStream(stream));
  CUDA_CALL(cudaMemcpy(&valid_num, &prefix_sum[in_size - 1], sizeof(int32_t),
                       cudaMemcpyDeviceToHost));
  // set the output shape forcefully
  mxnet::TShape s(2, in.shape().ndim());
  s[0] = valid_num;
  const_cast<NDArray &>(out).Init(s);
  // get the shape from the input
  NONZERO_NDIM_SWITCH(in.shape().ndim(), ndim, {
    mshadow::Shape<ndim> shape = in.shape().get<ndim>();
    mxnet_op::Kernel<NonzeroForwardKernel, gpu>::Launch(
      stream, in_size, out.data().dptr<int64_t>(), prefix_sum, shape);
  })
}

NNVM_REGISTER_OP(_npx_nonzero)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FComputeEx>("FComputeEx<gpu>", NonzeroForward);

} // namespace op
} // namespace mxnet