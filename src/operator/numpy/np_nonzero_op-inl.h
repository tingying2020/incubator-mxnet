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
 * \file np_nonzero_op-inl.h
*/

#ifndef MXNET_OPERATOR_NUMPY_NP_NONZERO_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_NONZERO_OP_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

#define NONZERO_NDIM_SWITCH(NDim, ndim, ...)       \
  if (NDim == 0) {                                 \
  } else if (NDim == 1) {                          \
    const int ndim = 1;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 2) {                          \
    const int ndim = 2;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 3) {                          \
    const int ndim = 3;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 4) {                          \
    const int ndim = 4;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 5) {                          \
    const int ndim = 5;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 6) {                          \
    const int ndim = 6;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 7) {                          \
    const int ndim = 7;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 8) {                          \
    const int ndim = 8;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 9) {                          \
    const int ndim = 9;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 10) {                         \
    const int ndim = 10;                           \
    {__VA_ARGS__}                                  \
  } else {                                         \
    LOG(FATAL) << "ndim=" << NDim << "too large "; \
  }

namespace mxnet {
namespace op {

struct PrefixSumInit{
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  int32_t* out,
                                  DType* in) {
    if(in[i]){
      out[i] = 1;
    } else {
      out[i] = 0;
    }
  }
};

struct NonzeroForwardKernel {
  template<int ndim>
  MSHADOW_XINLINE static void Map(int i,
                                  int64_t* out,
                                  const int32_t* idx,
                                  const mshadow::Shape<ndim>& shape) {
    int32_t prev = (i == 0) ? 0 : idx[i - 1];
    int32_t curr = idx[i];
    if (prev != curr) {
      mshadow::Shape<ndim> coord = mxnet_op::unravel<ndim>(i, shape);
      for(int j=0; j<ndim; j++){
        out[prev * ndim + j] = coord[j];
      }
    }
  }
};

} // namespace op
} // namespace mxnet

#endif