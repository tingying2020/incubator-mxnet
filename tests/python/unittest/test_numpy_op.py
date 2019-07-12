# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from __future__ import absolute_import
import numpy as _np
import mxnet as mx
from mxnet import np, npx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.base import MXNetError
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient, use_np
from common import assertRaises, with_seed
import random
import collections


@with_seed()
@use_np
def test_np_sum():
    class TestSum(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestSum, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.sum(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' in dtype

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'int8': 'int32', 'int32': 'int64', 'int64': 'int64'}
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                    for dtype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                        if is_int(dtype) and not is_int(itype):
                            continue
                        # test gluon
                        test_sum = TestSum(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_sum.hybridize()
                        if is_int(itype):
                            x = _np.random.randint(-128, 128, shape, dtype=itype)
                            x = mx.nd.array(x)
                        else:
                            x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
                        x = x.as_np_ndarray()
                        x.attach_grad()
                        expected_ret = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_sum(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                            atol=1e-5 if dtype == 'float16' else 1e-5)

                        y.backward()
                        assert same(x.grad.asnumpy(), _np.ones(shape=x.shape, dtype=x.dtype))

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_arctan2():
    class TestArctan2(HybridBlock):
        def __init__(self):
            super(TestArctan2, self).__init__()

        def hybrid_forward(self, F, x1, x2):
            return F.np.arctan2(x1, x2)

    def dimReduce(src, des):
        srcShape = src.shape
        desShape = des.shape
        if len(desShape) == 0:
            return src.sum()
        redu = []
        for i, j in zip(range(len(srcShape)-1, -1, -1), range(len(desShape)-1, -1, -1)):
            if srcShape[i] != desShape[j] and desShape[j] == 1:
                redu.append(i)
            if j == 0:
                for k in range(0, i):
                    redu.append(k)
                break
        if len(redu) > 0:
            src = _np.reshape(src.sum(axis=tuple(redu)), desShape)
        return src

    types = ['float64', 'float32', 'float16']
    for hybridize in [True, False]:
        for shape1, shape2 in [[(1,), (1,)],  # single elements
                               [(4, 5), (4, 5)],  # normal case
                               [(3, 2), (3, 2)],  # tall matrices
                               [(), ()],  # scalar only
                               [(3, 0, 2), (3, 0, 2)],  # zero-dim
                               [(3, 4, 5), (4, 1)],  # trailing dim broadcasting
                               [(3, 4, 5), ()],  # scalar broadcasting
                               [(), (1, 2, 3)],  # scalar broadcasting
                               [(4, 3), (4, 1)],  # single broadcasting
                               [(3, 4, 5), (3, 1, 5)]  # single broadcasting in the middle
                               ]:
            for oneType in types:
                if oneType == 'float16':
                    rtol=1e-2
                    atol = 1e-2
                else:
                    rtol=1e-3
                    atol = 1e-5
                test_arctan2 = TestArctan2()
                if hybridize:
                    test_arctan2.hybridize()
                x1 = rand_ndarray(shape1, dtype=oneType).as_np_ndarray()
                x2 = rand_ndarray(shape2, dtype=oneType).as_np_ndarray()
                x11 = x1.asnumpy()
                x21 = x2.asnumpy()
                x1.attach_grad()
                x2.attach_grad()
                np_out = _np.arctan2(x1.asnumpy(), x2.asnumpy())
                with mx.autograd.record():
                    mx_out = test_arctan2(x1, x2)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                mx_out.backward()
                np_backward_1 = x21 / (x11 * x11 + x21 * x21)
                np_backward_2 = -1 * x11 / (x11 * x11 + x21 * x21)
                np_backward_1 = dimReduce(np_backward_1, x11)
                np_backward_2 = dimReduce(np_backward_2, x21)
                assert_almost_equal(x1.grad.asnumpy(), np_backward_1, rtol=rtol, atol=atol)
                assert_almost_equal(x2.grad.asnumpy(), np_backward_2, rtol=rtol, atol=atol)

                mx_out = np.arctan2(x1, x2)
                np_out = _np.arctan2(x1.asnumpy(), x2.asnumpy())
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


if __name__ == '__main__':
    import nose
    nose.runmodule()
