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

"""Namespace for operators used in Gluon dispatched by F=ndarray."""
from __future__ import absolute_import
import numpy as np
from ...base import numeric_types
from ...context import current_context
from ..ndarray import NDArray
from . import _internal as _npi

__all__ = ['uniform', 'normal', 'multinomial']


def _random_helper(random, sampler, params, shape, dtype, ctx, out, kwargs):
    """Helper function for random generators."""
    from ...numpy import ndarray as np_ndarray
    if isinstance(params[0], np_ndarray):
        for i in params[1:]:
            assert isinstance(i, np_ndarray), \
                "Distribution parameters must all have the same type, but got " \
                "both %s and %s." % (type(params[0]), type(i))
        return sampler(*params, shape=shape, dtype=dtype, out=out, **kwargs)
    elif isinstance(params[0], numeric_types):
        if ctx is None:
            ctx = current_context()
        if shape is None and out is None:
            shape = ()
        for i in params[1:]:
            assert isinstance(i, numeric_types), \
                "Distribution parameters must all have the same type, but got " \
                "both %s and %s."%(type(params[0]), type(i))
        return random(*params, shape=shape, dtype=dtype, ctx=ctx, out=out, **kwargs)

    raise ValueError("Distribution parameters must be either mxnet.numpy.ndarray or numbers, "
                     "but got %s." % type(params[0]))


def uniform(low=0.0, high=1.0, size=None, **kwargs):
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.
    out : ndarray, optional
        Store output to an existing ndarray.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized uniform distribution.


    Notes
    -----
    This function currently does not support ``low`` and ``high`` as ndarrays.
    """
    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = 'float32'
    ctx = kwargs.pop('ctx', None)
    out = kwargs.pop('out', None)
    return _random_helper(_npi.random_uniform, None,
                          [low, high], size, dtype, ctx, out, kwargs)


def normal(loc=0.0, scale=1.0, size=None, **kwargs):
    r"""
    normal(loc=0.0, scale=1.0, size=None, dtype='float32', ctx=None, out=None)

    Draw random samples from a normal (Gaussian) distribution.

    The probability density function of the normal distribution, first
    derived by De Moivre and 200 years later by both Gauss and Laplace
    independently [1]_, is often called the bell curve because of
    its characteristic shape (see the example below).

    The normal distributions occurs often in nature.  For example, it
    describes the commonly occurring distribution of samples influenced
    by a large number of tiny, random disturbances, each with its own
    unique distribution [2]_.

    Parameters
    ----------
    loc : float, optional
        Mean (centre) of the distribution.
    scale : float, optional
        Standard deviation (spread or "width") of the distribution.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., `(m, n, k)`, then `m * n * k`
        samples are drawn. If size is `None` (default), a scalar tensor containing
        a single value is returned if loc and scale are both scalars.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : None or mxnet.cpu() or mxnet.gpu(gpuid), optional
        Device context to put the created array in.
    out : ndarray, optional
        Store output to an existing ndarray.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    This function currently does not support ``loc`` and ``scale`` as ndarrays.
    The probability density for the Gaussian distribution is

    .. math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                     e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },
    
    where :math:`\mu` is the mean and :math:`\sigma` the standard
    deviation. The square of the standard deviation, :math:`\sigma^2`,
    is called the variance.
    
    The function has its peak at the mean, and its "spread" increases with
    the standard deviation (the function reaches 0.607 times its maximum at
    :math:`x + \sigma` and :math:`x - \sigma` [2]_).  This implies that
    `numpy.random.normal` is more likely to return samples lying close to
    the mean, rather than those far away.
    
    This function differs to the original numpy.random.normal in the following aspects:
    
        - Argument must be named from the fourth.

    References
    ----------
    .. [1] Wikipedia, "Normal distribution",
           https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001,
           pp. 51, 51, 125.
 
    Examples
    --------
    >>> mu, sigma = 0, 0.1 # mean and standard deviation
    >>> s = np.random.normal(mu, sigma, 1000)
    
    Verify the mean:
    
    >>> abs(mu - (float)(np.mean(s))) < 0.01
    True

    >>> np.random.normal(1,1,dtype='float16',size=(2,3))
    array([[ 0.905 ,  1.433 , -0.3403],
           [ 1.994 ,  1.205 ,  2.512 ]], dtype=float16)
    """
    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = 'float32'
    ctx = kwargs.pop('ctx', None)
    out = kwargs.pop('out', None)
    return _random_helper(_npi.random_normal, None,
                          [loc, scale], size, dtype, ctx, out, kwargs)


def multinomial(n, pvals, size=None):
    """Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalisation of the binomial distribution.
    Take an experiment with one of ``p`` possible outcomes. An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6. Each sample drawn from the distribution represents n such experiments.
    Its values, ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome was ``i``.


    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the p different outcomes. These should sum to 1
        (however, the last element is always assumed to account for the remaining
        probability, as long as ``sum(pvals[:-1]) <= 1)``.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k`` sam-
        ples are drawn. Default is None, in which case a single value is returned.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape size, if that was provided. If not, the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional value drawn from the distribution.
    """
    if isinstance(pvals, NDArray):
        return _npi.multinomial(pvals, pvals=None, n=n, size=size)
    else:
        if isinstance(pvals, np.ndarray):
            pvals = pvals.tolist()
        if any(isinstance(i, list) for i in pvals):
            raise ValueError('object too deep for desired array')
        return _npi.multinomial(n=n, pvals=pvals, size=size)
