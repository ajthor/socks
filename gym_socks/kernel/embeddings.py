from functools import partial

import numpy as np

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse


def marginal_kernel_embedding(X, kernel_fn: "Kernel function." = None) -> "<f, m(x)>":
    """
    Compute the marginal kernel distribution embedding.

    Returns
    -------

    function
        Returns a function of one variable, which can be used to approximate the expectation with respect to a marginal distribution.

        Example:
        m = marginal_kernel_embedding(X)
        m(1)
    """

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    def marginal_embedding(eval_points):
        return np.sum(kernel_fn(X, Y=eval_points), axis=1)

    return marginal_embedding


def conditional_kernel_embedding(
    X,
    Y=None,
    l: "Regularization parameter." = None,
    kernel_fn: "Kernel function." = None,
) -> "<f, m(.|x)>":
    """
    Compute the conditional kernel distribution embedding.

    Returns
    -------

    function
        Returns a function of one two variables, which can be used to approximate the expectation with respect to a conditional distribution.

        Example:
        def fn(x):
            return x ** 2
        m = conditional_kernel_embedding(X)
        m(fn, 1)
    """

    if l is None:
        l = 1 / (X.shape[0] ** 2)
    else:
        assert l > 0, "l must be a strictly positive real value."

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    W = regularized_inverse(X, Y, kernel_fn=kernel_fn)

    def conditional_embedding(fn, eval_points):
        CXT = kernel_fn(X, eval_points)
        return np.matmul(fn(Y), np.matmul(W, CXT))

    return conditional_embedding
