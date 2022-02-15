"""
Kernel functions and helper utilities for kernel-based calculations.

Most of the commonly-used kernel functions are already implemented in
sklearn.metrics.pairwise. The RBF kernel and pairwise Euclidean distance function is
re-implemented here as an alternative, in case sklearn is unavailable. Most, if not all
of the kernel functions defined in sklearn.metrics.pairwise should be compatible with
the functions defined here.

To use the sklearn functions, use them like so:

    >>> from sklearn.metrics.pairwise import rbf_kernel
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = np.arange(4).reshape((2, 2))
    >>> Y = np.arange(6).reshape((3, 2))
    >>> K = gym_socks.kernel.metrics.rbf_kernel(X, Y, distance_fn=euclidean_distances)

"""

from functools import partial, reduce

import numpy as np
from numpy.linalg import inv


def euclidean_distance(
    X: np.ndarray,
    Y: np.ndarray = None,
    squared: bool = False,
) -> np.ndarray:
    """Euclidean distance function.

    The main difference between the way this function calculates euclidean
    distance over other implementations such as in sklearn.metrics.pairwise is
    that this implementation is largely agnostic to the dimensionality of the
    input data, and generally works well for high-dimensional vectors and dense
    data sets, such as state trajectories over a long time horizon.

    Args:
        X: The observations are oganized in ROWS.
        Y: The observations are oganized in ROWS.
        squared: Whether or not the result is the squared Euclidean distance.

    Returns:
        np.ndarray: Matrix of pairwise distances between points.

    """

    if Y is None:
        Y = X

    num_rows_X, num_cols_X = X.shape
    num_rows_Y, num_cols_Y = Y.shape

    def calc_dim(prev, next):
        x, y = next
        return prev + np.power(
            np.tile(y, (num_rows_X, 1)) - np.tile(x, (num_rows_Y, 1)).T,
            2,
        )

    distances = reduce(
        calc_dim, zip(X.T, Y.T), np.zeros(shape=(num_rows_X, num_rows_Y))
    )

    return distances if squared else np.sqrt(distances)


def rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray = None,
    sigma: float = None,
    distance_fn=None,
) -> np.ndarray:
    """RBF kernel function.

    Computes the pairwise evaluation of the RBF kernel on each vector in X and
    Y. For instance, if X has n vecotrs, and Y has m vectors, then the result is
    an n x m matrix K where K_ij = k(xi, yj).
    ::

        X = [[--- x1 ---],
             [--- x2 ---],
             ...
             [--- xn ---]]

        Y = [[--- y1 ---],
             [--- y2 ---],
             ...
             [--- ym ---]]

    The result is a matrix,
    ::

        K = [[k(x1,y1), ..., k(x1,ym)],
             ...
             [k(xn,y1), ..., k(xn,ym)]]

    The main difference between this implementation and the rbf_kernel in
    sklearn.metrics.pairwise is that this function optionally allows you to
    specify a different distance metric in the event the data is non-Euclidean.

    Args:
        X: The observations are oganized in ROWS.
        Y: The observations are oganized in ROWS.
        sigma: Strictly positive real-valued kernel parameter.
        distance_fn: Distance function to use in the kernel evaluation.

    Returns:
        np.ndarray: Gram matrix of pairwise evaluations of the kernel function.

    """

    # if distance_fn is None:
    #     distance_fn = euclidean_distance

    # K = distance_fn(X, Y, squared=True)

    if Y is None:
        Y = X

    N, M = np.shape(X)
    T = len(Y)

    K = np.zeros((N, T))

    for i in range(M):
        K += np.power(np.tile(Y[:, i], (N, 1)) - np.tile(X[:, i], (T, 1)).T, 2)

    if sigma is None:
        sigma = np.median(K)
    else:
        assert sigma > 0, "sigma must be a strictly positive real value."

    K /= -2 * (sigma ** 2)
    np.exp(K, K)

    return K


def rbf_kernel_derivative(
    X: np.ndarray,
    Y: np.ndarray = None,
    sigma: float = None,
    distance_fn=None,
) -> np.ndarray:

    # if distance_fn is None:
    #     distance_fn = euclidean_distance

    # D = distance_fn(X, Y, squared=False)

    if Y is None:
        Y = X

    N, M = np.shape(X)
    T = len(Y)

    D = np.zeros((N, T))

    for i in range(M):
        D += np.tile(Y[:, i], (N, 1)) - np.tile(X[:, i], (T, 1)).T

    if sigma is None:
        sigma = np.median(D)
    else:
        assert sigma > 0, "sigma must be a strictly positive real valued."

    D *= 2 / (2 * sigma ** 2)

    return D


def abel_kernel(
    X: np.ndarray,
    Y: np.ndarray = None,
    sigma: float = None,
    distance_fn=None,
) -> np.ndarray:
    """Abel kernel function.

    Args:
        X: The observations are oganized in ROWS.
        Y: The observations are oganized in ROWS.
        sigma: Strictly positive real-valued kernel parameter.
        distance_fn: Distance function to use in the kernel evaluation.

    Returns:
        np.ndarray: Gram matrix of pairwise evaluations of the kernel function.

    """

    if distance_fn is None:
        distance_fn = euclidean_distance

    K = distance_fn(X, Y, squared=False)

    if sigma is None:
        sigma = np.median(K)
    else:
        assert sigma > 0, "sigma must be a strictly positive real value."

    K /= -sigma
    np.exp(K, K)

    return K


def regularized_inverse(
    X: np.ndarray,
    Y: np.ndarray = None,
    U: np.ndarray = None,
    V: np.ndarray = None,
    regularization_param: float = None,
    kernel_fn=None,
) -> np.ndarray:
    r"""Regularized inverse.

    Computes the regularized matrix inverse.

    .. math::

        W = (K + \lambda M I)^{-1}, \quad
        K \in \mathbb{R}^{n \times n}, \quad
        K_{ij} = k(x_{i}, y_{j})

    Example:

        >>> from gym_socks.kernel.metrics import regularized_inverse
        >>> X = np.arange(4).reshape((2, 2))
        >>> Y = np.arange(6).reshape((3, 2))
        >>> K = regularized_inverse(X, Y)

    Args:
        X: The observations are oganized in ROWS.
        Y: The observations are oganized in ROWS.
        regularization_param: Regularization parameter, which is a strictly positive
            real value.
        kernel_fn: The kernel function is a function that returns an ndarray where
            each element is the pairwise evaluation of a kernel function. See sklearn.
            metrics.pairwise for more info. The default is the RBF kernel.

    Returns:
        Regularized matrix inverse.

    """

    if Y is None:
        Y = X

    if regularization_param is None:
        regularization_param = 1 / (X.shape[0] ** 2)
    else:
        assert (
            regularization_param > 0
        ), "regularization_param must be a strictly positive real value."

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    err_msg = "Parameters %r, %r must have the same shape." % (X, Y)
    assert X.shape == Y.shape, err_msg

    num_rows, num_cols = X.shape

    K = kernel_fn(X, Y)

    if U is not None:
        err_msg = "Parameters %r, %r must have the same sample size." % (X, U)
        assert X.shape[0] == U.shape[0], err_msg

        if V is not None:
            err_msg = "Parameters %r, %r must have the same shape." % (U, V)
            assert U.shape == V.shape, err_msg

        K = np.multiply(kernel_fn(U, V), K)

    I = np.identity(num_rows)

    return inv(K + regularization_param * num_rows * I)
