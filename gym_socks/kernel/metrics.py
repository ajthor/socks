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
        X: The observations oganized in ROWS.
        Y: The observations oganized in ROWS.
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
        X: The observations oganized in ROWS.
        Y: The observations oganized in ROWS.
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
        X: The observations oganized in ROWS.
        Y: The observations oganized in ROWS.
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


def delta_kernel(X: np.ndarray, Y: np.ndarray = None):
    """Delta (discrete) kernel function.

    The delta kernel is defined as :math:`k(x_{i}, x_{j}) = \delta_{ij}`, meaning the
    kernel returns a 1 if the vectors are the same, and a 0 otherwise. The vectors in
    `X` and `Y` should have discrete values, meaning each element in the vector should
    be a natural number or integer value.

    Args:
        X: A 2D ndarray with the observations oganized in ROWS.
        Y: A 2D ndarray with the observations oganized in ROWS.

    Returns:
        np.ndarray: Gram matrix of pairwise evaluations of the kernel function.

    """

    if X.ndim == 1:
        raise ValueError(
            "Expected 2D array for X, got 1D array instead. \n"
            "Reshape the data using array.reshape(-1, 1) "
            "if the data has only a single dimension "
            "or array.reshape(1, -1) if there is only a single sample."
        )

    if Y is None:
        Y = X

    else:

        if Y.ndim == 1:
            raise ValueError(
                "Expected 2D array for Y, got 1D array instead. \n"
                "Reshape the data using array.reshape(-1, 1) "
                "if the data has only a single dimension "
                "or array.reshape(1, -1) if there is only a single sample."
            )

    N, M = np.shape(X)
    T = len(Y)

    D = np.ones((N, T), dtype=int)

    for i in range(M):
        D &= np.tile(Y[:, i], (N, 1)) == np.tile(X[:, i], (T, 1)).T

    return D


def regularized_inverse(
    G: np.ndarray,
    regularization_param: float = None,
    kernel_fn=None,
) -> np.ndarray:
    r"""Regularized inverse.

    Computes the regularized matrix inverse.

    .. math::

        W = (G + \lambda M I)^{-1}, \quad
        G \in \mathbb{R}^{n \times n}, \quad
        G_{ij} = k(x_{i}, y_{j})

    Args:
        G: The Gram (kernel) matrix.
        regularization_param: Regularization parameter, which is a strictly positive
            real value.

    Returns:
        Regularized matrix inverse.

    """

    m, n = np.shape(G)
    assert m == n, "Gram matrix must be square."

    I = np.empty_like(G)
    np.fill_diagonal(I, regularization_param * m)

    return np.linalg.inv(G + I)
