"""
Kernel functions and helper utilities for kernel-based calculations.

Most of the functions are implemented already in sklearn.metrics.pairwise. They are re-implemented here as a lightweight alternative. Most, if not all of the kernel functions defined in sklearn.metrics.pairwise should be compatible with the functions defined here.

The only kernel implemented here is the RBF (Gaussian) kernel. If other kernels or distance metrics are desired, then import the kernel from sklearn.

E.g.
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel

"""
from functools import partial, reduce

import numpy as np
import numpy.matlib
from numpy.linalg import inv


def euclidean_distance(X, Y=None, squared: bool = False) -> "D":
    """
    Euclidean distance function.

    The main difference between the way this function calculates euclidean
    distance over other implementations such as in sklearn.metrics.pairwise is
    that this implementation is largely agnostic to the dimensionality of the
    input data, and generally works well for high-dimensional vectors and dense
    data sets, such as state trajectories over a long time horizon.

    Parameters
    ----------

    X : ndarray
        The observations are oganized in ROWS.

    Y : ndarray
        The observations are oganized in ROWS.

    squared : bool
        Whether or not the result is the squared Euclidean distance.
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
    X,
    Y=None,
    sigma: "Bandwidth parameter." = None,
    distance: "Distance function." = None,
) -> "K":
    """
    RBF kernel function.

    Computes the pairwise evaluation of the RBF kernel on each vector in X and
    Y. For instance, if X has n vecotrs, and Y has m vectors, then the result is
    an n x m matrix K where K_ij = k(xi, yj).

    X = [[--- x1 ---],
         [--- x2 ---],
         ...
         [--- xn ---]]

    Y = [[--- y1 ---],
         [--- y2 ---],
         ...
         [--- ym ---]]

    The result is a matrix,

    K = [[k(x1,y1), ..., k(x1,ym)],
          ...
         [k(xn,y1), ..., k(xn,ym)]]

    The main difference between this implementation and the rbf_kernel in
    sklearn.metrics.pairwise is that this function optionally allows you to
    specify a different distance metric in the event the data is non-Euclidean.

    Parameters
    ----------

    X : ndarray
        The observations are oganized in ROWS.

    Y : ndarray
        The observations are oganized in ROWS.

    sigma : float
        Strictly positive real-valued kernel parameter.

    distance : function
        Distance function to use in the kernel evaluation.
    """

    if distance is None:
        distance = euclidean_distance

    K = distance(X, Y, squared=True)

    if sigma is None:
        sigma = np.median(K)
    else:
        assert sigma > 0, "sigma must be a strictly positive real value."

    K /= -2 * (sigma ** 2)
    np.exp(K, K)

    return K


def regularized_inverse(
    X,
    Y=None,
    U=None,
    V=None,
    l: "Regularization parameter." = None,
    kernel_fn: "Kernel function." = None,
) -> "W":
    """
    Regularized inverse.

    Computes the regularized matrix inverse (K + lambda * M * I)^-1.


    Parameters
    ----------

    X : ndarray
        The observations are oganized in ROWS.

    Y : ndarray
        The observations are oganized in ROWS.

    l : float
        Regularization parameter, which is a strictly positive real value.

    kernel_fn : function
        The kernel function is a function that returns an ndarray where each element is the pairwise evaluation of a kernel function. See sklearn.metrics.pairwise for more info. The default is the RBF kernel.
    """

    if Y is None:
        Y = X

    if l is None:
        l = 1 / (X.shape[0] ** 2)
    else:
        assert l > 0, "l must be a strictly positive real value."

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

    return inv(K + l * num_rows * I)
