from functools import partial

import numpy as np
import numpy.matlib
from numpy.linalg import inv

# from sklearn.metrics import pairwise_distances
# from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.metrics.pairwise import rbf_kernel

# default_kernel = partial(rbf_kernel, sigma=0.02)


def euclidean_distance(X, Y=None, squared: bool = False):
    """
    Euclidean distance function.

    Parameters
    ----------

    X : ndarray
        The observations are oganized in ROWS.

    Y : ndarray
        The observations are oganized in ROWS.
    """

    if Y is None:
        Y = X

    num_rows_X, num_cols_X = X.shape
    num_rows_Y, num_cols_Y = Y.shape

    distances = np.zeros(shape=(num_rows_X, num_rows_Y))

    for i in range(num_cols_X):

        distances += np.power(
            np.matlib.repmat(Y[:, i], num_rows_X, 1)
            - np.matlib.repmat(X[:, i], num_rows_Y, 1).T,
            2,
        )

    return distances if squared else np.sqrt(distances)


def rbf_kernel(X, Y=None, sigma=None):
    """
    RBF kernel function.

    Computes the pairwise evaluation of the RBF kernel on each vector in X and Y. For instance, if X has n vecotrs, and Y has m vectors, then the result is an n x m matrix. 

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

    Parameters
    ----------

    X : ndarray
        The observations are oganized in ROWS.

    Y : ndarray
        The observations are oganized in ROWS.
    """

    if sigma is None:
        sigma = 0.1 / X.shape[0]

    G = euclidean_distance(X, Y, squared=True)
    G /= -2 * (sigma ** 2)
    np.exp(G, G)
    return G


def regularized_inverse(
    X,
    Y=None,
    l: "Regularization parameter." = None,
    kernel: "Kernel function." = None,
) -> "W":
    """
    Regularized inverse.

    Computes the regularized matrix inverse (G + lambda * M * I)^-1.

    Parameters
    ----------

    kernel : function
        The kernel function is a function that returns an ndarray where each element is the pairwise evaluation of a kernel function. See sklearn.metrics.pairwise for more info. The default is the RBF kernel.
    """

    if Y is None:
        Y = X

    if l is None:
        l = 1 / (X.shape[0] ** 2)

    if kernel is None:
        kernel = partial(rbf_kernel, sigma=1)

    err_msg = "Parameters %r, %r must have the same shape." % (X, Y)
    assert X.shape == Y.shape, err_msg

    num_rows, num_cols = X.shape

    G = kernel(X, Y)

    I = np.identity(num_rows)

    return inv(G + l * num_rows * I)
