from functools import partial

import numpy as np
import scipy

from scipy.linalg import cholesky
from scipy.linalg import cho_solve
from scipy.linalg import solve_triangular

from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.utils.validation import check_array
from gym_socks.utils.validation import check_matrix


def remove_regularization(A, A_inv, regularization_param):
    r"""

    .. math::

        W = (G + \lambda M I)^{-1}
        W^{*} = (G + \lambda M I - \lambda M I)^{-1}
        (A + B)^{-1} = A^{-1} - (A + A B^{-1} A)^{-1}

    """

    A = check_matrix(A, ensure_square=True)
    A_inv = check_matrix(A_inv, ensure_square=True)

    M = len(A_inv)

    C = np.zeros_like(A)
    C[np.diag_indices_from(C)] += -M * regularization_param
    D = np.zeros_like(C)
    D[np.diag_indices_from(D)] += -1 / (M * regularization_param)

    return D - np.linalg.inv(C + C @ A_inv @ C)


def _partitioned_inverse(A_inv, x, y, a):
    A_inv = check_matrix(A_inv, ensure_square=True)

    A11_inv = A_inv
    A22 = a

    A12 = x
    A21 = y.T

    F22 = A22 - A21 @ A11_inv @ A12
    F22_inv = np.linalg.inv(F22)

    F11_inv = A11_inv + A11_inv @ A12 @ F22_inv @ A21 @ A11_inv

    return np.block(
        [
            [F11_inv, -A11_inv @ A12 @ F22_inv],
            [-F22_inv @ A21 @ A11_inv, F22_inv],
        ]
    )


def add_sample(
    W: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn=None,
    regularization_param: float = None,
):
    W = check_matrix(W, ensure_square=True)

    X = check_array(X)
    y = np.atleast_2d(y)

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    if regularization_param is None:
        regularization_param = 1 / (len(X) ** 2)
    else:
        assert (
            regularization_param > 0
        ), "regularization_param must be a strictly positive real value."

    k = kernel_fn(X, y)
    a = kernel_fn(y) + regularization_param

    return _partitioned_inverse(W, k, k, a)


def remove_first_sample(W: np.ndarray):
    W = check_matrix(W, ensure_square=True)

    a = W[1:, 1:]
    b = W[0, 1:].reshape(-1, 1)
    d = W[0, 0]

    return a - (1 / d) * b @ b.T


def remove_last_sample(W: np.ndarray):
    W = check_matrix(W, ensure_square=True)

    a = W[:-1, :-1]
    b = W[:-1, -1].reshape(-1, 1)
    d = W[-1, -1]

    return a - (1 / d) * b @ b.T


def cho_update(L, x):

    L = check_matrix(L, ensure_square=True)

    x = np.squeeze(x)
    m = len(x)

    for j in range(m):
        Ljj = L[j, j]
        r = np.sqrt(Ljj ** 2 + x[j] ** 2)
        c = r / Ljj
        s = x[j] / Ljj

        L[j, j] = r

        k = j + 1
        L[j, k:] = (L[j, k:] + s * x[k:]) / c
        x[k:] = c * x[k:] - s * L[j, k:]

    return L


def cho_add_rc(
    c_and_lower: tuple,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn=None,
    regularization_param: float = None,
):

    L, lower = c_and_lower

    L = check_matrix(L, ensure_square=True)

    X = check_array(X)
    y = np.atleast_2d(y)

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    if regularization_param is None:
        regularization_param = 1 / (len(X) ** 2)
    else:
        assert (
            regularization_param > 0
        ), "regularization_param must be a strictly positive real value."

    k = kernel_fn(X, y)
    a = kernel_fn(y) + regularization_param

    S12 = solve_triangular(L, k, trans="T", lower=lower)

    return np.block(
        [
            [L, S12],
            [np.zeros_like(S12).T, np.sqrt(a - S12.T @ S12)],
        ]
    )


def cho_del_rc(
    c_and_lower: tuple,
    last: bool = True,
):

    L, lower = c_and_lower

    L = check_matrix(L, ensure_square=True)

    # If we are removing the last sample, then we simply extract the upper matrix.
    if last is True:
        return L[:-1, :-1]

    # Otherwise, we extract the lower matrix and perform a rank 1 update.
    else:
        S23 = L[0, 1:].reshape(1, -1)
        S33 = L[1:, 1:]
        return cho_update(S33, S23)
