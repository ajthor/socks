r"""
Matrix update functions for modifying the kernel-based approximation.

Given a finite sample :math:`X`, we generally compute the Gram (kernel) matrix :math:`G`
and the regularized inverse :math:`W` in kernel-based algorithms. Occasionally, we want
to modify the sample, either by adding, removing, or changing a sample point. However,
we don't want to recompute the matrix inverse :math:`W` every time the sample :math:`X`
is changed.

.. math::

    W = (G + \lambda M I)^{-1}

Instead, we can use rank-1 updates and linear algebra identities to modify the matrix
inverse using the pre-computed matrix :math:`W`.

.. attention::

    It is generally computationally cheaper to use the Cholesky factorization :math:`A =
    LL^{\top}` to solve a system of linear equations of the form :math:`A x = b` rather
    than explicitly computing :math:`A^{-1}`. In addition, it is easier to add, remove,
    and change the Cholesky factorization of the Gram matrix when the underlying sample
    is changed.

"""

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
    """Compute the partitioned matrix inverse."""

    A11_inv = check_matrix(A_inv, ensure_square=True)

    F22 = a - y.T @ A11_inv @ x
    F22_inv = np.linalg.inv(F22)

    F11_inv = A11_inv + A11_inv @ x @ F22_inv @ y.T @ A11_inv

    return np.block(
        [
            [F11_inv, -A11_inv @ x @ F22_inv],
            [-F22_inv @ y.T @ A11_inv, F22_inv],
        ]
    )


def rinv_add_rc(
    W: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn=None,
    regularization_param: float = None,
    precomputed: bool = False,
):
    """Add a sample to the regularized matrix inverse.

    When a new sample point :math:`y` is added to the sample :math:`X`, we generally do
    not want to explicitly recompute the matrix inverse :math:`W`. Instead, we can
    modify the inverse directly by adding a new row and column to :math:`W`.

    .. tip::

        The Cholesky factorization can be computationally cheaper to update and store
        than the explicit inverse of a positive semidefinite matrix. It is recommended
        to use the Cholesky factor update rather than explicitly modifying the
        regularized inverse matrix directly.

    Args:
        W: The regularized inverse matrix.
        X: The sample used to compute the inverse. Needed to compute the vector
            ``kernel_fn(X, y)``.
        y: The new sample to be added to the inverse. Needed to compute the vector
            ``kernel_fn(y) + regularization_param``.
        kernel_fn: The kernel function used to compute the inverse.
        regularization_param: The regularization parameter :math:`\lambda > 0`.
        precomputed: Whether the kernel over the new sample is precomputed. If this is
            ``True``, then :py:obj:`X` and :py:obj:`y` are assumed to be a vector of the
            same shape as ``kernel_fn(X, y)`` and scalar ``kernel_fn(y)``, respectively.

    Returns:
        The modified regularized inverse matrix.

    """

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

    if precomputed is False:
        k = kernel_fn(X, y)
        a = kernel_fn(y) + regularization_param
    else:
        k = X
        a = y

    return _partitioned_inverse(W, k, k, a)


def rinv_del_rc(W: np.ndarray, last: bool = True):
    """Remove a sample from the regularized matrix inverse.

    Given a sample :math:`X`, we generally compute the Gram (kernel) matrix :math:`G`
    and the regularized inverse :math:`W`. When a sample point is removed from
    :math:`X`, we can modify the inverse directly by removing a row and column from
    :math:`W`.

    .. tip::

        The Cholesky factorization can be computationally cheaper to update and store
        than the explicit inverse of a positive semidefinite matrix. It is recommended
        to use the Cholesky factor update rather than explicitly modifying the
        regularized inverse matrix directly.

    Currently, only the first and last samples can be removed from the inverse.
    Specifying :py:obj:`last=False` in the arguments removes the first sample. In order
    to remove samples from the middle of the sample set, first permute the matrix.

    Args:
        W: The regularized inverse matrix.
        last: Whether to remove the last sample or the first. Default is ``True``.

    Returns:
        The modified regularized inverse matrix.

    """

    W = check_matrix(W, ensure_square=True)

    if last is True:
        a = W[:-1, :-1]
        b = W[:-1, -1].reshape(-1, 1)
        d = W[-1, -1]

    else:
        a = W[1:, 1:]
        b = W[0, 1:].reshape(-1, 1)
        d = W[0, 0]

    return a - (1 / d) * b @ b.T


def cho_update(c_and_lower: tuple, x: np.ndarray):
    """Perform a rank-1 update of the Cholesky factorization.

    Args:
        c_and_lower: A tuple containing the Cholesky factor and a boolean variable
            indicating whether it is the upper or lower factor.
        x: The rank-1 update vector.

    Returns:
        The updated Cholesky factor.

    """

    L, lower = c_and_lower
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

        if lower is True:
            L[k:, j] = (L[k:, j] + s * x[k:]) / c
            x[k:] = c * x[k:] - s * L[k:, j]

        else:
            L[j, k:] = (L[j, k:] + s * x[k:]) / c
            x[k:] = c * x[k:] - s * L[j, k:]

    return L


def cho_add_rc(
    c_and_lower: tuple,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn=None,
    regularization_param: float = None,
    precomputed: bool = False,
):
    """Add a sample to the Cholesky factorization.

    Args:
        c_and_lower: A tuple containing the Cholesky factor and a boolean variable
            indicating whether it is the upper or lower factor.
        X: The sample used to compute the inverse. Needed to compute the vector
            ``kernel_fn(X, y)``.
        y: The new sample to be added to the inverse. Needed to compute the vector
            ``kernel_fn(y) + regularization_param``.
        kernel_fn: The kernel function used to compute the inverse.
        regularization_param: The regularization parameter :math:`\lambda > 0`.
        precomputed: Whether the kernel over the new sample is precomputed. If this is
            ``True``, then :py:obj:`X` and :py:obj:`y` are assumed to be a vector of the
            same shape as ``kernel_fn(X, y)`` and scalar ``kernel_fn(y)``, respectively.

    Returns:
        The modified Cholesky factor.

    """

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

    if precomputed is False:
        k = kernel_fn(X, y)
        a = kernel_fn(y) + regularization_param
    else:
        k = X
        a = y

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
    """Remove a sample from the Cholesky factorization.

    Currently, only the first and last samples can be removed from the inverse.
    Specifying :py:obj:`last=False` in the arguments removes the first sample.

    Args:
        c_and_lower: A tuple containing the Cholesky factor and a boolean variable
            indicating whether it is the upper or lower factor.
        last: Whether to remove the last sample or the first. Default is ``True``.

    Returns:
        The modified Cholesky factor.

    """

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
