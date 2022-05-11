"""
Kernel functions and helper utilities for kernel-based calculations.

Most of the commonly-used kernel functions are already implemented in
:py:mod:`sklearn.metrics.pairwise`. The RBF kernel and pairwise Euclidean distance
function is re-implemented here as an alternative, in case :py:mod:`sklearn` is
unavailable. Most, if not all of the kernel functions defined in
:py:mod:`sklearn.metrics.pairwise` should be compatible with the functions defined here.

Attention:

    In Matlab, data is typically ordered differently than in Python. In Matlab, data is
    ordered in columns, whereas in Python, data is ordered in **rows**. If you import
    data from a Matlab file, be sure to transpose it if needed to follow Python
    formatting, i.e. ``X = X.T``.

    For example, the data in ``X`` and ``Y`` should be organized as::

        X = [[--- x1 ---],
             [--- x2 ---],
             ...
             [--- xn ---]]

        Y = [[--- y1 ---],
             [--- y2 ---],
             ...
             [--- ym ---]]

"""

from functools import partial, reduce

import numpy as np

from gym_socks.utils.validation import check_array
from gym_socks.utils.validation import check_matrix


def check_pairwise_arrays(
    X,
    Y=None,
    dtype: np.dtype = np.float64,
    ensure_finite: bool = True,
    copy: bool = False,
):
    """Check pairwise arrays.

    Note:

        This function is intended as a simple replacement for the
        :py:func:`sklearn.metrics.pairwise.check_pairwise_arrays` function. Unlike
        :py:mod:`sklearn`, this function does not check sparse input data, and does not
        do sophisticated type checking or upcasting.

    Args:
        X: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.
        dtype: The data type of the resulting array.
        copy: Whether to create a forced copy of ``array``.
        ensure_finite: Whether to raise an error if the array is not finite.

    Returns:
        The validated arrays ``X`` and ``Y``.

    """

    if Y is None:
        X = check_array(X, dtype=dtype, ensure_finite=ensure_finite, copy=copy)
        Y = X

    else:
        X = check_array(X, dtype=dtype, ensure_finite=ensure_finite, copy=copy)
        Y = check_array(Y, dtype=dtype, ensure_finite=ensure_finite, copy=copy)

    return X, Y


def euclidean_distances(X, Y=None, squared: bool = False):
    """Compute the pairwise Euclidean distance matrix between points.

    Note:

        This function is intended as a simple replacement for the
        :py:func:`sklearn.metrics.pairwise.euclidean_distances` function. Unlike
        :py:mod:`sklearn`, this function does not check sparse input data, and does not
        do sophisticated type checking or upcasting.

    Args:
        X: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.
        squared: Whether the result is squared before returning.

    Returns:
        The matrix of pairwise Euclidean distances between points.

    """

    X, Y = check_pairwise_arrays(X, Y)

    XX = np.einsum("ij,ij->i", X, X)[:, np.newaxis]
    YY = np.einsum("ij,ij->i", Y, Y)[np.newaxis, :]

    distances = -2 * (X @ Y.T)
    distances += XX
    distances += YY

    np.maximum(distances, 0, out=distances)
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)


def check_pairwise_distances(D: np.ndarray, shape: tuple, copy: bool = True):
    """Validate the pairwise distance matrix.

    Performs checks to ensure the pairwise distance matrix is valid.

    Args:
        D: Pairwise distance matrix.
        shape: The desired shape of the array.
        copy: Whether to create a forced copy of ``D``.

    Returns:
        The validated matrix.

    """

    D = check_array(D, copy=copy)

    if D.shape != shape:
        raise ValueError(
            f"Expected pairwise distances to be of shape {shape}, "
            f"but got {D.shape} instead."
        )

    return D


def rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray = None,
    sigma: float = None,
    D: np.ndarray = None,
) -> np.ndarray:
    r"""RBF kernel function.

    Computes the pairwise evaluation of the RBF kernel on each vector in ``X`` and
    ``Y``. For example, if ``X`` has :math:`m` vecotrs, and ``Y`` has :math:`n` vectors,
    then the result is an :math:`m \times n` matrix :math:`K` where :math:`K_{ij} =
    k(x_i, y_j)`.

    .. math::

        K =
        \begin{bmatrix}
            k(x_1,y_1) & \cdots & k(x_1,y_n) \\
            \vdots & \ddots & \vdots \\
            k(x_m,y_1) & \cdots & k(x_m,y_n)
        \end{bmatrix}

    Note:

        This function is intended as a simple replacement for the
        :py:func:`sklearn.metrics.pairwise.rbf_kernel` function. Unlike
        :py:mod:`sklearn`, this function does not check sparse input data, and does not
        do sophisticated type checking or upcasting.

        The main difference between this implementation and the
        :py:func:`sklearn.metrics.pairwise.rbf_kernel` in
        :py:mod:`sklearn.metrics.pairwise` is that this function optionally allows you
        to specify a different distance metric in the event the data is non-Euclidean.

    Attention:

        If you are familiar with :py:func:`sklearn.metrics.pairwise.rbf_kernel`, note
        that the parameter :py:attr:`sigma` is *not* the same as the ``gamma`` parameter
        used by :py:func:`sklearn.metrics.pairwise.rbf_kernel`. However, they are
        related:

        .. math::

            \gamma = \frac{1}{2 \sigma^{2}}

    Args:
        X: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.
        sigma: Strictly positive real-valued kernel parameter.
        D: Pairwise distance matrix.

    Returns:
        Gram matrix of pairwise evaluations of the kernel function.

    """

    X, Y = check_pairwise_arrays(X, Y)

    if D is None:
        D = euclidean_distances(X, Y, squared=True)
    else:
        D = check_pairwise_distances(D, shape=(X.shape[0], Y.shape[0]))

    if sigma is None:
        sigma = np.median(D)
    else:
        assert sigma > 0, "sigma must be a strictly positive real value."

    D /= -2 * (sigma ** 2)
    np.exp(D, D)

    return D


def rbf_kernel_derivative(
    X: np.ndarray,
    Y: np.ndarray = None,
    sigma: float = None,
    D: np.ndarray = None,
) -> np.ndarray:

    X, Y = check_pairwise_arrays(X, Y)

    if D is None:
        D = euclidean_distances(X, Y, squared=False)
    else:
        D = check_pairwise_distances(D, shape=(X.shape[0], Y.shape[0]))

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
    D: np.ndarray = None,
) -> np.ndarray:
    """Abel kernel function.

    Args:
        X: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.
        sigma: Strictly positive real-valued kernel parameter.
        D: Pairwise distance matrix.

    Returns:
        Gram matrix of pairwise evaluations of the kernel function.

    """

    X, Y = check_pairwise_arrays(X, Y)

    if D is None:
        D = euclidean_distances(X, Y, squared=False)
    else:
        D = check_pairwise_distances(D, shape=(X.shape[0], Y.shape[0]))

    if sigma is None:
        sigma = np.median(D)
    else:
        assert sigma > 0, "sigma must be a strictly positive real value."

    D /= -sigma
    np.exp(D, D)

    return D


def delta_kernel(X: np.ndarray, Y: np.ndarray = None):
    r"""Delta (discrete) kernel function.

    The delta kernel is defined as :math:`k(x_{i}, x_{j}) = \delta_{ij}`, meaning the
    kernel returns a 1 if the vectors are the same, and a 0 otherwise. The vectors in
    ``X`` and ``Y`` should have discrete values, meaning each element in the vector should
    be a natural number or integer value.

    Args:
        X: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.

    Returns:
        Gram matrix of pairwise evaluations of the kernel function.

    """

    X, Y = check_pairwise_arrays(X, Y, dtype=int)

    D = np.all(X[:, :, None] == Y.T, axis=1).astype(int)

    return D


def _hybrid_distances(
    X: np.ndarray, Q: np.ndarray, Y: np.ndarray = None, R: np.ndarray = None
):
    r"""Hybrid distance function.

    The hybrid distance function is a distance metric that works for states that are a
    combination of a continuous state and a discrete mode.

    .. math::

        d((x, q), (x', q')) =
        \begin{cases}
            \zeta(x - x'), & q = q' \\
		    1, & q \neq q'
        \end{cases}

    .. math::

        \zeta(x) = (2/\pi) \max_{1 \leq i \leq n} \tan^{-1} \vert x_{i} \vert

    """

    X, Y = check_pairwise_arrays(X, Y)

    d = delta_kernel(Q, R)

    diff = np.abs(X[:, :, None] - Y.T)
    diff = (2 / np.pi) * np.max(np.arctan(diff), axis=1)

    D = np.where(d == 0, 1, diff)

    return D


def hybrid_kernel(
    X: np.ndarray, Q: np.ndarray, Y: np.ndarray = None, R: np.ndarray = None
):
    r"""Hybrid systems kernel.

    In a hybrid system, we split the sample according to the mode ``Q``. The vectors in
    ``Q`` and ``R`` should have discrete values, meaning each element in the vector
    should be a natural number or integer value.

    .. math::

        d((x, q), (x', q')) =
        \begin{cases}
            \zeta(x - x'), & q = q' \\
		    1, & q \neq q'
        \end{cases}

    .. math::

        \zeta(x) = (2/\pi) \max_{1 \leq i \leq n} \tan^{-1} \vert x_{i} \vert

    .. math::

        k(x, q, x', q') = 1 - d((x, q), (x', q'))

    Args:
        X: A 2D array with observations oganized in ROWS.
        Q: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.
        R: A 2D array with observations oganized in ROWS.

    Returns:
        Gram matrix of pairwise evaluations of the kernel function.

    """

    D = _hybrid_distances(X, Q, Y, R)

    return 1 - np.sqrt(D)


def regularized_inverse(
    G: np.ndarray,
    l: float = None,
    copy: bool = True,
) -> np.ndarray:
    r"""Regularized inverse.

    Computes the regularized matrix inverse.

    .. math::

        W = (G + \lambda M I)^{-1}, \quad
        G \in \mathbb{R}^{n \times n}, \quad
        G_{ij} = k(x_{i}, y_{j})

    Args:
        G: The Gram (kernel) matrix.
        l: The regularization parameter :math:`\lambda > 0`.
        copy: Whether to create a forced copy of ``G``.

    Returns:
        Regularized matrix inverse.

    """

    G = check_matrix(G, ensure_square=True, copy=copy)

    if l is None:
        l = 1 / (len(G) ** 2)
    else:
        assert l > 0, "l must be a strictly positive real value."

    G[np.diag_indices_from(G)] += l * len(G)

    return np.linalg.inv(G)


def woodbury_inverse(
    A: np.ndarray,
    U: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
    precomputed: bool = False,
) -> np.ndarray:
    r"""Computes the matrix inverse using the Woodbury matrix identity.

    .. math::

        W = (A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}

    where :math:`A` is :math:`n \times n`, :math:`C` is :math:`k \times k`, :math:`U` is
    :math:`n \times k`, and :math:`V` is :math:`k \times n`.

    This function is useful for computing the regularized inverse in a more
    computationally efficient manner. This happens because the matrices :math:`A` and
    :math:`C` are typically easy to invert manually, either because they are known a
    priori, are constant matrices, or identity, leading to a smaller matrix inversion in
    the calculations.

    Example:

        >>> import numpy as np
        >>> from gym_socks.kernel.metrics import regularized_inverse
        >>> from gym_socks.kernel.metrics import woodbury_inverse
        >>> X = np.random.randn(100, 2)
        >>> G = X @ X.T
        >>> timeit regularized_inverse(G, 1)
        184 µs ± 7.14 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
        >>> A = (1 / 100) * np.identity(100)
        >>> C = np.identity(2)
        >>> timeit woodbury_inverse(A, X, C, X.T, precomputed=True)
        78.9 µs ± 1.67 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        >>> W1 = regularized_inverse(G, 1)
        >>> W2 = woodbury_inverse(A, X, C, X.T, precomputed=True)
        >>> np.allclose(W1, W2)
        True

    Args:
        A: A conformable square matrix. Must be nonsingular.
        U: A conformable matrix.
        C: A conformable square matrix. Must be nonsingular.
        V: A conformable matrix.
        precomputed: Whether ``A`` and ``C`` are the precomputed inverses.

    Returns:
        The resulting inverse matrix :math:`W`.

    """

    A = check_matrix(A, ensure_square=True)
    C = check_matrix(C, ensure_square=True)

    U = check_matrix(U)
    V = check_matrix(V)

    m, n = np.shape(A)
    k, l = np.shape(C)

    assert U.shape == (n, k), f"Expected U to be {(n, k)}, instead got: {U.shape}"
    assert V.shape == (k, n), f"Expected V to be {(k, n)}, instead got: {V.shape}"

    if precomputed is False:
        A_inv = np.linalg.inv(A)
        C_inv = np.linalg.inv(C)
    else:
        A_inv = A
        C_inv = C

    VA = V @ A_inv
    D = C_inv + VA @ U
    if np.isscalar(D):
        np.reciprocal(D, out=D)

        W = A_inv - D * A_inv @ U @ VA

    else:
        W = A_inv - A_inv @ U @ np.linalg.solve(D, VA)

    return W
