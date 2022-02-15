import numpy as np

from functools import partial

from gym_socks.kernel.metrics import regularized_inverse
from gym_socks.kernel.metrics import rbf_kernel


def maximum_mean_discrepancy(
    X, Y, kernel_fn=None, biased: bool = False, squared: bool = False
):
    r"""Maximum mean discrepancy between two empirical distributions.

    Compute the maximum mean discrepancy between two distributions (samples :math:`X`
    and :math:`Y`) using a kernel-based statistic. The biased estimator
    :math:`\widehat{\textrm{MMD}}_{b}^{2}(X, Y, \mathscr{H})` uses U-statistics, but can
    be negative. The unbiased estimator :math:`\widehat{\textrm{MMD}}_{u}^{2}(X, Y,
    \mathscr{H})` uses V-statistics.

    .. math::

        \widehat{\textrm{MMD}}_{b}^{2}(X, Y, \mathscr{H}) =
        \frac{1}{m^{2}} \sum_{i=1}^{m} \sum_{j=1}^{m} k(x_{i}, x_{j})
        + \frac{1}{n^{2}} \sum_{i=1}^{n} \sum_{j=1}^{n} k(y_{i}, y_{j})
        - \frac{2}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} k(x_{i}, y_{j})

    .. math::

        \widehat{\textrm{MMD}}_{u}^{2}(X, Y, \mathscr{H}) =
        \frac{1}{m(m - 1)} \sum_{i=1}^{m} \sum_{j \neq i}^{m} k(x_{i}, x_{j})
        + \frac{1}{n(n - 1)} \sum_{i=1}^{n} \sum_{j \neq i}^{n} k(y_{i}, y_{j})
        - \frac{2}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} k(x_{i}, y_{j})

    Args:
        X: The observations from distribution P organized in ROWS.
        Y: The observations from distribution Q organized in ROWS.
        kernel_fn: The kernel function is a function that returns an ndarray where
            each element is the pairwise evaluation of a kernel function. See sklearn.
            metrics.pairwise for more info. The default is the RBF kernel.
        biased: Whether to use the biased MMD.
        squared: Whether or not the result is squared.

    Returns:
        The scalar value representing the MMD.

    """

    m = len(X)
    n = len(Y)

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    GX = kernel_fn(X)
    GY = kernel_fn(Y)
    GXY = kernel_fn(X, Y)

    if biased is True:
        # Biased statistic.
        c1 = 1 / (m ** 2)
        c2 = 1 / (n ** 2)
        c3 = 2 / (m * n)
        A = np.sum(GX)
        B = np.sum(GY)
        C = np.sum(GXY)
        mmd = c1 * A + c2 * B - c3 * C
    else:
        # Unbiased statistic.
        c1 = 1 / (m * (m - 1))
        c2 = 1 / (n * (n - 1))
        c3 = 2 / (m * n)
        A = np.sum(GX - np.diag(np.diag(GX)))
        B = np.sum(GY - np.diag(np.diag(GY)))
        C = np.sum(GXY)
        mmd = c1 * A + c2 * B - c3 * C

    return mmd if squared else np.sqrt(mmd)


def witness_function(X, Y, t, kernel_fn=None):

    m = len(X)
    n = len(Y)

    if kernel_fn is None:
        kernel_fn = partial(rbf_kernel, sigma=1)

    c1 = 1 / m
    c2 = 1 / n
    A = np.sum(kernel_fn(X, t), axis=0)
    B = np.sum(kernel_fn(Y, t), axis=0)

    return c1 * A - c2 * B


def kernel_sum_rule(
    G,
    K,
    alpha,
    regularization_param: float = None,
):
    """Computes the kernel sum rule using Gram matrices.

    The kernel sum rule computes the marginal of a conditional distribution by
    integrating out the conditioning variables.

    .. math::

        P(Y) = \int_{\Omega} P(Y \mid X) P(d X)

    Args:
        G: The gram matrix of Y. Typically computed as `G = kernel_fn(Y)`.
        K: The Gram matrix of X. Typically computed as `K = kernel_fn(X)`.
        alpha: The coefficient vector corresponding to the marginal embedding of P(X).
        regularization_param: The regularization parameter of the matrix inverse.

    Returns:
        The coefficients of the marginal embedding of P(Y).

    """

    num_rows, _ = G.shape

    return np.einsum(
        "ii,ij,j->j",
        np.linalg.inv(G + regularization_param * num_rows * np.identity(num_rows)),
        K,
        alpha,
    )


def kernel_chain_rule(
    G,
    K,
    alpha,
    regularization_param: float = None,
):
    """Computes the kernel chain rule using Gram matrices.

    The kernel chain rule computes the joint distribution embedding by taking the
    product of conditional and marginal distribution embeddings.

    .. math::

        P(Y, X) = P(Y \mid X) P(X)

    Args:
        G: The gram matrix of Y. Typically computed as `G = kernel_fn(Y)`.
        K: The Gram matrix of X. Typically computed as `K = kernel_fn(X)`.
        alpha: The coefficient vector corresponding to the marginal embedding of P(X).
        regularization_param: The regularization parameter of the matrix inverse.

    Returns:
        The coefficients of the joint distribution embedding of P(Y, X).

    """

    num_rows, _ = G.shape

    return np.einsum(
        "ii,ij,jk->ik",
        np.linalg.inv(G + regularization_param * num_rows * np.identity(num_rows)),
        K,
        np.diag(alpha),
    )


def kernel_bayes_rule(
    G,
    K,
    Kx,
    alpha,
    regularization_param: float = None,
):
    r"""Computes the kernel Bayes' rule using Gram matrices.

    The kernel Bayes' rule.

    .. math::

        P(Y \mid x) = \frac{P(x \mid Y) P(Y)}{P(x)},
        \quad P(x) = \int_{\Omega} P(x \mid y) P(d y)

    Args:
        G: The gram matrix of Y. Typically computed as `G = kernel_fn(Y)`.
        K: The Gram matrix of X. Typically computed as `K = kernel_fn(X)`.
        Kx: The feature vector of x. Typically computed as `Kx = kernel_fn(X, x)`.
        alpha: The coefficient vector corresponding to the marginal embedding of P(X).
        regularization_param: The regularization parameter of the matrix inverse.

    Returns:
        The coefficients of the conditional distribution embedding of P(Y | x).

    """

    num_rows, _ = G.shape

    W = np.linalg.inv(G + regularization_param * num_rows * np.identity(num_rows))

    L = kernel_chain_rule(G, K, alpha, regularization_param=regularization_param)
    D = np.diag(kernel_sum_rule(G, K, alpha, regularization_param=regularization_param))

    return np.einsum(
        "ji,ii,jk,kl,l->jl",
        L,
        np.linalg.ing(
            np.linalg.matrix_power(D @ K, 2)
            + regularization_param * num_rows * np.identity(num_rows)
        ),
        K,
        D,
        Kx,
    )
