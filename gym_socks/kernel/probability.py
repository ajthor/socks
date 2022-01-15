import numpy as np

from gym_socks.kernel.metrics import regularized_inverse


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
    """Computes the kernel Bayes' rule using Gram matrices.

    The kernel Bayes' rule.

    .. math::

        P(Y \mid x) = (P(x \mid Y) P(Y))/P(x),
        where P(x) = \int_{\Omega} P(x \mid y) P(d y)

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
