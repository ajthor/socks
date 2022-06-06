from abc import ABC, abstractmethod

import numpy as np

from scipy.linalg import cholesky
from scipy.linalg import cho_solve

from functools import partial

from gym_socks.algorithms.base import RegressorMixin

from gym_socks.utils.validation import check_array
from gym_socks.utils.validation import check_matrix


def _regression_score(y_true, y_pred):
    return np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean()


class ConditionalEmbedding(RegressorMixin):
    r"""Conditional distribution embedding.

    Computes the conditional distribution embedding using sample data.

    This class is primarily used internally to compute the regularized matrix inverse
    and predict the values of a function.

    .. math::

        m(x) = f^{\top} (G + \lambda M I)^{-1} K

    Args:
        regularization_param: The regularization parameter :math:`\lambda`.

    """

    _cholesky_lower = False  # Whether to use lower triangular cholesky factorization.

    def __init__(
        self,
        regularization_param: float = None,
    ):
        self.regularization_param = regularization_param

        self._alpha = None

    def fit(self, G: np.ndarray, y: np.ndarray):
        r"""Fit the model.

        Computes the matrix inverse :math:`W = (G + \lambda M I)^{-1}`.

        Args:
            G: The (kernel) gram matrix. Usually computed as ``kernel_fn(X_train)``.
            y: The output values of the function.

        Returns:
            self

        """

        G = check_matrix(G, ensure_square=True, copy=True)

        if self.regularization_param is None:
            self.regularization_param = 1 / (len(G) ** 2)
        else:
            assert (
                self.regularization_param > 0
            ), "regularization_param must be a strictly positive real value."

        G[np.diag_indices_from(G)] += self.regularization_param * len(G)

        try:
            self._L = cholesky(G, lower=self._cholesky_lower)
        except np.linalg.LinAlgError as e:
            e.args = (
                "The Gram matrix is not positive definite. "
                "Try increasing the regularization parameter.",
            ) + e.args
            raise

        self._alpha = cho_solve((self._L, self._cholesky_lower), y, check_finite=False)

        return self

    def predict(self, K: np.ndarray):
        r"""Predict the values.

        Computes the product :math:`y^{\top} W K`.

        Args:
            K: The kernel matrix. Usually computed as ``kernel_fn(X_train, X_test)``.

        Returns:
            An ndarray of predicted y values.

        """

        K = check_matrix(K, copy=True)

        return np.einsum("ij,ik->k", self._alpha, K)

    def score(self, y_test: np.ndarray, K: np.ndarray, y_train: np.ndarray):
        return _regression_score(y_test, self.predict(y_train, K))


def _sample_uniform_ball(num_samples: int, dim: int = 3, radius: float = 1):
    """Generate uniform random points in an ND ball.

    Args:
        num_samples: Number of points to generate.
        dim: The dimensionality of the ball.
        radius: The radius of the ball.

    Returns:
        A sample of points of shape `(dim, num_samples)`.

    """

    S = np.random.randn(dim, num_samples)
    S /= np.linalg.norm(S, axis=0)

    U = np.random.random(num_samples) ** (1 / dim)

    return radius * (S * U).T


class GenerativeModel(ConditionalEmbedding):
    r"""Generative model.

    The generative model is a distribution over functions, meaning given a set of
    inputs, it outputs a random function that *could* fit the data. Unlike the
    :py:class:`ConditionalEmbedding` class, the :py:class:`GenerativeModel` class does
    **not** compute the *best fit* function that matches the mean of the data. Instead,
    it computes a function that is within a Hilbert space ball of the embedding.

    In other words, if :math:`m \in \mathscr{H}` is the conditional embedding, then the
    generative model produces a random function :math:`f \in \mathscr{H}` such that
    :math:`\lVert m - f \rVert \leq \delta`, where :math:`\delta` is the radius of the
    ball.

    This is useful in generating functions or trajectories which are superficially
    similar to the true embedding.

    Important:

        It is important to note that the generated functions are close to the empirical
        conditional embedding, not the true embedding. Therefore, it should not be
        expected that the generated functions obey the same confidence bounds as the
        conditional embedding.

    Args:
        regularization_param: The regularization parameter :math:`\lambda`.

    """

    def sample(
        self,
        K: np.ndarray,
        num_samples: int = 1,
        radius: float = None,
    ):
        r"""Generate predicted functions.

        Generates a function that is within ``radius`` of the conditional embedding in
        the RKHS norm.

        Args:
            K: The kernel matrix. Usually computed as ``kernel_fn(X_train, X_test)``.
            num_samples: Number of samples to compute.
            radius: The radius of the Hilbert space ball.

        Returns:
            An ndarray of predicted y values.

        """

        radius = 1 if radius is None else radius

        K = check_matrix(K, copy=True)

        # Generate points in a Hilbert space ball centered at the mean.
        coeffs = _sample_uniform_ball(num_samples, len(self._alpha), radius)
        coeffs += self._alpha.T

        y_samples = coeffs @ K

        return y_samples

    def score(self, y: np.ndarray):
        raise NotImplementedError()
