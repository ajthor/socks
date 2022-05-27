from abc import ABC, abstractmethod

import numpy as np

from scipy.linalg import cholesky
from scipy.linalg import cho_solve
from scipy.linalg import solve_triangular

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
        self._G = G

        if self.regularization_param is None:
            self.regularization_param = 1 / (len(G) ** 2)
        else:
            assert (
                self.regularization_param > 0
            ), "regularization_param must be a strictly positive real value."

        G[np.diag_indices_from(G)] += self.regularization_param * len(G)
        try:
            L = cholesky(G, lower=True)
        except np.linalg.LinAlgError as e:
            e.args = (
                "The Gram matrix is not positive definite. "
                "Try increasing the regularization parameter.",
            ) + e.args
            raise

        self._alpha = cho_solve((L, True), y, check_finite=False)

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
