from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag

from functools import partial

from gym_socks.algorithms.base import RegressorMixin

from sklearn.preprocessing import normalize


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

        self._W = None

    def fit(self, G: np.ndarray):
        r"""Fit the model.

        Computes the matrix inverse :math:`W = (G + \lambda M I)^{-1}`.

        Args:
            G: The (kernel) gram matrix. Usually computed as ``kernel_fn(X_train)``.

        Returns:
            self

        """

        s = np.shape(G)
        if len(s) != 2 and s[0] != s[1]:
            raise ValueError("Gram matrix must be square.")

        if self.regularization_param is None:
            self.regularization_param = 1 / (len(G) ** 2)
        else:
            assert (
                self.regularization_param > 0
            ), "regularization_param must be a strictly positive real value."

        I = np.zeros(np.shape(G))
        np.fill_diagonal(I, self.regularization_param * len(G))
        self._W = G + I
        return self

    def predict(self, y: np.ndarray, K: np.ndarray):
        r"""Predict the values.

        Computes the product :math:`y^{\top} W K`.

        Args:
            y: The output values of the function.
            K: The kernel matrix. Usually computed as ``kernel_fn(X_train, X_test)``.

        Returns:
            An ndarray of predicted y values.

        """

        return (y.T @ np.linalg.solve(self._W, K)).T

    def score(self, y_test: np.ndarray, K: np.ndarray, y_train: np.ndarray):
        return _regression_score(y_test, self.predict(y_train, K))

