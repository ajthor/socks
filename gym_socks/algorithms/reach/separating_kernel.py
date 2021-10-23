"""Separating kernel classifier.

Separating kernel classifier, useful for forward stochastic reachability analysis.

References:
    .. [1] `Learning sets with separating kernels, 2014
            De Vito, Ernesto, Lorenzo Rosasco, and Alessandro Toigo.
            Applied and Computational Harmonic Analysis 37(2)`_

    .. [2] `Learning Approximate Forward Reachable Sets Using Separating Kernels, 2021
            Adam J. Thorpe, Kendric R. Ortiz, Meeko M. K. Oishi
            Learning for Dynamics and Control,
            <https://arxiv.org/abs/2011.09678>`_

"""
from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface
from gym_socks.kernel.metrics import abel_kernel, regularized_inverse

from gym_socks.utils import generate_batches

import numpy as np


class SeparatingKernelClassifier(AlgorithmInterface):
    """Separating kernel classifier."""

    def __init__(self, kernel_fn=None, regularization_param=None, *args, **kwargs):
        """Initialize the algorithm."""
        super().__init__(*args, **kwargs)

        self.kernel_fn = kernel_fn
        self.regularization_param = regularization_param

    def _validate_params(self, S):

        if self.kernel_fn is None:
            self.kernel_fn = partial(abel_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

    def _validate_data(self, X):

        if X is None:
            raise ValueError("Must supply a sample.")

    def fit(self, X: np.ndarray):
        """Fit separating kernel classifier.

        Args:
            X : Data drawn from distribution.

        Returns:
            self : Instance of SeparatingKernelClassifier

        """

        self._validate_data(X)
        X = np.array(X)

        K = self.kernel_fn(X)

        W = regularized_inverse(
            X, kernel_fn=self.kernel_fn, regularization_param=self.regularization_param
        )

        tau = 1 - np.min(np.diagonal((1 / len(X)) * K.T @ W @ K))

        self.X = X
        self.W = W

        self.tau = tau

        return self

    def predict(self, X: np.ndarray):
        """Predict using the separating kernel classifier.

        Args:
            X : Evaluation points where the separating kernel classifier is evaluated.

        Returns:
            y (bool) : Boolean indicator of classifier.

        """

        self._validate_data(X)
        T = np.array(X)

        K = self.kernel_fn(self.X, T)

        C = np.diagonal((1 / len(self.X)) * K.T @ self.W @ K)

        return C >= 1 - self.tau
