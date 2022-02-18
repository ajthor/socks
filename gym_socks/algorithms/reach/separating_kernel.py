"""Separating kernel classifier.

Separating kernel classifier, useful for forward stochastic reachability analysis.

"""

from functools import partial

from gym_socks.algorithms.base import ClassifierMixin
from gym_socks.kernel.metrics import abel_kernel
from gym_socks.kernel.metrics import regularized_inverse

import numpy as np


class SeparatingKernelClassifier(ClassifierMixin):
    """Separating kernel classifier.

    A kernel-based support classifier for unknown distributions. Given a set of data taken iid from the distribution, the `SeparatingKernelClassifier` constructs a kernel-based classifier of the support of the distribution based on the theory of separating kernels.

    Note:
        The sample used by the classifier is from the marginal distribution, not the
        joint or conditional. Thus, the data should be an array of points organized such
        that each point occupies a single row in a 2D-array.

    Args:
        kernel_fn: The kernel function used by the classifier.
        regularization_param: The regularization parameter used in the regularized
            least-squares problem. Determines the smoothness of the solution.

    Example:
        >>> from gym_socks.algorithms.reach import SeparatingKernelClassifier
        >>> from gym_socks.kernel.metrics import abel_kernel
        >>> from functools import partial
        >>> kernel_fn = partial(abel_kernel, sigma=0.1)
        >>> classifier = SeparatingKernelClassifier(kernel_fn)
        >>> classifier.fit(S)
        >>> classifications = classifier.predict(T)

    """

    def __init__(
        self,
        kernel_fn=None,
        regularization_param=None,
        *args,
        **kwargs,
    ):
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
            X: Data drawn from distribution.

        Returns:
            self: Instance of SeparatingKernelClassifier

        """

        self._validate_data(X)
        X = np.array(X)

        self.X = X

        K = self.kernel_fn(X)

        self.W = regularized_inverse(
            X, kernel_fn=self.kernel_fn, regularization_param=self.regularization_param
        )

        self.tau = 1 - np.min(np.diagonal((1 / len(X)) * K.T @ self.W @ K))

        return self

    def predict(self, T: np.ndarray) -> list:
        """Predict using the separating kernel classifier.

        Args:
            T: Evaluation points where the separating kernel classifier is evaluated.

        Returns:
            Boolean indicator of classifier.

        """

        self._validate_data(T)
        T = np.array(T)

        K = self.kernel_fn(self.X, T)

        C = np.diagonal((1 / len(self.X)) * K.T @ self.W @ K)

        return C >= 1 - self.tau

    def score(self):
        raise NotImplementedError
