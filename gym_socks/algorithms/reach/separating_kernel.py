"""Separating kernel classifier.

Separating kernel classifier, useful for forward stochastic reachability analysis.

"""

from functools import partial

import numpy as np

from gym_socks.algorithms.base import ClassifierMixin
from gym_socks.kernel.metrics import abel_kernel
from gym_socks.kernel.metrics import regularized_inverse

from gym_socks.utils.validation import check_array


class SeparatingKernelClassifier(ClassifierMixin):
    r"""Separating kernel classifier.

    A kernel-based support classifier for unknown distributions. Given a set of data
    taken i.i.d. from the distribution, the `SeparatingKernelClassifier` constructs a
    kernel-based classifier of the support of the distribution based on the theory of
    separating kernels.

    Args:
        kernel_fn: The kernel function used by the algorithm.
        l: The regularization parameter :math:`\lambda > 0`. Determines the smoothness
            of the solution.

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
        l: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.kernel_fn = kernel_fn
        self.l = l

    def fit(self, X: np.ndarray):
        """Fit separating kernel classifier.

        Note:

            The sample used by the classifier is from the marginal distribution, not the
            joint or conditional. Thus, the data should be an array of points organized
            such that each point occupies a single row in a 2D-array.

        Args:
            X: Data drawn from distribution.

        Returns:
            self: Instance of SeparatingKernelClassifier

        """

        if self.kernel_fn is None:
            self.kernel_fn = partial(abel_kernel, sigma=0.1)

        self._X = check_array(X)

        if self.l is None:
            self.l = 1 / len(self._X)

        # Precompute matrices which are used in prediction.
        K = self.kernel_fn(X)
        self._W = regularized_inverse(K, self.l, copy=True)

        self._tau = 1 - np.min(np.diagonal(K.T @ self._W @ K))

        return self

    def predict(self, T: np.ndarray) -> list:
        """Predict using the separating kernel classifier.

        Args:
            T: Evaluation points where the separating kernel classifier is evaluated.

        Returns:
            Boolean indicator of classifier.

        """

        T = check_array(T)

        K = self.kernel_fn(self._X, T)
        C = np.diagonal(K.T @ self._W @ K)

        return np.array(C >= 1 - self._tau, dtype=bool)

    def score(self):
        raise NotImplementedError
