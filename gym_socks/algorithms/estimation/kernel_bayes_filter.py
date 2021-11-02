"""Kernel Bayes filter for partially observable state estimation.

The algorithm uses kernel distribution embeddings and kernel-based probability rules to compute an estimate of the state of the system based on received actions and observations.

Note:
    This algorithm implements an `update` function, which is used to update the estimate of the internal belief state. It should be used as part of the simulation loop, after the action and observation are generated.

Example:

    >>> from gym_socks.algorithms.estimation import KernelBayesFilter
    >>> bayes = KernelBayesFilter()
    >>> bayes.fit(S)
    >>> for t in range(env.num_time_steps):
    >>>     action = policy(time=t, state=env.state)
    >>>     obs, c, d, _ = env.step(action)
    >>>     bayes.update(action, observation)
    >>>     prediction = bayes.predict()

"""

from functools import partial

import gym_socks
from gym_socks.algorithms.algorithm import AlgorithmInterface

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse

from gym_socks.envs.sample import transpose_sample

import numpy as np


def kernel_bayes_filter(
    S: np.ndarray,
    kernel_fn=None,
    regularization_param: float = None,
    verbose: bool = False,
) -> np.ndarray:
    """Kernel bayes filter.

    Args:
        S: Sample of (x, u, y) tuples taken iid from the system evolution. The sample
            should be in the form of a list of tuples.
        regularization_param: Regularization parameter used in the solution to the
            regularized least-squares problem.
        verbose: Boolean flag to indicate verbose output.

    Returns:
        The fitted model.

    """

    alg = KernelBayesFilter(
        kernel_fn=kernel_fn,
        regularization_param=regularization_param,
        verbose=verbose,
    )
    alg.fit(S)
    return alg


class KernelBayesFilter(AlgorithmInterface):
    """Kernel bayes filter.

    Args:
        regularization_param: Regularization parameter used in the solution to the
            regularized least-squares problem.
        verbose: Boolean flag to indicate verbose output.

    Returns:
        The fitted model.

    """

    def __init__(
        self,
        kernel_fn=None,
        regularization_param: float = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.kernel_fn = kernel_fn
        self.regularization_param = regularization_param

        self.verbose = verbose

        self._belief_coeffs = None
        self._belief_data = None

    @property
    def belief_state_coeffs(self):
        """The belief state coefficients."""
        return self._belief_coeffs

    @property
    def belief_state_data(self):
        """The belief state data points."""
        return self._belief_data

    def _validate_params(self, S):

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

    def _validate_data(self, S):

        if S is None:
            raise ValueError("Must supply a sample.")

    def fit(self, S: np.ndarray):
        """Run the algorithm.

        Args:
            S : Sample of (x, u, y) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.

        Returns:
            Instance of KernelBayesFilter class.

        """

        self._validate_data(S)
        self._validate_params(S)

        self.len_sample = len(S)

        X, U, Y, Z = transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)
        Z = np.array(Z)

        self._belief_data = Y
        self._belief_coeffs = np.full(
            (self.len_sample,), 1 / self.len_sample, dtype=np.float32
        )

        self.CXX = self.kernel_fn(X)
        self.CUu = lambda u: self.kernel_fn(U, u)

        self.W1 = regularized_inverse(
            X=X,
            U=U,
            regularization_param=self.regularization_param,
            kernel_fn=self.kernel_fn,
        )

        self.CYY = self.kernel_fn(Y)

        self.W2 = regularized_inverse(
            X=Y,
            U=U,
            regularization_param=self.regularization_param,
            kernel_fn=self.kernel_fn,
        )

        self.K = self.kernel_fn(Z)
        self.Kz = lambda z: self.kernel_fn(Z, z)

        return self

    def update(self, action: np.ndarray, observation: np.ndarray):
        """Update the belief state.

        Args:
            action : The action of the system.
            observation : The observation received from the system.

        Returns:
            Instance of KernelBayesFilter class.

        """

        CUU = self.CUu(action)

        a = self.W1 @ (self.CXX * CUU) @ self._belief_coeffs

        D = np.diag(self.W2 @ (self.CYY * CUU) @ a)
        DK = D @ self.K

        self._belief_coeffs = DK @ np.linalg.solve(
            (
                np.linalg.matrix_power(DK, 2)
                + self.regularization_param
                * self.len_sample
                * np.identity(self.len_sample)
            ),
            D @ self.Kz(observation),
        )

        return self

    def predict(self) -> np.ndarray:
        """Predict.

        Returns:
            The predicted state of the system.

        """

        return self.B @ self._belief_data
