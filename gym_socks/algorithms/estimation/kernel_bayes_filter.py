"""Kernel Bayes filter for partially observable state estimation.

The algorithm uses kernel distribution embeddings and kernel-based probability rules to
compute an estimate of the state of the system based on received actions and
observations.

Note:
    This algorithm implements an `update` function, which is used to update the estimate
    of the internal belief state. It should be used as part of the simulation loop,
    after the action and observation are generated.

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

import numpy as np

from gym import Env

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse

from gym_socks.sampling import sample_fn
from gym_socks.sampling.transform import transpose_sample

from gym_socks.utils.validation import check_array

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

logger = logging.getLogger(__name__)


@sample_fn
def kernel_bayes_sampler(env: Env, state_sampler, action_sampler):
    """Sample function for kernel Bayes' filter algorithm.

    Args:
        env: The environment to sample from.
        state_sampler: The state sample function.
        action_sampler: The action sample function.

    Returns:
        Tuples of state, action, next state, and the associated observation.

    """

    state = next(state_sampler)
    action = next(action_sampler)

    env.state = state
    observation, *_ = env.step(action=action)
    next_state = env.state

    return state, action, next_state, observation


def kernel_bayes_filter(
    S: np.ndarray,
    initial_condition: np.ndarray = None,
    kernel_fn=None,
    regularization_param: float = None,
    verbose: bool = False,
) -> np.ndarray:
    """Kernel bayes filter.

    Args:
        S: Sample of (x, u, y) tuples taken iid from the system evolution. The sample
            should be in the form of a list of tuples.
        initial_condition: Initial state of the system.
        kernel_fn: Kernel function used by the estimator.
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
    alg.fit(S, initial_condition)
    return alg


class KernelBayesFilter(object):
    """Kernel bayes filter.

    Args:
        kernel_fn: Kernel function used by the estimator.
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

    def fit(self, S: list, initial_condition: np.ndarray = None):
        """Run the algorithm.

        Args:
            S : Sample of (x, u, y, z) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.
            initial_condition: Initial state of the system.

        Returns:
            Instance of KernelBayesFilter class.

        """

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

        self.len_sample = len(S)

        X, U, Y, Z = transpose_sample(S)
        X = check_array(X)
        U = check_array(U)
        Y = check_array(Y)
        Z = check_array(Z)

        logger.debug("Computing covariance matrices.")

        # Compute Gram (kernel) matrices.
        self.CXX = self.kernel_fn(X)
        self.CYY = self.kernel_fn(Y)

        CUU = self.kernel_fn(U)

        self.K = self.kernel_fn(Z)

        # Define feature vectors as functions.
        self.CUu = lambda u: self.kernel_fn(U, u)
        self.Kz = lambda z: self.kernel_fn(Z, z)

        logger.debug("Computing regularized inverses.")
        self.W1 = regularized_inverse(
            self.CXX * CUU, regularization_param=self.regularization_param
        )

        self.W2 = regularized_inverse(
            self.CYY * CUU, regularization_param=self.regularization_param
        )

        logger.debug("Initialize belief state.")
        self._belief_data = Y
        belief_shape = (self.len_sample, 1)

        if initial_condition is not None:
            # initial_belief = check_array(initial_belief, copy=True)

            # # Check shape of initial belief state.
            # if initial_belief.shape != belief_shape:
            #     raise ValueError(
            #         f"Expected initial belief to have shape {belief_shape}.\n"
            #         f"Got shape {initial_belief.shape} instead."
            #     )

            logger.debug("Computing initial belief state coefficients.")
            initial_condition = np.atleast_2d(np.asarray(initial_condition))

            self._belief_coeffs = regularized_inverse(
                self.CXX, regularization_param=self.regularization_param, copy=True
            ) @ self.kernel_fn(X, initial_condition)

        else:

            # Initialize belief state to default distribution (uniform over data).
            self._belief_coeffs = np.full(
                belief_shape, 1 / self.len_sample, dtype=np.float32
            )

        return self

    def predict(self, action: np.ndarray, observation: np.ndarray):
        """Update the belief state.

        Args:
            action : The action of the system.
            observation : The observation received from the system.

        Returns:
            The predicted state of the system.

        """

        logger.debug("Computing belief update.")
        CUU = self.CUu(action)

        # a = self.W1 @ (self.CXX * CUU) @ self._belief_coeffs
        # a = np.squeeze(a)
        a = np.einsum("ii,ij,jk->j", self.W1, self.CXX * CUU, self._belief_coeffs)

        # D = np.diag(self.W2 @ (self.CYY * CUU) @ a)
        D = np.diag(np.einsum("ii,ij,j->j", self.W2, self.CYY * CUU, a))

        DK = D @ self.K

        DK2 = np.linalg.matrix_power(DK, 2)
        DK2[np.diag_indices_from(DK2)] += self.regularization_param * self.len_sample

        self._belief_coeffs = DK @ np.linalg.solve(DK2, D @ self.Kz(observation))

        # self._belief_coeffs = DK @ np.linalg.solve(
        #     np.linalg.matrix_power(DK, 2)
        #     + self.regularization_param
        #     * self.len_sample
        #     * np.identity(self.len_sample),
        #     D @ self.Kz(observation),
        # )

        return (self._belief_coeffs.T @ self._belief_data)[0]
