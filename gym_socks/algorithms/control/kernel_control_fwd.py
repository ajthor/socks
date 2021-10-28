"""Forward in time stochastic optimal control.

References:
    .. [1] `Stochastic Optimal Control via
            Hilbert Space Embeddings of Distributions, 2021
            Adam J. Thorpe, Meeko M. K. Oishi
            IEEE Conference on Decision and Control,
            <https://arxiv.org/abs/2103.12759>`_
"""

from functools import partial

import gym_socks

from gym_socks.envs.policy import BasePolicy
from gym_socks.algorithms.control.control_common import compute_solution
from gym_socks.kernel.metrics import rbf_kernel, regularized_inverse
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

import numpy as np

from tqdm.auto import tqdm


def kernel_control_fwd(
    S: np.ndarray,
    A: np.ndarray,
    cost_fn=None,
    constraint_fn=None,
    heuristic: bool = False,
    regularization_param: float = None,
    kernel_fn=None,
    verbose: bool = True,
):
    """Stochastic optimal control policy forward in time.

    Computes the optimal control action at each time step in a greedy fashion. In other
    words, at each time step, the policy optimizes the cost function from the current
    state. It does not "look ahead" in time.

    Args:
        S: Sample taken iid from the system evolution.
        A: Collection of admissible control actions.
        cost_fn: The cost function. Should return a real value.
        constraint_fn: The constraint function. Should return a real value.
        heuristic: Whether to use the heuristic solution instead of solving the LP.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.
        verbose: Whether the algorithm should print verbose output.

    Returns:
        The policy.
    """

    alg = KernelControlFwd(
        cost_fn=cost_fn,
        constraint_fn=constraint_fn,
        heuristic=heuristic,
        regularization_param=regularization_param,
        kernel_fn=kernel_fn,
        verbose=verbose,
    )
    alg.train(S, A)

    return alg


class KernelControlFwd(BasePolicy):
    """Stochastic optimal control policy forward in time.

    Computes the optimal control action at each time step in a greedy fashion. In other
    words, at each time step, the policy optimizes the cost function from the current
    state. It does not "look ahead" in time.

    Args:
        cost_fn: The cost function. Should return a real value.
        constraint_fn: The constraint function. Should return a real value.
        heuristic: Whether to use the heuristic solution instead of solving the LP.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.
        verbose: Whether the algorithm should print verbose output.

    """

    def __init__(
        self,
        cost_fn=None,
        constraint_fn=None,
        heuristic: bool = False,
        regularization_param: float = None,
        kernel_fn=None,
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.cost_fn = cost_fn
        self.constraint_fn = constraint_fn

        self.heuristic = heuristic

        self.kernel_fn = kernel_fn
        self.regularization_param = regularization_param

        self.verbose = verbose

    def _validate_params(self, S=None, A=None):

        if S is None:
            raise ValueError("Must supply a sample.")

        if A is None:
            raise ValueError("Must supply a sample.")

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

        if self.cost_fn is None:
            raise ValueError("Must supply a cost function.")

        if self.constraint_fn is None:
            # raise ValueError("Must supply a constraint function.")
            self._constrained = False
        else:
            self._constrained = True

    def _validate_data(self, S):

        if S is None:
            raise ValueError("Must supply a sample.")

    def train(self, S: np.ndarray, A: np.ndarray):
        """Train the algorithm.

        Args:
            S: Sample taken iid from the system evolution.
            A: Collection of admissible control actions.

        Returns:
            self: An instance of the KernelControlFwd algorithm class.

        """

        self._validate_data(S)
        self._validate_data(A)
        self._validate_params(S=S, A=A)

        X, U, Y = gym_socks.envs.sample.transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        self.X = X

        A = np.array(A)

        self.A = A

        gym_socks.logger.debug("Computing matrix inverse.")
        self.W = regularized_inverse(
            X,
            U=U,
            kernel_fn=self.kernel_fn,
            regularization_param=self.regularization_param,
        )

        gym_socks.logger.debug("Computing covariance matrix.")
        self.CUA = self.kernel_fn(U, A)

        gym_socks.logger.debug("Computing cost function.")
        self.cost_fn = partial(self.cost_fn, state=Y)

        gym_socks.logger.debug("Computing constraint function.")
        if self.constraint_fn is not None:
            self.constraint_fn = partial(self.constraint_fn, state=Y)

        return self

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(state)

        # Compute covariance matrix.
        CXT = self.kernel_fn(self.X, T)
        # Compute beta.
        betaXT = self.W @ (CXT * self.CUA)

        # Compute cost vector.
        C = np.matmul(np.array(self.cost_fn(time=time), dtype=np.float32), betaXT)

        # Compute constraint vector.
        D = None
        if self.constraint_fn is not None:
            D = np.matmul(
                np.array(self.constraint_fn(time=time), dtype=np.float32), betaXT
            )

        # Compute the solution to the LP.
        gym_socks.logger.debug("Computing solution to the LP.")
        sol = compute_solution(C, D, heuristic=self.heuristic)
        idx = np.argmax(sol)
        return np.array(self.A[idx], dtype=np.float32)
