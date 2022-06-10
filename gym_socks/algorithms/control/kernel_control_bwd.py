r"""Backward in time stochastic optimal control.

The backward in time (dynamic programming) stochastic optimal control algorithm computes
the control actions working backward in time from the terminal time step to the current
time step. It computes a sequence of "value" functions, and then as the system
evolves forward in time, it chooses a control action that optimizes the value function,
rather than the actual cost.

The policy is specified as a sequence of stochastic kernels :math:`\pi = \lbrace
\pi_{0}, \pi_{1}, \ldots, \pi_{N-1} \rbrace`. Typically, the cost is formulated as an
additive cost structure, where at each time step the system incurs a *stage cost*
:math:`g_{t}(x_{t}, u_{t})`, and at the final time step, it incurs a *terminal cost*
:math:`g_{N}(x_{N})`.

.. math::

    \min_{\pi} \quad \mathbb{E} \biggl[ g_{N}(x_{N}) +
    \sum_{t=0}^{N-1} g_{t}(x_{t}, u_{t}) \biggr]

In dynamic programming, we solve the problem iteratively, by considering each time step
independently. We can do this by defining a sequence of *value functions* :math:`V_{0},
\ldots, V_{N}` that describe a type of "overall" cost at each time step, starting with
the terminal cost :math:`V_{N}(x) = g_{N}(x)`, and then substituting the subsequent
value function into the current one. Then, the policy is chosen to minimize (or
maximize, as is the convention for RL) the value functions.

.. math::

    V_{t}(x)
    = \max_{\pi_{t}} \quad \int_{\mathcal{U}} \int_{\mathcal{X}} g_{t}(x_{t}, u_{t})
    + V_{t+1}(y) Q(\mathrm{d} y \mid x, u) \pi_{t}(\mathrm{d} u \mid x)

Note:
    See :py:mod:`examples.benchmark_tracking_problem` for a complete example.

"""

from functools import partial

import numpy as np

from scipy.linalg import cholesky
from scipy.linalg import cho_solve

from gym_socks.policies import BasePolicy
from gym_socks.algorithms.control.common import _compute_solution

from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.sampling.transform import transpose_sample

from gym_socks.utils import normalize
from gym_socks.utils.validation import check_array

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

logger = logging.getLogger(__name__)


def kernel_control_bwd(
    S: np.ndarray,
    A: np.ndarray,
    time_horizon: int = None,
    cost_fn=None,
    regularization_param: float = None,
    kernel_fn=None,
):
    """Stochastic optimal control policy backward in time.

    Computes the optimal policy using dynamic programming. The solution computes an approximation of the value functions starting at the terminal time and working backwards. Then, when the policy is evaluated, it moves forward in time, optimizing over the value functions and choosing the action which has the highest "value".

    Args:
        S: Sample taken iid from the system evolution.
        A: Collection of admissible control actions.
        cost_fn: The cost function. Should return a real value.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.

    """

    alg = KernelControlBwd(
        time_horizon=time_horizon,
        cost_fn=cost_fn,
        regularization_param=regularization_param,
        kernel_fn=kernel_fn,
    )
    alg.train(S, A)

    return alg


class KernelControlBwd(BasePolicy):
    """Stochastic optimal control policy backward in time.

    Computes the optimal policy using dynamic programming. The solution computes an approximation of the value functions starting at the terminal time and working backwards. Then, when the policy is evaluated, it moves forward in time, optimizing over the value functions and choosing the action which has the highest "value".

    Args:
        S: Sample taken iid from the system evolution.
        A: Collection of admissible control actions.
        cost_fn: The cost function. Should return a real value.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.

    """

    _cholesky_lower = False  # Whether to use lower triangular cholesky factorization.

    def __init__(
        self,
        time_horizon: int = None,
        cost_fn=None,
        regularization_param: float = None,
        kernel_fn=None,
    ):

        self.time_horizon = time_horizon

        self.cost_fn = cost_fn

        self.kernel_fn = kernel_fn
        self.regularization_param = regularization_param

    def train(self, S: np.ndarray, A: np.ndarray):
        """Train the algorithm.

        Args:
            S: Sample taken iid from the system evolution.
            A: Collection of admissible control actions.

        Returns:
            An instance of the KernelControlFwd algorithm class.

        """

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1 / (len(S) ** 2)
        else:
            assert (
                self.regularization_param > 0
            ), "regularization_param must be a strictly positive real value."

        if self.cost_fn is None:
            raise ValueError("Must supply a cost function.")

        X, U, Y = transpose_sample(S)
        X = check_array(X)
        U = check_array(U)
        Y = check_array(Y)

        self.X = X

        A = check_array(A)

        self.A = A

        logger.debug("Computing covariance matrices.")
        CXU = self.kernel_fn(X) * self.kernel_fn(U)
        CXU[np.diag_indices_from(CXU)] += self.regularization_param

        self.CUA = self.kernel_fn(U, A)
        CXY = self.kernel_fn(X, Y)

        try:
            logger.debug("Computing Cholesky factorization.")
            self._L = cholesky(
                CXU,
                lower=self._cholesky_lower,
                overwrite_a=True,
            )
        except np.linalg.LinAlgError as e:
            e.args = (
                "The Gram matrix is not positive definite. "
                "Try increasing the regularization parameter.",
            ) + e.args
            raise

        logger.debug("Computing cost function.")
        self.cost_fn = partial(self.cost_fn, state=Y)

        logger.debug("Computing value functions.")
        value_functions = np.zeros((self.time_horizon, len(Y)))
        value_functions[self.time_horizon - 1] = self.cost_fn(
            time=self.time_horizon - 1
        )

        for t in range(self.time_horizon - 2, -1, -1):

            beta = cho_solve((self._L, self._cholesky_lower), value_functions[t + 1])
            C = np.einsum("i,ij->ij", beta @ CXY, self.CUA)
            V = np.min(C, axis=1)

            value_functions[t] = self.cost_fn(time=t) + V

        self.value_functions = value_functions

        return self

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        state = np.atleast_2d(np.asarray(state, dtype=float))

        T = np.array(state)

        CXT = self.kernel_fn(self.X, T)
        beta = cho_solve((self._L, self._cholesky_lower), self.value_functions[time])
        C = beta @ (CXT * self.CUA)

        # Compute the solution to the LP.
        logger.debug("Computing solution to the LP.")
        sol = _compute_solution(C, None)
        sol = normalize(sol)  # Normalize the vector.
        idx = np.random.choice(len(self.A), p=sol)
        return np.array(self.A[idx], dtype=float)
