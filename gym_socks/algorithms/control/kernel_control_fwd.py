r"""Forward in time stochastic optimal control.

The policy is specified as a sequence of stochastic kernels :math:`\pi = \lbrace
\pi_{0}, \pi_{1}, \ldots, \pi_{N-1} \rbrace`. At each time step, the problem seeks
to solve a constrained optimization problem.

.. math::
    :label: optimization_problem

    \min_{\pi_{t}} \quad & \int_{\mathcal{U}} \int_{\mathcal{X}} f_{0}(y, u)
    Q(\mathrm{d} y \mid x, u) \pi_{t}(\mathrm{d} u \mid x) \\
    \textnormal{s.t.} \quad & \int_{\mathcal{U}} \int_{\mathcal{X}} f_{i}(y, u)
    Q(\mathrm{d} y \mid x, u) \pi_{t}(\mathrm{d} u \mid x), i = 1, \ldots, m

Using kernel embeddings of disrtibutions, assuming the cost and constraint functions
:math:`f_{0}, \ldots, f_{m}` are in an RKHS, the integral with respect to the stochastic
kernel :math:`Q` and the policy :math:`\pi_{t}` can be approximated by an inner product,
i.e. :math:`\int_{\mathcal{X}} f_{0}(y) Q(\mathrm{d} y \mid x, u) \approx \langle f_{0},
\hat{m}(x, u) \rangle`. We use this to construct an approximate problem to
:eq:`optimization_problem` and solve for a policy represented as an element in an RKHS.

.. math::

    p_{t}(x) = \sum_{i=1}^{P} \gamma_{i}(x) k(\tilde{u}_{i}, \cdot)

The approximate problem is a linear program (LP), and can be solved efficiently using
standard optimization solvers.

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


def kernel_control_fwd(
    S: np.ndarray,
    A: np.ndarray,
    cost_fn=None,
    constraint_fn=None,
    regularization_param: float = None,
    kernel_fn=None,
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
        regularization_param=regularization_param,
        kernel_fn=kernel_fn,
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
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.

    """

    _cholesky_lower = False  # Whether to use lower triangular cholesky factorization.

    def __init__(
        self,
        cost_fn=None,
        constraint_fn=None,
        regularization_param: float = None,
        kernel_fn=None,
    ):

        self.cost_fn = cost_fn
        self.constraint_fn = constraint_fn

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

        if self.constraint_fn is not None:
            logger.debug("Computing constraint function.")
            self.constraint_fn = partial(self.constraint_fn, state=Y)

        return self

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        state = np.atleast_2d(np.asarray(state, dtype=float))

        T = np.array(state)

        # Compute cost vector.
        CXT = self.kernel_fn(self.X, T) * self.CUA
        beta = cho_solve(
            (self._L, self._cholesky_lower),
            self.cost_fn(time=time),
        )
        C = beta @ CXT

        # Compute constraint vector.
        D = None
        if self.constraint_fn is not None:
            beta = cho_solve(
                (self._L, self._cholesky_lower),
                self.constraint_fn(time=time),
            )
            D = beta @ CXT

        # Compute the solution to the LP.
        logger.debug("Computing solution to the LP.")
        sol = _compute_solution(C, D)
        sol = normalize(sol)  # Normalize the vector.
        idx = np.random.choice(len(self.A), p=sol)
        return np.array(self.A[idx], dtype=float)
