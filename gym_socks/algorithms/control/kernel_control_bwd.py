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

from gym_socks.policies import BasePolicy
from gym_socks.algorithms.control.common import compute_solution

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse

from gym_socks.sampling.transform import transpose_sample

from gym_socks.utils.batch import batch_generator

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

logger = logging.getLogger(__name__)


def _compute_backward_recursion(
    Y,
    W,
    CXY,
    CUA,
    time_horizon=None,
    cost_fn=None,
    constraint_fn=None,
    value_functions=None,
    out=None,
    verbose=False,
):
    # Compute beta.
    logger.debug("Computing beta.")
    beta = np.einsum("ij,ik->ijk", CXY, CUA)

    pbar = ms_tqdm(
        total=time_horizon,
        bar_format=_progress_fmt,
        disable=False if verbose is True else True,
    )

    # set up empty array to hold value functions
    if out is None:
        out = np.zeros((time_horizon, len(Y)), dtype=np.float32)
    out[time_horizon - 1, :] = np.array(
        cost_fn(time=time_horizon - 1), dtype=np.float32
    )
    pbar.update()

    # run backwards in time and compute the safety probabilities
    for t in range(time_horizon - 2, -1, -1):

        Z = np.matmul(value_functions[t + 1, :], W)
        C = np.einsum("i,ijk->jk", Z, beta)

        V = np.zeros((len(Y),))

        if constraint_fn is not None:
            R = np.matmul(constraint_fn(time=t), W)
            D = np.einsum("i,ijk->jk", R, beta)

            for i in range(len(Y)):
                CA = C[i][D[i] <= 0]
                if CA.size == 0:
                    V[i] = np.min(C[i])
                else:
                    V[i] = np.min(CA)

        else:
            V = np.min(C, axis=1)

        # for i in range(len(Y)):
        #     sol = compute_solution(C[i], D[i])
        #     # idx = np.argmax(sol)
        #     # V =
        #     V[i] = sol

        out[t, :] = cost_fn(time=t) + V

        pbar.update()

    pbar.close()

    return out


def _compute_backward_recursion_batch(
    Y,
    W,
    CXY,
    CUA,
    time_horizon=None,
    cost_fn=None,
    constraint_fn=None,
    value_functions=None,
    out=None,
    batch_size=None,
    verbose=False,
):
    pbar = ms_tqdm(
        total=time_horizon,
        bar_format=_progress_fmt,
        disable=False if verbose is True else True,
    )

    # set up empty array to hold value functions
    if out is None:
        out = np.zeros((time_horizon + 1, len(Y)), dtype=np.float32)
    out[time_horizon - 1, :] = np.array(
        cost_fn(time=time_horizon - 1), dtype=np.float32
    )
    pbar.update()

    # run backwards in time and compute the safety probabilities
    for t in range(time_horizon - 2, -1, -1):

        Z = np.matmul(value_functions[t + 1, :], W)
        # C = np.empty((1, len(Y)))
        C = np.empty_like(CUA)

        # if constraint_fn is not None:
        # D = np.empty((1, len(Y)))
        D = np.empty_like(CUA)

        for batch in batch_generator(Y, size=batch_size):

            beta = np.einsum("ij,ik->ijk", CXY[:, batch], CUA)
            C[batch, :] = np.einsum("i,ijk->jk", Z, beta)

            if constraint_fn is not None:
                R = np.matmul(constraint_fn(time=t), W)
                D[batch, :] = np.einsum("i,ijk->jk", R, beta)

        # V = np.zeros((len(Y),))
        # for i in range(len(Y)):
        #     # sol = compute_solution(C[i], D[i])
        #     # idx = np.argmax(sol)
        #     # # print(f"V shape {V.shape}")
        #     # # print(f"V shape {V[i].shape}")
        #     # # print(f"C chape {C.shape}")
        #     # V[i] = C[:, idx]

        #     CA = C[i][D[i] <= 0]
        #     if CA.size == 0:
        #         V[i] = np.min(C[i])
        #     else:
        #         V[i] = np.min(CA)

        #     #     out[t, :] = cost_fn(time=t) + V

        #     # else:

        #     #     Z = np.matmul(value_functions[t + 1, :], W)

        #     #     C = np.zeros_like(CUA)

        #     #     for batch in batch_generator(Y, size=batch_size):

        #     #         beta = np.einsum("ij,ik->ijk", CXY[:, batch], CUA)
        #     #         C[batch, :] = np.einsum("i,ijk->jk", Z, beta)

        #     #     V = np.min(C, axis=1)

        if constraint_fn is not None:

            V = np.zeros((len(Y),))

            for i in range(len(Y)):

                CA = C[i][D[i] <= 0]
                if CA.size == 0:
                    V[i] = np.min(C[i])
                else:
                    V[i] = np.min(CA)
        else:
            V = np.min(C, axis=1)

        out[t, :] = cost_fn(time=t) + V

        pbar.update()

    pbar.close()

    return out


def kernel_control_bwd(
    S: np.ndarray,
    A: np.ndarray,
    time_horizon: int = None,
    cost_fn=None,
    constraint_fn=None,
    heuristic: bool = False,
    regularization_param: float = None,
    kernel_fn=None,
    batch_size: int = None,
    verbose: bool = True,
):
    """Stochastic optimal control policy backward in time.

    Computes the optimal policy using dynamic programming. The solution computes an approximation of the value functions starting at the terminal time and working backwards. Then, when the policy is evaluated, it moves forward in time, optimizing over the value functions and choosing the action which has the highest "value".

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

    """

    alg = KernelControlBwd(
        time_horizon=time_horizon,
        cost_fn=cost_fn,
        constraint_fn=constraint_fn,
        heuristic=heuristic,
        regularization_param=regularization_param,
        kernel_fn=kernel_fn,
        batch_size=batch_size,
        verbose=verbose,
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
        constraint_fn: The constraint function. Should return a real value.
        heuristic: Whether to use the heuristic solution instead of solving the LP.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.
        verbose: Whether the algorithm should print verbose output.

    """

    def __init__(
        self,
        time_horizon: int = None,
        cost_fn=None,
        constraint_fn=None,
        heuristic: bool = False,
        regularization_param: float = None,
        kernel_fn=None,
        batch_size: int = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ):

        self.time_horizon = time_horizon

        self.cost_fn = cost_fn
        self.constraint_fn = constraint_fn

        self.heuristic = heuristic

        self.kernel_fn = kernel_fn
        self.regularization_param = regularization_param

        self.batch_size = batch_size

        self.verbose = verbose

    def _validate_params(self, S=None, A=None):

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

        if S is None:
            raise ValueError("Must supply a sample.")

        if A is None:
            raise ValueError("Must supply a sample.")

        if self.cost_fn is None:
            raise ValueError("Must supply a cost function.")

        if self.constraint_fn is None:
            # raise ValueError("Must supply a constraint function.")
            self._constrained = False
        else:
            self._constrained = True

        if self.batch_size is not None:
            self._compute_backward_recursion_caller = partial(
                _compute_backward_recursion_batch, batch_size=self.batch_size
            )

        else:
            self._compute_backward_recursion_caller = _compute_backward_recursion

    def _validate_data(self, S):

        if S is None:
            raise ValueError("Must supply a sample.")

    def train(self, S: np.ndarray, A: np.ndarray):
        """Train the algorithm.

        Args:
            S: Sample taken iid from the system evolution.
            A: Collection of admissible control actions.

        Returns:
            An instance of the KernelControlFwd algorithm class.

        """

        self._validate_data(S)
        self._validate_data(A)
        self._validate_params(S, A)

        X, U, Y = transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        self.X = X

        A = np.array(A)

        self.A = A

        logger.debug("Computing matrix inverse.")
        self.W = regularized_inverse(
            X,
            U=U,
            kernel_fn=self.kernel_fn,
            regularization_param=self.regularization_param,
        )

        logger.debug("Computing covariance matrices.")
        self.CUA = self.kernel_fn(U, A)
        CXY = self.kernel_fn(X, Y)

        logger.debug("Computing cost function.")
        self.cost_fn = partial(self.cost_fn, state=Y)

        logger.debug("Computing constraint function.")
        if self.constraint_fn is not None:
            self.constraint_fn = partial(self.constraint_fn, state=Y)

        logger.debug("Computing value functions.")
        value_functions = np.empty((self.time_horizon, len(Y)))
        self.value_functions = self._compute_backward_recursion_caller(
            Y,
            self.W,
            CXY,
            self.CUA,
            time_horizon=self.time_horizon,
            cost_fn=self.cost_fn,
            constraint_fn=self.constraint_fn,
            value_functions=value_functions,
            out=value_functions,
            verbose=True,
        )

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        state = np.atleast_2d(np.asarray(state, dtype=np.float32))

        T = np.array(state)

        # Compute covariance matrix.
        CXT = self.kernel_fn(self.X, T)

        # Compute beta.
        betaXT = self.W @ (CXT * self.CUA)

        # Compute cost vector.
        C = np.matmul(np.array(self.value_functions[time], dtype=np.float32), betaXT)

        # Compute constraint vector.
        D = None
        if self.constraint_fn is not None:
            D = np.matmul(
                np.array(self.constraint_fn(time=time), dtype=np.float32), betaXT
            )

        # Compute the solution to the LP.
        logger.debug("Computing solution to the LP.")
        sol = compute_solution(C, D, heuristic=self.heuristic)
        idx = np.argmax(sol)
        return np.array(self.A[idx], dtype=np.float32)
