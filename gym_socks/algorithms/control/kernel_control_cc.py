"""Chance-constrained stochastic optimal control.

"""

from functools import partial

import gym_socks

from gym.utils import seeding

from gym_socks.envs.policy import BasePolicy

from gym_socks.algorithms.control.control_common import compute_solution

from gym_socks.kernel.metrics import rbf_kernel, regularized_inverse

from scipy.optimize import linprog

from gym_socks.utils import normalize
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

import numpy as np

from tqdm.auto import tqdm


def _compute_chance_constrained_solution(
    C: np.ndarray,
    D: np.ndarray,
    heuristic=False,
    delta=None,
) -> np.ndarray:
    """Compute the chance-constrained solution to the LP.

    Args:
        C: Array holding values of the cost function evaluated at sample points.
        D: Array holding values of the constraint function evaluated at sample points.
        heuristic: Whether to compute the heuristic solution.

    Returns:
        gamma: Probability vector.

    """
    # C = (Cx @ K + Cu)
    # D = (Dx @ K + Du)

    print(f"min C: {C.min()}, max C: {C.max()}")
    print(f"min D: {D.min()}, max D: {D.max()}")

    if heuristic is False:

        if len(D.shape) == 1:
            D = D.reshape(-1, 1)

        obj = C.T
        A_ub = -D.T
        b_ub = -1 + delta
        A_eq = np.ones((1, len(C)))
        b_eq = 1
        # Bounds are automatically set so that decision variables are nonnegative.
        # bounds = [(0, None)] * len(C)

        gym_socks.logger.debug("Computing solution via scipy LP solver.")
        sol = linprog(
            obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
        )

        gym_socks.logger.debug(f"Solver completed with status code: {sol.status}")
        # 0 : Optimization terminated successfully.
        # 1 : Iteration limit reached.
        # 2 : Problem appears to be infeasible.
        # 3 : Problem appears to be unbounded.
        # 4 : Numerical difficulties encountered.

        if sol.success is True:
            print(f"cost: {np.dot(C, sol.x)}")
            return sol.x
        else:
            gym_socks.logger.debug("No solution found via scipy.optimize.linprog.")
            gym_socks.logger.debug("Returning heuristic solution.")

    heuristic_sol = np.zeros((len(C),))
    satisfies_constraints = np.where(D <= 0)
    if len(satisfies_constraints[0]) == 0:
        gym_socks.logger.warn("No feasible solution found!")
        gym_socks.logger.debug("Returning minimal unconstrained solution.")
        idx = C.argmin()
    else:
        idx = satisfies_constraints[0][C[satisfies_constraints].argmin()]
    heuristic_sol[idx] = 1

    return heuristic_sol


def kernel_control_cc(
    S: np.ndarray,
    A: np.ndarray,
    cost_fn_x=None,
    cost_fn_u=None,
    constraint_fn_x=None,
    constraint_fn_u=None,
    delta: float = None,
    heuristic: bool = False,
    regularization_param: float = None,
    kernel_fn_x=None,
    kernel_fn_u=None,
    verbose: bool = True,
    seed=None,
):
    """Stochastic optimal control policy (chance-constrained).

    Args:
        S: Sample taken iid from the system evolution.
        A: Collection of admissible control actions.
        cost_fn: The cost function. Should return a real value.
        constraint_fn: The constraint function. Should return a real value.
        delta: Tolerable probability of failure.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.
        verbose: Whether the algorithm should print verbose output.
        seed: The random seed.

    Returns:
        The policy.
    """

    alg = KernelControlCC(
        cost_fn_x=cost_fn_x,
        cost_fn_u=cost_fn_u,
        constraint_fn_x=constraint_fn_x,
        constraint_fn_u=constraint_fn_u,
        delta=delta,
        heuristic=heuristic,
        regularization_param=regularization_param,
        kernel_fn_x=kernel_fn_x,
        kernel_fn_u=kernel_fn_u,
        verbose=verbose,
        seed=seed,
    )
    alg.train(S, A)

    return alg


class KernelControlCC(BasePolicy):
    """Stochastic optimal control policy (chance-constrained).

    Args:
        cost_fn: The cost function. Should return a real value.
        constraint_fn: The constraint function. Should return a real value.
        delta: Tolerable probability of failure.
        heuristic: Whether to use the heuristic solution instead of solving the LP.
        regularization_param: Regularization prameter for the regularized least-squares
            problem used to construct the approximation.
        kernel_fn: The kernel function used by the algorithm.
        verbose: Whether the algorithm should print verbose output.
        seed: The random seed.

    """

    def __init__(
        self,
        cost_fn_x=None,
        cost_fn_u=None,
        constraint_fn_x=None,
        constraint_fn_u=None,
        delta: float = None,
        heuristic: bool = False,
        regularization_param: float = None,
        kernel_fn_x=None,
        kernel_fn_u=None,
        verbose: bool = True,
        seed=None,
        *args,
        **kwargs,
    ):

        self.cost_fn_x = cost_fn_x
        self.cost_fn_u = cost_fn_u
        self.constraint_fn_x = constraint_fn_x
        self.constraint_fn_u = constraint_fn_u

        self.delta = delta

        self.heuristic = heuristic

        self.kernel_fn_x = kernel_fn_x
        self.kernel_fn_u = kernel_fn_u
        self.regularization_param = regularization_param

        self.verbose = verbose

        self.probability_vector = None

        self._seed = None
        self._np_random = None
        if seed is not None:
            self._seed = self.seed(seed)

    @property
    def np_random(self):
        """Random number generator."""
        if self._np_random is None:
            self._seed = self.seed()

        return self._np_random

    def seed(self, seed=None):
        """Sets the seed of the random number generator.

        Args:
            seed: Integer value representing the random seed.

        Returns:
            The seed of the RNG.

        """
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _validate_params(self, S=None, A=None):

        if S is None:
            raise ValueError("Must supply a sample.")

        if A is None:
            raise ValueError("Must supply a sample.")

        if self.kernel_fn_x is None:
            self.kernel_fn_x = partial(rbf_kernel, sigma=0.1)

        if self.kernel_fn_u is None:
            self.kernel_fn_u = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

        if self.cost_fn_x is None:
            raise ValueError("Must supply a cost function.")

        if self.cost_fn_u is None:
            raise ValueError("Must supply a cost function.")

        if self.constraint_fn_x is None:
            raise ValueError("Must supply a constraint function.")

        if self.constraint_fn_u is None:
            raise ValueError("Must supply a constraint function.")

        if self.delta is None:
            raise ValueError("Must supply a delta.")

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
        # self.W = regularized_inverse(
        #     X,
        #     U=U,
        #     kernel_fn=self.kernel_fn,
        #     regularization_param=self.regularization_param,
        # )
        K = self.kernel_fn_x(X) * self.kernel_fn_u(U)
        self.W = np.linalg.inv(
            K + self.regularization_param * len(X) * np.identity(len(X))
        )

        gym_socks.logger.debug("Computing covariance matrix.")
        self.CUA = self.kernel_fn_u(U, A)

        gym_socks.logger.debug("Computing cost function.")
        self.cost_fn_x = partial(self.cost_fn_x, state=Y)
        self.cost_fn_u = partial(self.cost_fn_u, action=A)

        gym_socks.logger.debug("Computing constraint function.")
        self.constraint_fn_x = partial(self.constraint_fn_x, state=Y)
        self.constraint_fn_u = partial(self.constraint_fn_u, action=A)

        # self._nkernel_fn = partial(rbf_kernel, sigma=0.1)
        # self._nW = regularized_inverse(
        #     U,
        #     kernel_fn=self._nkernel_fn,
        #     regularization_param=1,
        # )

        # self._nCUA = self._nkernel_fn(U, A)

        return self

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(state)

        # Compute covariance matrix.
        CXT = self.kernel_fn_x(self.X, T)
        # Compute beta.
        betaXT = self.W @ (CXT * self.CUA)

        CU = np.array(self.cost_fn_u(time=time), dtype=np.float32)
        print(f"min CU: {CU.min()}, max CU: {CU.max()}")

        print(f"min self.W: {self.W.min()}, max self.W: {self.W.max()}")
        print(f"min CXT: {CXT.min()}, max CXT: {CXT.max()}")
        print(f"min self.CUA: {self.CUA.min()}, max self.CUA: {self.CUA.max()}")

        # Compute cost vector.
        C = np.matmul(
            np.array(self.cost_fn_x(time=time), dtype=np.float32), betaXT
        ) + np.array(self.cost_fn_u(time=time), dtype=np.float32)

        # Compute constraint vector.
        DY = np.array(self.constraint_fn_x(time=time), dtype=np.float32)
        print(f"nnz DY: {np.count_nonzero(DY)}")
        print(f"min DY: {DY.min()}, max DY: {DY.max()}")
        print(f"min betaXT: {betaXT.min()}, max betaXT: {betaXT.max()}")

        # print(f"min nW: {self._nW.min()}, max nW: {self._nW.max()}")
        # print(f"min nCUA: {self._nCUA.min()}, max nCUA: {self._nCUA.max()}")
        # beta = normalize(self._nW @ self._nCUA)
        # print(f"min beta: {beta.min()}, max beta: {beta.max()}")

        D = np.matmul(
            np.array(self.constraint_fn_x(time=time), dtype=np.float32), betaXT
        ) + np.array(self.constraint_fn_u(time=time), dtype=np.float32)

        # D = normalize(D + D.min())
        D = np.clip(D, 0, 1)

        print(f"D shape: {np.shape(D)}")

        # Compute the solution to the LP.
        gym_socks.logger.debug("Computing solution to the LP.")
        solution = _compute_chance_constrained_solution(C, D, delta=self.delta)
        solution = normalize(solution)  # Normalize the vector.
        self.probability_vector = solution
        idx = self.np_random.choice(len(solution), size=None, p=solution)
        return np.array(self.A[idx], dtype=np.float32)
