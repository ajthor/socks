"""Common functions for kernel control algorithms.

This file contains common functions used by the kernel optimal control algorithms, and
implements an LP solver to compute the probability vector gamma. This functionality is
accessed via the `compute_solution` function, which serves as a single entrypoint, and
the unconstrained version is chosen if the constraint matrix D is None.

References:
    .. [1] `Stochastic Optimal Control via
            Hilbert Space Embeddings of Distributions, 2021
            Adam J. Thorpe, Meeko M. K. Oishi
            IEEE Conference on Decision and Control,
            <https://arxiv.org/abs/2103.12759>`_

"""
import gym_socks
import numpy as np

from scipy.optimize import linprog


def _check_constraint_matrix(D):
    return np.any(np.negative(D))


def compute_solution(
    C: np.ndarray, D: np.ndarray = None, heuristic: bool = False
) -> np.ndarray:
    """Compute the solution to the LP.

    Computes a solution to the linear program, choosing either to delegate to the unconstrained or constrained solver depending on whether D is `None`.

    Args:
        C: Array holding values of the cost function evaluated at sample points.
        D: Array holding values of the constraint function evaluated at sample points.
        heuristic: Whether to compute the heuristic solution.

    Returns:
        gamma: Probability vector.
    """
    if D is None:
        return _compute_unconstrained_solution(C, heuristic)
    else:
        return _compute_constrained_solution(C, D, heuristic)


def _compute_unconstrained_solution(C: np.ndarray, heuristic=False) -> np.ndarray:
    """Compute the unconstrained solution to the LP.

    NOTE: A heuristic solution is available, but due to the speed of the LP solver for
    feasible problems, there is usually no need to invoke the heuristic solution, and
    the computation times are roughly equal. The main reason to use the heuristic
    solution is when scipy is unavailable.

    Note that in the unconstrained case, a closed-form solution is available by the
    Lagrangian dual. Thus, using the heuristic solution can be much faster in the
    unconstrained case.

    Args:
        C: Array holding values of the cost function evaluated at sample points.
        heuristic: Whether to compute the heuristic solution.

    Returns:
        gamma: Probability vector.

    """
    # C = (Cx @ K + Cu)

    if heuristic is False:

        obj = C.T
        A_eq = np.ones((1, len(C)))
        b_eq = 1
        # Bounds are automatically set so that decision variables are nonnegative.
        # bounds = [(0, None)] * len(C)

        gym_socks.logger.debug("Computing solution via scipy LP solver.")
        sol = linprog(
            obj,
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
            return sol.x
        else:
            gym_socks.logger.warn("No solution found via scipy.optimize.linprog.")
            gym_socks.logger.warn("Returning heuristic solution.")

    gym_socks.logger.debug("Computing heuristic solution.")
    heuristic_sol = np.zeros((len(C),))
    idx = np.argmin(C)
    heuristic_sol[idx] = 1
    return heuristic_sol


def _compute_constrained_solution(
    C: np.ndarray, D: np.ndarray, heuristic=False
) -> np.ndarray:
    """Compute the constrained solution to the LP.

    NOTE: A heuristic solution is available, but due to the speed of the LP solver for
    feasible problems, there is usually no need to invoke the heuristic solution, and
    the computation times are roughly equal. The main reason to use the heuristic
    solution is when scipy is unavailable.

    For the constrained problem, there is no closed-form solution via the Lagrangian
    dual. Thus, the heuristic solution computes the probability vector by masking the
    entries of D which are positive (solutions which do not satisfy the constraints),
    and then finding the minimum value of the masked cost vector.

    Args:
        C: Array holding values of the cost function evaluated at sample points.
        D: Array holding values of the constraint function evaluated at sample points.
        heuristic: Whether to compute the heuristic solution.

    Returns:
        gamma: Probability vector.

    """
    # C = (Cx @ K + Cu)
    # D = (Dx @ K + Du)
    _check_constraint_matrix(D)

    if heuristic is False:

        if len(D.shape) == 1:
            D = D.reshape(-1, 1)

        obj = C.T
        A_ub = D.T
        b_ub = 0
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
            return sol.x
        else:
            gym_socks.logger.warn(
                "No solution found via scipy.optimize.linprog."
                "Returning heuristic solution."
            )

    heuristic_sol = np.zeros((len(C),))
    satisfies_constraints = np.where(D <= 0)
    idx = satisfies_constraints[0][C[satisfies_constraints].argmin()]
    heuristic_sol[idx] = 1

    return heuristic_sol
