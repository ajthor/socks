r"""Common functions for kernel control algorithms.

This file contains common functions used by the kernel optimal control algorithms, and
implements an LP solver to compute the probability vector :math:`\gamma`. This
functionality is accessed via the :py:func:``compute_solution`` function, which serves
as a single entrypoint, and the unconstrained version is chosen if the constraint matrix
``D`` is None.

"""

import numpy as np

from scipy.optimize import linprog

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

logger = logging.getLogger(__name__)


def _compute_solution(C: np.ndarray, D: np.ndarray = None):
    """Compute the solution to the LP.

    Computes a solution to the linear program, choosing either to delegate to the unconstrained or constrained solver depending on whether D is `None`.

    Args:
        C: Array holding values of the cost function evaluated at sample points.
        D: Array holding values of the constraint function evaluated at sample points.

    Returns:
        Probability vector.

    """

    obj = C.T
    A_ub = None
    b_ub = None
    A_eq = np.ones((1, len(C)))
    b_eq = 1

    if D is not None:
        if D.ndim == 1:
            D = D.reshape(-1, 1)

        # Check for valid constraint solutions.
        if not np.any(D <= 0):
            raise ValueError("No points satisfy constraints.")

        A_ub = D.T
        b_ub = 0

    logger.debug("Computing solution via scipy.optimize.linprog...")
    sol = linprog(
        obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
    )

    logger.debug(f"Solver completed with status code: {sol.status}")
    # 0 : Optimization terminated successfully.
    # 1 : Iteration limit reached.
    # 2 : Problem appears to be infeasible.
    # 3 : Problem appears to be unbounded.
    # 4 : Numerical difficulties encountered.

    if sol.success is True:
        return sol.x
    else:
        logger.debug("No solution found via scipy.optimize.linprog.")
