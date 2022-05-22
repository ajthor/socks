"""Kernel belief propagation algorithm.

Algorithm for nonparametric inference in graphical models.

Example:

    >>> from gym_socks.algorithms.estimation import KernelBeliefPropagation
    >>> estimator = KernelBeliefPropagation()

"""

from functools import partial

import numpy as np

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse

from gym_socks.sampling.transform import transpose_sample

from gym_socks.utils.validation import check_array
from gym_socks.utils.validation import check_matrix

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError as e:
    nx = None
    logger.warning(ImportWarning(e.msg))


def kernel_belief_propagation(
    A: np.ndarray,
    S: list,
    kernel_fn=None,
    regularization_param: float = None,
    verbose: bool = False,
) -> np.ndarray:
    """Kernel bayes filter.

    Args:
        A: Adjacency matrix of the graph.
        S: List of samples taken from each node. Each element of the list should be a
            list of tuples.
        kernel_fn: Kernel function used by the estimator.
        regularization_param: Regularization parameter used in the solution to the
            regularized least-squares problem.
        verbose: Boolean flag to indicate verbose output.

    Returns:
        The fitted model.

    """

    alg = KernelBeliefPropagation(
        kernel_fn=kernel_fn,
        regularization_param=regularization_param,
        verbose=verbose,
    )
    alg.fit(A, S)
    return alg


class KernelBeliefPropagation(object):
    """Kernel belief propagation.

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

    def fit(self, A: np.ndarray, S: np.ndarray):
        """Run the algorithm.

        Args:
            S : Sample of (x, u, y, z) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.
            initial_condition: Initial state of the system.

        Returns:
            Instance of KernelBeliefPropagation class.

        """

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

        A = check_matrix(A, ensure_square=True)
        self.num_nodes = len(A)

        X, U, Y, Z = transpose_sample(S)
        X = check_array(X)
        U = check_array(U)
        Y = check_array(Y)
        Z = check_array(Z)

        logger.debug("Computing covariance matrices.")

        return self

    def predict(self, action: np.ndarray, observation: np.ndarray, num_iter: int):
        """Update the belief state.

        Args:
            action : The action of the system.
            observation : The observation received from the system.
            num_iter: Number of iterations.

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
