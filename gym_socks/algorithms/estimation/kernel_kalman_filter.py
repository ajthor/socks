"""Kernel-based linear system identification.

The algorithm uses a concatenated state space representation to compute the state and
input matrices given a sample of system observations. Uses the matrix inversion lemma
and a linear kernel to compute the linear relationship between observations.

"""

from functools import partial

import gym_socks
from gym_socks.algorithms.algorithm import AlgorithmInterface
from gym_socks.envs.sample import transpose_sample

import numpy as np


def kernel_linear_id(
    S: np.ndarray,
    regularization_param: float = None,
    verbose: bool = False,
) -> np.ndarray:
    """Stochastic linear system identification using kernel distribution embeddings.

    Computes an approximation of the state and input matrices, as well as an estimate of
    the stochastic disturbance mean given a sample of system observations.

    Args:
        S: Sample of (x, u, y) tuples taken iid from the system evolution. The sample
            should be in the form of a list of tuples.
        regularization_param: Regularization parameter used in the solution to the
            regularized least-squares problem.
        verbose: Boolean flag to indicate verbose output.

    Returns:
        The fitted model computed using the linear identification algorithm.

    """

    alg = KernelLinearId(
        regularization_param=regularization_param,
        verbose=verbose,
    )
    alg.fit(S)
    return alg


class KernelLinearId(AlgorithmInterface):
    """Stochastic linear system identification using kernel distribution embeddings.

    Computes an approximation of the state and input matrices, as well as an estimate of
    the stochastic disturbance mean given a sample of system observations.

    Args:
        regularization_param : Regularization parameter used in the solution to the
            regularized least-squares problem.
        verbose : Boolean flag to indicate verbose output.

    """

    def __init__(
        self,
        regularization_param: float = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.regularization_param = regularization_param

        self.verbose = verbose

        self._state_matrix = None
        self._input_matrix = None
        self._disturbance_mean = None

    @property
    def state_matrix(self):
        """The estimated state matrix."""
        return self._state_matrix

    @property
    def input_matrix(self):
        """The estimated input matrix."""
        return self._input_matrix

    @property
    def disturbance_mean(self):
        """The estimated stochastic disturbance mean."""
        return self._disturbance_mean

    def _validate_params(self, S):

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
            self : Instance of KernelLinearId class.

        """

        self._validate_data(S)
        self._validate_params(S)

        len_sample = len(S)

        X, U, Y = transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        x_dim = X.shape[1]  # dimensionality of the input data
        u_dim = U.shape[1]  # dimensionality of the input data

        # Create concatenated state representation.
        Z = np.concatenate((X, U, np.ones((len_sample, 1))), axis=1)

        z_dim = Z.shape[1]

        if len_sample < z_dim:
            msg = f"The sample size is less than the dim of H: {len_sample} < {z_dim}."
            gym_socks.logger.warn(msg)

        gym_socks.logger.debug("Computing covariance matrices.")
        CZZ = (Z.T @ Z) + self.regularization_param * len_sample * np.identity(z_dim)
        CYZ = Y.T @ Z

        gym_socks.logger.debug("Computing estimates.")
        H = np.linalg.solve(
            CZZ.T,
            CYZ.T,
        ).T

        self._state_matrix = H[:x_dim, :x_dim]
        self._input_matrix = H[:x_dim, x_dim : x_dim + u_dim]
        self._disturbance_mean = H[:x_dim, -1:]

        return self

    def predict(self, T: np.ndarray, U: np.ndarray = None) -> np.ndarray:
        """Predict.

        Args:
            T : State vectors. Should be in the form of a 2D-array, where each row
                indicates a state.
            U : Input vectors. Should be in the form of a 2D-array, where each row
                indicates an input.

        Returns:
            The predicted resulting states after propagating through the estimated
            system dynamics.

        """

        self._validate_data(T)
        T = np.array(T)

        result = self._state_matrix @ T.T

        if U is not None:
            self._validate_data(U)
            U = np.array(U)

            result += self._input_matrix @ U.T

        result += np.squeeze(self._disturbance_mean.T)

        return result
