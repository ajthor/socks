"""Kernel-based stochastic reachability.

References:
    .. [1] `Model-Free Stochastic Reachability
            Using Kernel Distribution Embeddings, 2019
           Adam J. Thorpe, Meeko M. K. Oishi
           IEEE Control Systems Letters,
           <https://arxiv.org/abs/1908.00697>`_

"""

from functools import partial

import gym_socks
from gym_socks.algorithms.algorithm import AlgorithmInterface
from gym_socks.algorithms.reach.reach_common import _tht_step, _fht_step
from gym_socks.envs.sample import transpose_sample
from gym_socks.kernel.metrics import rbf_kernel, regularized_inverse

from gym_socks.utils import normalize, indicator_fn, generate_batches
from gym_socks.utils.logging import ms_tqdm, _progress_fmt
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np


def _compute_backward_recursion(
    Y,
    W,
    CXY,
    CUA,
    time_horizon=None,
    constraint_tube=None,
    target_tube=None,
    value_functions=None,
    out=None,
    step_fn=None,
    verbose=False,
):
    # Compute beta.
    gym_socks.logger.debug("Computing beta.")
    betaXY = normalize(
        np.einsum("ii,ij,ik->ijk", W, CXY, CUA, optimize=["einsum_path", (0, 1, 2)])
    )

    pbar = ms_tqdm(
        total=time_horizon,
        bar_format=_progress_fmt,
        desc="Computing backward recursion.",
        disable=False if verbose is True else True,
    )

    # Initialize the backward recursion.
    if out is None:
        out = np.zeros((time_horizon, len(Y)))
    out[time_horizon - 1, :] = indicator_fn(Y, target_tube[time_horizon - 1])
    pbar.update()

    # Backward recursion.
    for t in range(time_horizon - 2, -1, -1):

        wX = np.einsum(
            "i,ijk->jk",
            value_functions[t + 1, :],
            betaXY,
            optimize=["einsum_path", (0, 1)],
        )

        VX = np.max(wX, axis=1)
        out[t, :] = step_fn(Y, VX, constraint_tube[t], target_tube[t])

        pbar.update()

    pbar.close()

    return out


def _compute_backward_recursion_batch(
    Y,
    W,
    CXY,
    CUA,
    time_horizon=None,
    constraint_tube=None,
    target_tube=None,
    value_functions=None,
    out=None,
    step_fn=None,
    batch_size=None,
    verbose=False,
):
    pbar = ms_tqdm(
        total=time_horizon,
        bar_format=_progress_fmt,
        desc="Computing backward recursion.",
        disable=False if verbose is True else True,
    )

    # Initialize the backward recursion.
    if out is None:
        out = np.zeros((time_horizon, len(Y)))
    out[time_horizon - 1, :] = indicator_fn(Y, target_tube[time_horizon - 1])
    pbar.update()

    # Backward recursion.
    for t in range(time_horizon - 2, -1, -1):

        for batch in generate_batches(len(Y), batch_size=batch_size):

            # Compute beta.
            betaXY = normalize(
                np.einsum(
                    "ii,ij,ik->ijk",
                    W,
                    CXY[:, batch],
                    CUA,
                    optimize=["einsum_path", (0, 1, 2)],
                )
            )

            wX = np.einsum(
                "i,ijk->jk",
                value_functions[t + 1, :],
                betaXY,
                optimize=["einsum_path", (0, 1)],
            )

            VX = np.max(wX, axis=1)
            out[t, batch] = step_fn(Y[batch], VX, constraint_tube[t], target_tube[t])

        pbar.update()

    pbar.close()

    return out


def kernel_sr_max(
    S: np.ndarray,
    A: np.ndarray,
    T: np.ndarray,
    time_horizon: int = None,
    constraint_tube: list = None,
    target_tube: list = None,
    problem: str = "THT",
    regularization_param: float = None,
    kernel_fn=None,
    batch_size: int = None,
    verbose: bool = False,
):
    """Stochastic reachability using kernel distribution embeddings.

    Computes an approximation of the maximal safety probabilities of the stochastic
    reachability problem using kernel methods.

    Args:
        S: Sample of (x, u, y) tuples taken iid from the system evolution. The
            sample should be in the form of a list of tuples.
        A: Collection of admissible control action to choose from. Should be in the
            form of a 2D-array, where each row indicates a point.
        T: Evaluation points to evaluate the safety probabilities at. Should be in the
            form of a 2D-array, where each row indicates a point.
        time_horizon: Number of time steps to compute the approximation.
        constraint_tube: List of spaces or constraint functions. Must be the same
            length as `time_horizon`.
        target_tube: List of spaces or target functions. Must be the same length as
            `time_horizon`.
        problem: One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
            problem and `"FHT"` specifies the first-hitting time problem.
        kernel_fn: Kernel function used by the approximation.
        regularization_param: Regularization parameter used in the solution to the
            regularized least-squares problem.
        batch_size: The batch size for more memory-efficient computations. Omit this
            parameter or set to `None` to compute without batch processing.
        verbose: Boolean flag to indicate verbose output.

    Returns:
        An array of safety probabilities of shape {len(T), time_horizon}, where each row
        indicates the safety probabilities of the evaluation points at a different time
        step.

    """

    alg = KernelMaximalSR(
        time_horizon=time_horizon,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem=problem,
        regularization_param=regularization_param,
        kernel_fn=kernel_fn,
        batch_size=batch_size,
        verbose=verbose,
    )
    alg.fit(S, A)
    return alg.predict(T)


class KernelMaximalSR(AlgorithmInterface):
    """Stochastic reachability using kernel distribution embeddings.

    Computes an approximation of the maximal safety probabilities of the stochastic
    reachability problem using kernel methods.

    Args:
        time_horizon: Number of time steps to compute the approximation.
        constraint_tube: List of spaces or constraint functions. Must be the same
            length as `time_horizon`.
        target_tube: List of spaces or target functions. Must be the same length as
            `time_horizon`.
        problem: One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
            problem and `"FHT"` specifies the first-hitting time problem.
        kernel_fn: Kernel function used by the approximation.
        regularization_param: Regularization parameter used in the solution to the
            regularized least-squares problem.
        batch_size: The batch size for more memory-efficient computations. Omit this
            parameter or set to `None` to compute without batch processing.
        verbose: Boolean flag to indicate verbose output.

    """

    def __init__(
        self,
        time_horizon: int = None,
        constraint_tube: list = None,
        target_tube: list = None,
        problem: str = "THT",
        regularization_param: float = None,
        kernel_fn=None,
        batch_size: int = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.time_horizon = time_horizon
        self.constraint_tube = constraint_tube
        self.target_tube = target_tube

        self.problem = problem

        self.kernel_fn = kernel_fn
        self.regularization_param = regularization_param

        self.batch_size = batch_size

        self.verbose = verbose

    def _validate_params(self, S, A):

        if self.kernel_fn is None:
            self.kernel_fn = partial(rbf_kernel, sigma=0.1)

        if self.regularization_param is None:
            self.regularization_param = 1

        if self.time_horizon is None:
            raise ValueError("Must supply a time horizon.")

        assert self.time_horizon >= 0 and isinstance(
            self.time_horizon, (int, np.integer)
        )

        if self.constraint_tube is None:
            raise ValueError("Must supply constraint tube.")

        if self.target_tube is None:
            raise ValueError("Must supply target tube.")

        if self.problem not in ("FHT", "THT"):
            raise ValueError(f"problem is not in {'THT', 'FHT'}")

        if self.problem == "THT":
            self.step_fn = _tht_step
        elif self.problem == "FHT":
            self.step_fn = _fht_step

        if self.batch_size is not None:
            self._compute_backward_recursion_caller = partial(
                _compute_backward_recursion_batch, batch_size=self.batch_size
            )

        else:
            self._compute_backward_recursion_caller = _compute_backward_recursion

    def _validate_data(self, S=None):

        if S is None:
            raise ValueError("Must supply a sample.")

    def fit(self, S: np.ndarray, A: np.ndarray):
        """Run the algorithm.

        Args:
            S: Sample of (x, u, y) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.
            A: Collection of admissible control action to choose from. Should be in
                the form of a 2D-array, where each row indicates a point.

        Returns:
            self : Instance of KernelMaximalSR class.

        """

        self._validate_params(S, A)
        self._validate_data(S)
        self._validate_data(A)

        X, U, Y = transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        self.X = X

        A = np.array(A)

        gym_socks.logger.debug("Computing matrix inverse.")
        self.W = regularized_inverse(
            X,
            U=U,
            kernel_fn=self.kernel_fn,
            regularization_param=self.regularization_param,
        )

        gym_socks.logger.debug("Computing covariance matrices.")
        CXY = self.kernel_fn(X, Y)
        self.CUA = self.kernel_fn(U, A)

        gym_socks.logger.debug("Computing value functions.")
        value_functions = np.empty((self.time_horizon, len(Y)))
        self.value_functions = self._compute_backward_recursion_caller(
            Y,
            self.W,
            CXY,
            self.CUA,
            time_horizon=self.time_horizon,
            constraint_tube=self.constraint_tube,
            target_tube=self.target_tube,
            value_functions=value_functions,
            out=value_functions,
            step_fn=self.step_fn,
            verbose=True,
        )

    def predict(self, T):
        """Predict.

        Args:
            T: Evaluation points to evaluate the safety probabilities at. Should be in
                the form of a 2D-array, where each row indicates a point.

        Returns:
            An array of safety probabilities of shape {len(T), time_horizon}, where each
            row indicates the safety probabilities of the evaluation points at a
            different time step.

        """

        self._validate_data(T)
        T = np.array(T)

        gym_socks.logger.debug("Computing covariance matrix.")
        CXT = self.kernel_fn(self.X, T)

        gym_socks.logger.debug("Computing safety probabilities.")
        safety_probabilities = self._compute_backward_recursion_caller(
            T,
            self.W,
            CXT,
            self.CUA,
            time_horizon=self.time_horizon,
            constraint_tube=self.constraint_tube,
            target_tube=self.target_tube,
            value_functions=self.value_functions,
            step_fn=self.step_fn,
            verbose=True,
        )

        return safety_probabilities
