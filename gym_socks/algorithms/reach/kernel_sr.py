"""Kernel-based stochastic reachability.

Stochastic reachability seeks to compute the likelihood that a system will satisfy
pre-specified safety constraints.

"""

from functools import partial

import numpy as np

from gym_socks.algorithms.base import RegressorMixin
from gym_socks.algorithms.reach.common import _fht_step
from gym_socks.algorithms.reach.common import _tht_step

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse

from gym_socks.sampling.transform import transpose_sample

from gym_socks.utils import indicator_fn
from gym_socks.utils import normalize
from gym_socks.utils.batch import batch_generator

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

logger = logging.getLogger(__name__)


def _compute_backward_recursion(
    Y,
    W,
    CXY,
    time_horizon=None,
    constraint_tube=None,
    target_tube=None,
    value_functions=None,
    out=None,
    step_fn=None,
    verbose=False,
):
    # Compute beta.
    logger.debug("Computing beta.")
    betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))

    pbar = ms_tqdm(
        total=time_horizon,
        bar_format=_progress_fmt,
        desc="Computing backward recursion.",
        disable=False if verbose is True else True,
    )

    # Initialize the backward recursion.
    if out is None:
        out = np.empty((time_horizon, len(Y)))
    out[time_horizon - 1, :] = indicator_fn(Y, target_tube[time_horizon - 1])
    pbar.update()

    # Backward recursion.
    for t in range(time_horizon - 2, -1, -1):
        V = np.einsum("i,ij->j", value_functions[t + 1, :], betaXY)
        out[t, :] = step_fn(Y, V, constraint_tube[t], target_tube[t])

        pbar.update()

    pbar.close()

    return out


def _compute_backward_recursion_batch(
    Y,
    W,
    CXY,
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
        out = np.empty((time_horizon, len(Y)))
    out[time_horizon - 1, :] = indicator_fn(Y, target_tube[time_horizon - 1])
    pbar.update()

    # Backward recursion.
    for t in range(time_horizon - 2, -1, -1):

        for batch in batch_generator(Y, size=batch_size):

            # Compute beta.
            betaXY = normalize(np.einsum("ii,ij->ij", W, CXY[:, batch]))

            V = np.einsum("i,ij->j", value_functions[t + 1, :], betaXY)
            out[t, batch] = step_fn(Y[batch], V, constraint_tube[t], target_tube[t])

        pbar.update()

    pbar.close()

    return out


def kernel_sr(
    S: np.ndarray,
    T: np.ndarray,
    time_horizon: int = None,
    constraint_tube: list = None,
    target_tube: list = None,
    problem: str = "THT",
    regularization_param: float = None,
    kernel_fn=None,
    batch_size: int = None,
    verbose: bool = False,
) -> np.ndarray:
    """Stochastic reachability using kernel distribution embeddings.

    Computes an approximation of the safety probabilities of the stochastic reachability
    problem using kernel methods.

    Args:
        S: Sample of (x, u, y) tuples taken iid from the system evolution. The sample
            should be in the form of a list of tuples.
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

    alg = KernelSR(
        time_horizon=time_horizon,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem=problem,
        regularization_param=regularization_param,
        kernel_fn=kernel_fn,
        batch_size=batch_size,
        verbose=verbose,
    )
    alg.fit(S)
    return alg.predict(T)


class KernelSR(RegressorMixin):
    """Stochastic reachability using kernel distribution embeddings.

    Computes an approximation of the safety probabilities of the stochastic reachability
    problem using kernel methods.

    Args:
        time_horizon : Number of time steps to compute the approximation.
        constraint_tube : List of spaces or constraint functions. Must be the same
            length as `time_horizon`.
        target_tube : List of spaces or target functions. Must be the same length as
            `time_horizon`.
        problem : One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
            problem and `"FHT"` specifies the first-hitting time problem.
        kernel_fn : Kernel function used by the approximation.
        regularization_param : Regularization parameter used in the solution to the
            regularized least-squares problem.
        batch_size : The batch size for more memory-efficient computations. Omit this
            parameter or set to `None` to compute without batch processing.
        verbose : Boolean flag to indicate verbose output.

    """

    def __init__(
        self,
        time_horizon: int = None,
        constraint_tube: list = None,
        target_tube: list = None,
        problem: str = "THT",
        kernel_fn=None,
        regularization_param: float = None,
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

    def _validate_params(self, S):

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

    def _validate_data(self, S):

        if S is None:
            raise ValueError("Must supply a sample.")

    def fit(self, S: np.ndarray):
        """Run the algorithm.

        Args:
            S : Sample of (x, u, y) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.

        Returns:
            self : Instance of KernelSR class.

        """

        self._validate_data(S)
        self._validate_params(S)

        X, U, Y = transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        self.X = X

        logger.debug("Computing matrix inverse.")
        self.W = regularized_inverse(
            self.kernel_fn(X),
            self.regularization_param,
            copy=False,
        )

        logger.debug("Computing covariance matrix.")
        CXY = self.kernel_fn(X, Y)

        logger.debug("Computing value functions.")
        value_functions = np.empty((self.time_horizon, len(Y)))
        self.value_functions = self._compute_backward_recursion_caller(
            Y,
            self.W,
            CXY,
            time_horizon=self.time_horizon,
            constraint_tube=self.constraint_tube,
            target_tube=self.target_tube,
            value_functions=value_functions,
            out=value_functions,
            step_fn=self.step_fn,
            verbose=self.verbose,
        )

    def predict(self, T: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            T : Evaluation points to evaluate the safety probabilities at. Should be in
                the form of a 2D-array, where each row indicates a point.

        Returns:
            An array of safety probabilities of shape {len(T), time_horizon}, where each
            row indicates the safety probabilities of the evaluation points at a
            different time step.

        """

        self._validate_data(T)
        T = np.array(T)

        logger.debug("Computing covariance matrix.")
        CXT = self.kernel_fn(self.X, T)

        logger.debug("Computing safety probabilities.")
        safety_probabilities = self._compute_backward_recursion_caller(
            T,
            self.W,
            CXT,
            time_horizon=self.time_horizon,
            constraint_tube=self.constraint_tube,
            target_tube=self.target_tube,
            value_functions=self.value_functions,
            step_fn=self.step_fn,
            verbose=self.verbose,
        )

        return safety_probabilities

    def score(self):
        raise NotImplementedError
