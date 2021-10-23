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
    num_steps=None,
    constraint_tube=None,
    target_tube=None,
    value_functions=None,
    out=None,
    step_fn=None,
    verbose=False,
):
    # Compute beta.
    gym_socks.logger.debug("Computing beta.")
    betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))

    pbar = ms_tqdm(
        total=num_steps,
        bar_format=_progress_fmt,
        desc="Computing backward recursion.",
        disable=False if verbose is True else True,
    )

    # Initialize the backward recursion.
    if out is None:
        out = np.empty((num_steps, len(Y)))
    out[num_steps - 1, :] = indicator_fn(Y, target_tube[num_steps - 1])
    pbar.update()

    # Backward recursion.
    for t in range(num_steps - 2, -1, -1):
        V = np.einsum("i,ij->j", value_functions[t + 1, :], betaXY)
        out[t, :] = step_fn(Y, V, constraint_tube[t], target_tube[t])

        pbar.update()

    pbar.close()

    return out


def _compute_backward_recursion_batch(
    Y,
    W,
    CXY,
    num_steps=None,
    constraint_tube=None,
    target_tube=None,
    value_functions=None,
    out=None,
    step_fn=None,
    batch_size=None,
    verbose=False,
):
    pbar = ms_tqdm(
        total=num_steps,
        bar_format=_progress_fmt,
        desc="Computing backward recursion.",
        disable=False if verbose is True else True,
    )

    # Initialize the backward recursion.
    if out is None:
        out = np.empty((num_steps, len(Y)))
    out[num_steps - 1, :] = indicator_fn(Y, target_tube[num_steps - 1])
    pbar.update()

    # Backward recursion.
    for t in range(num_steps - 2, -1, -1):

        for batch in generate_batches(len(Y), batch_size=batch_size):

            # Compute beta.
            betaXY = normalize(np.einsum("ii,ij->ij", W, CXY[:, batch]))

            V = np.einsum("i,ij->j", value_functions[t + 1, :], betaXY)
            out[t, batch] = step_fn(Y[batch], V, constraint_tube[t], target_tube[t])

        pbar.update()

    pbar.close()

    return out


def kernel_sr(
    S=None,
    T=None,
    num_steps=None,
    constraint_tube=None,
    target_tube=None,
    problem="THT",
    regularization_param=None,
    kernel_fn=None,
    batch_size=None,
    verbose=False,
):

    alg = KernelSR(
        num_steps=num_steps,
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


class KernelSR(AlgorithmInterface):
    """Stochastic reachability using kernel distribution embeddings."""

    def __init__(
        self,
        num_steps=None,
        constraint_tube=None,
        target_tube=None,
        problem="THT",
        kernel_fn=None,
        regularization_param=None,
        batch_size=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        """Initialize the algorithm."""
        super().__init__(*args, **kwargs)

        self.num_steps = num_steps
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

        if self.num_steps is None:
            raise ValueError("Must supply a num_steps.")

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

    def fit(self, S):
        """Run the algorithm.

        Args:

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

        gym_socks.logger.debug("Computing matrix inverse.")
        self.W = regularized_inverse(
            X, kernel_fn=self.kernel_fn, regularization_param=self.regularization_param
        )

        gym_socks.logger.debug("Computing covariance matrix.")
        CXY = self.kernel_fn(X, Y)

        gym_socks.logger.debug("Computing value functions.")
        value_functions = np.empty((self.num_steps, len(Y)))
        self.value_functions = self._compute_backward_recursion_caller(
            Y,
            self.W,
            CXY,
            num_steps=self.num_steps,
            constraint_tube=self.constraint_tube,
            target_tube=self.target_tube,
            value_functions=value_functions,
            out=value_functions,
            step_fn=self.step_fn,
            verbose=True,
        )

    def predict(self, T):
        """Predict."""
        self._validate_data(T)
        T = np.array(T)

        gym_socks.logger.debug("Computing covariance matrix.")
        CXT = self.kernel_fn(self.X, T)

        gym_socks.logger.debug("Computing safety probabilities.")
        safety_probabilities = self._compute_backward_recursion_caller(
            T,
            self.W,
            CXT,
            num_steps=self.num_steps,
            constraint_tube=self.constraint_tube,
            target_tube=self.target_tube,
            value_functions=self.value_functions,
            step_fn=self.step_fn,
            verbose=True,
        )

        return safety_probabilities
