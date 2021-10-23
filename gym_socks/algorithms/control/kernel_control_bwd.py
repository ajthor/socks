"""Backward in time stochastic optimal control.

References:
    .. [1] `Stochastic Optimal Control via
            Hilbert Space Embeddings of Distributions, 2021
            Adam J. Thorpe, Meeko M. K. Oishi
            IEEE Conference on Decision and Control,
            <https://arxiv.org/abs/2103.12759>`_

"""

from functools import partial

import gym_socks

from gym_socks.envs.policy import BasePolicy
from gym_socks.algorithms.control.control_common import compute_solution
from gym_socks.kernel.metrics import rbf_kernel, regularized_inverse
from gym_socks.envs.sample import transpose_sample
from gym_socks.utils import generate_batches
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

import numpy as np


def _compute_backward_recursion(
    Y,
    W,
    CXY,
    CUA,
    num_steps=None,
    cost_fn=None,
    constraint_fn=None,
    value_functions=None,
    out=None,
    verbose=False,
):
    # Compute beta.
    gym_socks.logger.debug("Computing beta.")
    beta = np.einsum("ij,ik->ijk", CXY, CUA)

    pbar = ms_tqdm(
        total=num_steps,
        bar_format=_progress_fmt,
        disable=False if verbose is True else True,
    )

    # set up empty array to hold value functions
    if out is None:
        out = np.zeros((num_steps, len(Y)), dtype=np.float32)
    out[num_steps - 1, :] = np.array(cost_fn(time=num_steps - 1), dtype=np.float32)
    pbar.update()

    # run backwards in time and compute the safety probabilities
    for t in range(num_steps - 2, -1, -1):

        Z = np.matmul(value_functions[t + 1, :], W)
        C = np.einsum("i,ijk->jk", Z, beta)

        if constraint_fn is not None:
            R = np.matmul(constraint_fn(time=t), W)
            D = np.einsum("i,ijk->jk", R, beta)

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


def _compute_backward_recursion_batch(
    Y,
    W,
    CXY,
    CUA,
    num_steps=None,
    cost_fn=None,
    constraint_fn=None,
    value_functions=None,
    out=None,
    batch_size=None,
    verbose=False,
):
    pbar = ms_tqdm(
        total=num_steps,
        bar_format=_progress_fmt,
        disable=False if verbose is True else True,
    )

    # set up empty array to hold value functions
    if out is None:
        out = np.zeros((num_steps + 1, len(Y)), dtype=np.float32)
    out[num_steps - 1, :] = np.array(cost_fn(time=num_steps), dtype=np.float32)
    pbar.update()

    # run backwards in time and compute the safety probabilities
    for t in range(num_steps - 2, -1, -1):

        if constraint_fn is not None:

            Z = np.matmul(value_functions[t + 1, :], W)
            R = np.matmul(constraint_fn(time=t), W)

            C = np.zeros_like(CUA)
            D = np.zeros_like(CUA)

            for batch in generate_batches(num_elements=len(Y), batch_size=batch_size):

                beta = np.einsum("ij,ik->ijk", CXY[:, batch], CUA)
                C[batch, :] = np.einsum("i,ijk->jk", Z, beta)
                D[batch, :] = np.einsum("i,ijk->jk", R, beta)

            V = np.zeros((len(Y),))
            for i in range(len(Y)):
                CA = C[i][D[i] <= 0]
                if CA.size == 0:
                    V[i] = np.min(C[i])
                else:
                    V[i] = np.min(CA)

            out[t, :] = cost_fn(time=t) + V

        else:

            Z = np.matmul(value_functions[t + 1, :], W)

            C = np.zeros_like(CUA)

            for batch in generate_batches(num_elements=len(Y), batch_size=batch_size):

                beta = np.einsum("ij,ik->ijk", CXY[:, batch], CUA)
                C[batch, :] = np.einsum("i,ijk->jk", Z, beta)

            V = np.min(C, axis=1)

            out[t, :] = cost_fn(time=t) + V

        pbar.update()

    pbar.close()

    return out


def kernel_control_bwd(
    S=None,
    A=None,
    num_steps=None,
    cost_fn=None,
    constraint_fn=None,
    heuristic=False,
    regularization_param=None,
    kernel_fn=None,
    batch_size=None,
    verbose: bool = True,
):

    alg = KernelControlBwd(
        num_steps=num_steps,
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
    """Stochastic optimal control policy backward in time."""

    def __init__(
        self,
        num_steps=None,
        cost_fn=None,
        constraint_fn=None,
        heuristic=False,
        regularization_param=None,
        kernel_fn=None,
        batch_size=None,
        verbose: bool = True,
        *args,
        **kwargs
    ):
        """Initialize the algorithm."""
        super().__init__(*args, **kwargs)

        self.num_steps = num_steps

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

    def train(self, S=None, A=None):

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

        gym_socks.logger.debug("Computing matrix inverse.")
        self.W = regularized_inverse(
            X,
            U=U,
            kernel_fn=self.kernel_fn,
            regularization_param=self.regularization_param,
        )

        gym_socks.logger.debug("Computing covariance matrices.")
        self.CUA = self.kernel_fn(U, A)
        CXY = self.kernel_fn(X, Y)

        gym_socks.logger.debug("Computing cost function.")
        self.cost_fn = partial(self.cost_fn, state=Y)

        gym_socks.logger.debug("Computing constraint function.")
        if self.constraint_fn is not None:
            self.constraint_fn = partial(self.constraint_fn, state=Y)

        gym_socks.logger.debug("Computing value functions.")
        value_functions = np.empty((self.num_steps, len(Y)))
        self.value_functions = self._compute_backward_recursion_caller(
            Y,
            self.W,
            CXY,
            self.CUA,
            num_steps=self.num_steps,
            cost_fn=self.cost_fn,
            constraint_fn=self.constraint_fn,
            value_functions=value_functions,
            out=value_functions,
            verbose=True,
        )

    def __call__(self, time=0, T=None, *args, **kwargs):

        if T is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(T)

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
        gym_socks.logger.debug("Computing solution to the LP.")
        sol = compute_solution(C, D, heuristic=self.heuristic)
        idx = np.argmax(sol)
        return np.array(self.A[idx], dtype=np.float32)
