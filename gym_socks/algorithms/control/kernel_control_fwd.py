"""Forward in time stochastic optimal control."""

from functools import partial

import gym_socks

from gym_socks.envs.policy import BasePolicy
import gym_socks.kernel.metrics as kernel
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

import numpy as np

from tqdm.auto import tqdm


class KernelControlFwd(BasePolicy):
    """
    Stochastic optimal control policy forward in time.

    References
    ----------
    .. [1] `Stochastic Optimal Control via
            Hilbert Space Embeddings of Distributions, 2021
           Adam J. Thorpe, Meeko M. K. Oishi
           IEEE Conference on Decision and Control,
           <https://arxiv.org/abs/2103.12759>`_

    """

    def __init__(self, kernel_fn=None, l=None, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        if l is None:
            l = 1

        self.kernel_fn = kernel_fn
        self.l = l

    def _validate_inputs(
        self,
        system=None,
        S: "State sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
        constraint_fn: "Constraint function." = None,
    ):

        if system is None:
            raise ValueError("Must supply a system.")

        if S is None:
            raise ValueError("Must supply a sample.")

        if A is None:
            raise ValueError("Must supply a sample.")

        if cost_fn is None:
            raise ValueError("Must supply a cost function.")

    def train(
        self,
        system=None,
        S: "State sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
        constraint_fn: "Constraint function." = None,
    ):

        self._validate_inputs(
            system=system, S=S, A=A, cost_fn=cost_fn, constraint_fn=constraint_fn
        )

        kernel_fn = self.kernel_fn
        l = self.l

        X, U, Y = gym_socks.envs.sample.transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        A = np.array(A)
        # A = A[:, 0, :]

        pbar = ms_tqdm(total=3, bar_format=_progress_fmt)

        W = kernel.regularized_inverse(X, U=U, kernel_fn=kernel_fn, l=l)
        pbar.update()

        CUA = kernel_fn(U, A)
        pbar.update()

        self.X = X
        self.A = A

        self.CUA = CUA

        self.W = W

        self.cost = partial(cost_fn, state=Y)

        self.constraint = None
        if constraint_fn is not None:
            self.constraint = partial(constraint_fn, state=Y)

        pbar.update()
        pbar.close()

    def train_batch(
        self,
        system=None,
        S: "State sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
        constraint_fn: "Constraint function." = None,
        batch_size: "Batch size." = 5,
    ):
        raise NotImplementedError

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(state)

        n = len(self.A)

        CXT = self.kernel_fn(self.X, T)

        betaXT = self.W @ (CXT * self.CUA)
        C = np.matmul(np.array(self.cost(time=time), dtype=np.float32), betaXT)

        if self.constraint is not None:
            D = np.matmul(
                np.array(self.constraint(time=time), dtype=np.float32), betaXT
            )

            satisfies_constraints = np.where(D <= 0)
            CA = C[satisfies_constraints]

            if CA.size == 0:
                idx = np.argmin(C)
                return np.array(self.A[idx], dtype=np.float32)

            idx = np.argmin(CA)
            return np.array(self.A[satisfies_constraints][idx], dtype=np.float32)

        else:
            idx = np.argmin(C)
            return np.array(self.A[idx], dtype=np.float32)
