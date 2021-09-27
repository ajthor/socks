"""Forward in time stochastic optimal control."""

from functools import partial

from gym_socks.envs.policy import BasePolicy

import gym_socks.kernel.metrics as kernel

import numpy as np


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
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
        constraint_fn: "Constraint function." = None,
    ):

        if system is None:
            raise ValueError("Must supply a system.")

        if S is None:
            raise ValueError("Must supply a sample.")

        if U is None:
            raise ValueError("Must supply a sample.")

        if A is None:
            raise ValueError("Must supply a sample.")

        if cost_fn is None:
            raise ValueError("Must supply a cost function.")

    def train(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
        constraint_fn: "Constraint function." = None,
    ):

        self._validate_inputs(
            system=system, S=S, U=U, A=A, cost_fn=cost_fn, constraint_fn=constraint_fn
        )

        kernel_fn = self.kernel_fn
        l = self.l

        S = np.array(S)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        U = np.array(U)
        U = U[:, 0, :]

        A = np.array(A)
        A = A[:, 0, :]

        W = kernel.regularized_inverse(X, U=U, kernel_fn=kernel_fn, l=l)

        CUA = kernel_fn(U, A)

        self.X = X
        self.A = A

        self.CUA = CUA

        self.W = W

        self.cost = partial(cost_fn, state=Y)

        self.constraint = None
        if constraint_fn is not None:
            self.constraint = partial(constraint_fn, state=Y)

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
