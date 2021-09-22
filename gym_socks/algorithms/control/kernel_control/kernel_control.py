from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface
from gym_socks.envs.policy import BasePolicy

import gym_socks.kernel.metrics as kernel

import numpy as np
from numpy.linalg import norm


class KernelControlFwd(BasePolicy):
    """Stochastic optimal control policy forward in time."""

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

    def train(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
    ):

        if system is None:
            print("Must supply a system.")

        if S is None:
            print("Must supply a sample.")

        if U is None:
            print("Must supply a sample.")
            return None

        if A is None:
            print("Must supply a sample.")
            return None

        if cost_fn is None:
            print("Must supply a cost function.")
            return None

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

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(state)

        n = len(self.A)

        CXT = self.kernel_fn(self.X, T)

        # betaXT = np.einsum(
        #     "ii,ij,ik->ikj", self.W, CXT, self.CUA, optimize=["einsum_path", (0, 1, 2)]
        # )
        #
        # w = np.einsum(
        #     "i,ijk->kj",
        #     np.array(self.cost(time=time)),
        #     betaXT,
        #     optimize=["einsum_path", (0, 1)],
        # )

        w = np.einsum(
            "i,ii,ij,ik->jk",
            np.array(self.cost(time=time), dtype=np.float32),
            self.W,
            CXT,
            self.CUA,
            optimize="greedy",
        )

        idx = np.argmin(w, axis=1)

        return self.A[idx]


class KernelControlBwd(BasePolicy):
    """Stochastic optimal control policy forward in time."""

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

    def train(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
    ):

        if system is None:
            print("Must supply a system.")

        if S is None:
            print("Must supply a sample.")

        if U is None:
            print("Must supply a sample.")
            return None

        if A is None:
            print("Must supply a sample.")
            return None

        if cost_fn is None:
            print("Must supply a cost function.")
            return None

        kernel_fn = self.kernel_fn
        l = self.l

        num_time_steps = system.num_time_steps - 1

        S = np.array(S)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        U = np.array(U)
        U = U[:, 0, :]

        A = np.array(A)
        A = A[:, 0, :]

        W = kernel.regularized_inverse(X, U=U, kernel_fn=kernel_fn, l=l)

        CUA = kernel_fn(U, A)

        CXY = kernel_fn(X, Y)

        betaXY = np.einsum(
            "ii,ij,ik->jk", W, CXY, CUA, optimize=["einsum_path", (0, 1), (0, 1)]
        )

        # betaXY = np.einsum('ii,ij,ik->ikj', W, CXY, CUA)

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)), dtype=np.float32)

        Vt[num_time_steps, :] = np.array(
            cost_fn(time=num_time_steps, state=Y), dtype=np.float32
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            w = np.einsum("i,ik->ik", Vt[t + 1, :], betaXY)

            # w = np.einsum('i,ikj->jk', Vt[t + 1, :], betaXY)

            # Z = np.einsum('i,ii->i', Vt[t + 1, :], W)
            #
            # w = np.zeros((len(X), len(A)))
            # for p in range(len(X)):
            #     w[p, :] = np.matmul(Z, np.multiply(kernel_fn(X, Y[p].reshape(1, -1)), CUA))

            V = np.min(w, axis=1)

            Vt[t, :] = cost_fn(time=t, state=Y) + V

        self.X = X
        self.A = A

        self.CUA = CUA

        self.W = W

        self.cost = Vt

    def __call__(self, time=0, state=None, *args, **kwargs):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(state)

        n = len(self.A)

        CXT = self.kernel_fn(self.X, T)

        betaXT = np.einsum(
            "ii,ij,ik->ikj", self.W, CXT, self.CUA, optimize=["einsum_path", (0, 1, 2)]
        )

        w = np.einsum(
            "i,ikj->jk",
            np.array(self.cost[time], dtype=np.float32),
            betaXT,
            optimize=["einsum_path", (0, 1)],
        )

        # w = np.einsum(
        #     "i,ii,ij,ik->jk",
        #     np.array(self.cost[time], dtype=np.float32),
        #     self.W,
        #     CXT,
        #     self.CUA,
        #     optimize="greedy",
        # )

        idx = np.argmin(w, axis=1)

        return self.A[idx]
