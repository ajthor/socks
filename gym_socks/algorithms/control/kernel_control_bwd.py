"""Backward in time stochastic optimal control."""

from functools import partial

from gym_socks.envs.policy import BasePolicy

import gym_socks.kernel.metrics as kernel

from gym_socks.utils import generate_batches

import numpy as np


class KernelControlBwd(BasePolicy):
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

        # betaXY = np.einsum("ii,ij,ik->ijk", W, CXY, CUA)

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)), dtype=np.float32)

        Vt[num_time_steps, :] = np.array(
            cost_fn(time=num_time_steps, state=Y), dtype=np.float32
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            if constraint_fn is not None:

                Z = np.matmul(Vt[t + 1, :], W)
                R = np.matmul(constraint_fn(time=t, state=Y), W)

                beta = np.einsum("ij,ik->ijk", CXY, CUA)
                C = np.einsum("i,ijk->jk", Z, beta)
                D = np.einsum("i,ijk->jk", R, beta)

                # D = np.einsum("i,ijk->jk", constraint_fn(time=t, state=Y), betaXY)

                V = np.zeros((len(X),))
                for i in range(len(X)):
                    CA = C[i][D[i] <= 0]
                    if CA.size == 0:
                        V[i] = np.min(C[i])
                    else:
                        V[i] = np.min(CA)

            else:

                Z = np.matmul(Vt[t + 1, :], W)

                beta = np.einsum("ij,ik->ijk", CXY, CUA)
                C = np.einsum("i,ijk->jk", Z, beta)

                V = np.min(C, axis=1)

            Vt[t, :] = cost_fn(time=t, state=Y) + V

        self.X = X
        self.A = A

        self.CUA = CUA

        self.W = W

        self.cost = Vt
        self.constraint = None
        if constraint_fn is not None:
            self.constraint = partial(constraint_fn, state=Y)

    def train_batch(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        cost_fn: "Cost function." = None,
        constraint_fn: "Constraint function." = None,
        batch_size: "Batch size." = 5,
    ):

        self._validate_inputs(
            system=system, S=S, U=U, A=A, cost_fn=cost_fn, constraint_fn=constraint_fn
        )

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

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)), dtype=np.float32)

        Vt[num_time_steps, :] = np.array(
            cost_fn(time=num_time_steps, state=Y), dtype=np.float32
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            if constraint_fn is not None:

                Z = np.matmul(Vt[t + 1, :], W)
                R = np.matmul(constraint_fn(time=t, state=Y), W)

                C = np.zeros((len(X), len(A)))
                D = np.zeros((len(X), len(A)))

                for batch in generate_batches(
                    num_elements=len(X), batch_size=batch_size
                ):

                    beta = np.einsum("ij,ik->ijk", CXY[:, batch], CUA)
                    C[batch, :] = np.einsum("i,ijk->jk", Z, beta)
                    D[batch, :] = np.einsum("i,ijk->jk", R, beta)

                V = np.zeros((len(X),))
                for i in range(len(X)):
                    CA = C[i][D[i] <= 0]
                    if CA.size == 0:
                        V[i] = np.min(C[i])
                    else:
                        V[i] = np.min(CA)

                Vt[t, :] = cost_fn(time=t, state=Y) + V

            else:

                Z = np.matmul(Vt[t + 1, :], W)

                C = np.zeros((len(X), len(A)))

                for batch in generate_batches(
                    num_elements=len(X), batch_size=batch_size
                ):

                    beta = np.einsum("ij,ik->ijk", CXY[:, batch], CUA)
                    C[batch, :] = np.einsum("i,ijk->jk", Z, beta)

                V = np.min(C, axis=1)

                Vt[t, :] = cost_fn(time=t, state=Y) + V

        self.X = X
        self.A = A

        self.CUA = CUA

        self.W = W

        self.cost = Vt
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
        C = np.matmul(np.array(self.cost[time], dtype=np.float32), betaXT)

        if self.constraint is not None:

            D = np.matmul(
                np.array(self.constraint(time=time), dtype=np.float32), betaXT
            )

            satisfies_constraints = np.where(D <= 0)
            CA = C[satisfies_constraints]

            idx = np.argmin(CA)
            return np.array(self.A[satisfies_constraints][idx], dtype=np.float32)

        else:
            idx = np.argmin(C)
            return np.array(self.A[idx], dtype=np.float32)
