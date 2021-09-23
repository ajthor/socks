from functools import partial

from gym_socks.envs.policy import BasePolicy

import gym_socks.kernel.metrics as kernel

from gym_socks.utils import normalize, indicator_fn
from gym_socks.utils.batch import generate_batches

import numpy as np


class MaximallySafePolicy(BasePolicy):
    """Maximally safe policy."""

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
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):
        if system is None:
            print("Must supply a system.")
            return None

        if S is None:
            print("Must supply a sample.")
            return None

        if U is None:
            print("Must supply a sample.")
            return None

        if A is None:
            print("Must supply a sample.")
            return None

        if constraint_tube is None:
            print("Must supply constraint tube.")
            return None

        if target_tube is None:
            print("Must supply target tube.")
            return None

        if problem != "THT" and problem != "FHT":
            raise ValueError("problem is not in {'THT', 'FHT'}")

    def train(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):

        self._validate_inputs(
            system=system,
            S=S,
            T=T,
            constraint_tube=constraint_tube,
            target_tube=target_tube,
            problem=problem,
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

        CXY = kernel_fn(X, Y)
        CUA = kernel_fn(U, A)

        betaXY = normalize(np.einsum(
            "ii,ij,ik->ikj", W, CXY, CUA, optimize=["einsum_path", (0, 1, 2)]
        ))

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)), dtype=np.float32)

        Vt[num_time_steps, :] = indicator_fn(Y, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            w = np.einsum("i,ijk->kj", Vt[t + 1, :], betaXY,  optimize=["einsum_path", (0, 1)])

            V = np.max(w, axis=1)

            Y_in_safe_set = indicator_fn(Y, constraint_tube[t])

            if problem == "THT":

                Vt[t, :] = Y_in_safe_set * V

            elif problem == "FHT":

                Y_in_target_set = indicator_fn(Y, target_tube[t])

                Vt[t, :] = Y_in_target_set + (Y_in_safe_set & ~Y_in_target_set) * V

        self.kernel_fn = kernel_fn

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

        betaXT = normalize(self.W @ (CXT * self.CUA))

        C = np.matmul(self.cost[time + 1, :], betaXT)

        idx = np.argmax(C)

        return np.array(self.A[idx], dtype=np.float32)
