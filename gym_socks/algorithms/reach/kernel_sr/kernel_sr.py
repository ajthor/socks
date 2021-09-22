from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface
from gym_socks.envs.policy import BasePolicy

import gym_socks.kernel.metrics as kernel

import numpy as np


def normalize(v):
    return v / np.sum(v, axis=0)


class KernelSR(AlgorithmInterface):
    """
    Stochastic reachability using kernel distribution embeddings.
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

    def run(
        self,
        system=None,
        S: "State sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):
        """
        Run the algorithm.
        """

        if system is None:
            print("Must supply a system.")
            return None

        if S is None:
            print("Must supply a sample.")
            return None

        if T is None:
            print("Must supply test points.")
            return None

        if constraint_tube is None:
            print("Must supply constraint tube.")
            return None

        if target_tube is None:
            print("Must supply target tube.")
            return None

        if problem != "THT" and problem != "FHT":
            raise ValueError("problem is not in {'THT', 'FHT'}")

        kernel_fn = self.kernel_fn
        l = self.l

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(T)

        S = np.array(S)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        CXY = kernel_fn(X, Y)
        CXT = kernel_fn(X, T)

        betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))
        betaXT = normalize(np.einsum("ii,ij->ij", W, CXT))

        tt_low = target_tube[num_time_steps].low
        tt_high = target_tube[num_time_steps].high

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            np.all(Y >= tt_low, axis=1) & np.all(Y <= tt_high, axis=1), dtype=np.float32
        )

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)))

        Pr[num_time_steps, :] = np.array(
            np.all(T >= tt_low, axis=1) & np.all(T <= tt_high, axis=1), dtype=np.float32
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            Vt_betaXY = np.einsum("i,ij->j", Vt[t + 1, :], betaXY)
            Vt_betaXT = np.einsum("i,ij->j", Vt[t + 1, :], betaXT)

            ct_low = constraint_tube[t].low
            ct_high = constraint_tube[t].high

            Y_in_safe_set = np.all(Y >= ct_low, axis=1) & np.all(Y <= ct_high, axis=1)
            T_in_safe_set = np.all(T >= ct_low, axis=1) & np.all(T <= ct_high, axis=1)

            if problem == "THT":

                Vt[t, :] = np.array(Y_in_safe_set, dtype=np.float32) * Vt_betaXY
                Pr[t, :] = np.array(T_in_safe_set, dtype=np.float32) * Vt_betaXT

            elif problem == "FHT":

                tt_low = target_tube[t].low
                tt_high = target_tube[t].high

                Y_in_target_set = np.all(Y >= tt_low, axis=1) & np.all(
                    Y <= tt_high, axis=1
                )

                Vt[t, :] = (
                    np.array(Y_in_target_set, dtype=np.float32)
                    + np.array(
                        Y_in_safe_set & ~np.array(Y_in_target_set, dtype=bool),
                        dtype=np.float32,
                    )
                    * Vt_betaXY
                )

                T_in_target_set = np.all(T >= tt_low, axis=1) & np.all(
                    T <= tt_high, axis=1
                )

                Pr[t, :] = (
                    np.array(T_in_target_set, dtype=np.float32)
                    + np.array(
                        T_in_safe_set & ~np.array(T_in_target_set, dtype=bool),
                        dtype=np.float32,
                    )
                    * Vt_betaXT
                )

        return Pr, Vt


class KernelMaximalSR(AlgorithmInterface):
    """
    Stochastic reachability using kernel distribution embeddings.

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

    def run(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):
        """
        Run the algorithm.
        """

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

        if T is None:
            print("Must supply test points.")
            return None

        if constraint_tube is None:
            print("Must supply constraint tube.")
            return None

        if target_tube is None:
            print("Must supply target tube.")
            return None

        if problem != "THT" and problem != "FHT":
            raise ValueError("problem is not in {'THT', 'FHT'}")

        kernel_fn = self.kernel_fn
        l = self.l

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(T)

        # make sure shape of sample is (:, 2, :)

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

        CXT = kernel_fn(X, T)

        betaXY = normalize(
            np.einsum("ii,ij,ik->ikj", W, CXY, CUA, optimize=["einsum_path", (0, 1, 2)])
        )

        betaXT = normalize(
            np.einsum("ii,ij,ik->ikj", W, CXT, CUA, optimize=["einsum_path", (0, 1, 2)])
        )

        tt_low = target_tube[num_time_steps].low
        tt_high = target_tube[num_time_steps].high

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)), dtype=np.float32)

        Vt[num_time_steps, :] = np.array(
            np.all(Y >= tt_low, axis=1) & np.all(Y <= tt_high, axis=1), dtype=np.float32
        )

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)), dtype=np.float32)

        Pr[num_time_steps, :] = np.array(
            np.all(T >= tt_low, axis=1) & np.all(T <= tt_high, axis=1), dtype=np.float32
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            # use optimized multiplications
            wX = np.einsum(
                "i,ikj->jk", Vt[t + 1, :], betaXY, optimize=["einsum_path", (0, 1)]
            )
            wT = np.einsum(
                "i,ikj->jk", Vt[t + 1, :], betaXT, optimize=["einsum_path", (0, 1)]
            )

            VX = np.max(wX, axis=1)
            VT = np.max(wT, axis=1)

            ct_low = constraint_tube[t].low
            ct_high = constraint_tube[t].high

            Y_in_safe_set = np.all(Y >= ct_low, axis=1) & np.all(Y <= ct_high, axis=1)
            T_in_safe_set = np.all(T >= ct_low, axis=1) & np.all(T <= ct_high, axis=1)

            if problem == "THT":

                Vt[t, :] = np.array(Y_in_safe_set, dtype=np.float32) * VX
                Pr[t, :] = np.array(T_in_safe_set, dtype=np.float32) * VT

            elif problem == "FHT":

                tt_low = target_tube[t].low
                tt_high = target_tube[t].high

                Y_in_target_set = np.all(Y >= tt_low, axis=1) & np.all(
                    Y <= tt_high, axis=1
                )

                Vt[t, :] = (
                    np.array(Y_in_target_set, dtype=np.float32)
                    + np.array(
                        Y_in_safe_set & ~np.array(Y_in_target_set, dtype=bool),
                        dtype=np.float32,
                    )
                    * VX
                )

                T_in_target_set = np.all(T >= tt_low, axis=1) & np.all(
                    T <= tt_high, axis=1
                )

                Pr[t, :] = (
                    np.array(T_in_target_set, dtype=np.float32)
                    + np.array(
                        T_in_safe_set & ~np.array(T_in_target_set, dtype=bool),
                        dtype=np.float32,
                    )
                    * VT
                )

        return Pr, Vt


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

        kernel_fn = self.kernel_fn
        l = self.l

        num_time_steps = system.num_time_steps - 1

        # make sure shape of sample is (:, 2, :)

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

        betaXY = np.einsum(
            "ii,ij,ik->ikj", W, CXY, CUA, optimize=["einsum_path", (0, 1, 2)]
        )
        betaXY = normalize(betaXY)

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Y],
            dtype=np.float32,
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            # Vt_betaXY = np.matmul(Vt[t + 1, :], betaXY)

            # w = np.zeros((len(X), len(A)))
            #
            # for p in range(len(A)):
            #     w[:, p] = np.matmul(Vt[t + 1, :], np.squeeze(betaXY[:, p, :]))

            w = np.einsum("i,ijk->kj", Vt[t + 1, :], betaXY)

            V = np.max(w, axis=1)

            Y_in_safe_set = [
                constraint_tube[t].contains(np.array(point)) for point in Y
            ]

            if problem == "THT":

                Vt[t, :] = np.array(Y_in_safe_set, dtype=np.float32) * V

            elif problem == "FHT":

                Y_in_target_set = [
                    target_tube[t].contains(np.array(point)) for point in Y
                ]

                Vt[t, :] = (
                    np.array(Y_in_target_set, dtype=np.float32)
                    + np.array(
                        Y_in_safe_set & ~np.array(Y_in_target_set, dtype=bool),
                        dtype=np.float32,
                    )
                    * V
                )

        self.kernel_fn = kernel_fn

        self.X = X
        self.A = A

        self.CUA = CUA

        self.W = W

        self.Vt = Vt

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
        betaXT = normalize(betaXT)

        w = np.einsum(
            "i,ijk->kj", self.Vt[time + 1, :], betaXT, optimize=["einsum_path", (0, 1)]
        )

        idx = np.argmax(w, axis=1)

        return self.A[idx]
