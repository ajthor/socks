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

    def __init__(self, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        # Global algorithm parameters go here.

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

        kernel_fn = getattr(self, "kernel_fn", None)
        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        l = getattr(self, "l", None)
        if l is None:
            l = 1

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(T)

        # make sure shape of sample is (:, 2, :)

        S = np.array(S)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        CXY = kernel_fn(X, Y)
        CXT = kernel_fn(X, T)

        betaXY = normalize(np.matmul(W, CXY))
        betaXT = normalize(np.matmul(W, CXT))

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Y],
            dtype=np.float32,
        )

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)))

        Pr[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in T],
            dtype=np.float32,
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            print(f"Computing for k={t}")

            Vt_betaXY = np.matmul(Vt[t + 1, :], betaXY)
            Vt_betaXT = np.matmul(Vt[t + 1, :], betaXT)

            Y_in_safe_set = [
                constraint_tube[t].contains(np.array(point)) for point in Y
            ]

            T_in_safe_set = [
                constraint_tube[t].contains(np.array(point)) for point in T
            ]

            if problem == "THT":

                Vt[t, :] = np.array(Y_in_safe_set, dtype=np.float32) * Vt_betaXY
                Pr[t, :] = np.array(T_in_safe_set, dtype=np.float32) * Vt_betaXT

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
                    * Vt_betaXY
                )

                T_in_target_set = [
                    target_tube[t].contains(np.array(point)) for point in T
                ]

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

    def __init__(self, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        # Global algorithm parameters go here.

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

        kernel_fn = getattr(self, "kernel_fn", None)
        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        l = getattr(self, "l", None)
        if l is None:
            l = 1

        Xt = np.array(T)
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

        CUA = np.expand_dims(kernel_fn(U, A), axis=2)

        CXT = kernel_fn(X, T)

        betaXY = np.repeat(np.expand_dims(np.matmul(W, CXY), axis=2), len(A), axis=2)
        betaXY = np.moveaxis(betaXY, 1, 2)
        betaXY = np.multiply(betaXY, CUA)
        betaXY = normalize(betaXY)

        betaXT = np.repeat(np.expand_dims(np.matmul(W, CXT), axis=2), len(A), axis=2)
        betaXT = np.moveaxis(betaXT, 1, 2)
        betaXT = np.multiply(betaXT, CUA)
        betaXT = normalize(betaXT)

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Y],
            dtype=np.float32,
        )

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)))

        Pr[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in T],
            dtype=np.float32,
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            print(f"Computing for k={t}")

            wX = np.zeros((len(X), len(A)))
            wT = np.zeros((len(T), len(A)))

            for p in range(len(A)):
                wX[:, p] = np.matmul(Vt[t + 1, :], np.squeeze(betaXY[:, p, :]))
                wT[:, p] = np.matmul(Vt[t + 1, :], np.squeeze(betaXT[:, p, :]))

            VX = np.max(wX, axis=1)
            VT = np.max(wT, axis=1)

            # compute value functions
            Y_in_safe_set = [
                constraint_tube[t].contains(np.array(point)) for point in Y
            ]

            # compute safety probabilities
            T_in_safe_set = [
                constraint_tube[t].contains(np.array(point)) for point in T
            ]

            if problem == "THT":

                Vt[t, :] = np.array(Y_in_safe_set, dtype=np.float32) * VX
                Pr[t, :] = np.array(T_in_safe_set, dtype=np.float32) * VT

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
                    * VX
                )

                T_in_target_set = [
                    target_tube[t].contains(np.array(point)) for point in T
                ]

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        kernel_fn = getattr(self, "kernel_fn", None)
        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        l = getattr(self, "l", None)
        if l is None:
            l = 1

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
        CUA = np.expand_dims(kernel_fn(U, A), axis=2)

        # betaXY = normalize(np.matmul(W, CXY))
        betaXY = np.repeat(np.expand_dims(np.matmul(W, CXY), axis=2), len(A), axis=2)
        betaXY = np.moveaxis(betaXY, 1, 2)
        betaXY = np.multiply(betaXY, CUA)
        betaXY = normalize(betaXY)

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Y],
            dtype=np.float32,
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            print(f"Computing for k={t}")

            # Vt_betaXY = np.matmul(Vt[t + 1, :], betaXY)

            w = np.zeros((len(X), len(A)))

            for p in range(len(A)):
                w[:, p] = np.matmul(Vt[t + 1, :], np.squeeze(betaXY[:, p, :]))

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
        # betaXT = np.multiply(self.W, CXT)
        # betaXT = np.expand_dims(betaXT, axis=2)
        # betaXT = np.repeat(betaXT, n, axis=2)
        betaXT = np.repeat(np.expand_dims(np.matmul(self.W, CXT), axis=2), n, axis=2)
        betaXT = np.moveaxis(betaXT, 1, 2)
        betaXT = np.multiply(betaXT, self.CUA)
        betaXT = normalize(betaXT)

        w = np.zeros((len(T), len(self.A)))

        for p in range(len(self.A)):
            w[:, p] = np.matmul(self.Vt[time + 1, :], np.squeeze(betaXT[:, p, :]))

        idx = np.argmax(w, axis=1)

        return self.A[idx]
