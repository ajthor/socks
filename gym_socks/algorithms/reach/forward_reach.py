from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface

import gym_socks.kernel.metrics as kernel

from gym_socks.utils import normalize, indicator_fn, generate_batches

import numpy as np


class KernelForwardReachClassifier(AlgorithmInterface):
    """
    Stochastic reachability using kernel distribution embeddings (RFF).
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
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):

        if system is None:
            raise ValueError("Must supply a system.")

        if S is None:
            raise ValueError("Must supply a sample.")

        if T is None:
            raise ValueError("Must supply test points.")

        if constraint_tube is None:
            raise ValueError("Must supply constraint tube.")

        if target_tube is None:
            raise ValueError("Must supply target tube.")

        if problem != "THT" and problem != "FHT":
            raise ValueError("problem is not in {'THT', 'FHT'}")

    def train(
        self,
        system=None,
        S: "State sample." = None,
    ):
        """
        Run the algorithm.
        """

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

        S = np.array(S)
        X = S[:, 0, :]
        # Y = S[:, 1, :]

        K = kernel_fn(X)

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        tau = 1 - np.min(np.diagonal((1 / len(X)) * K.T @ W @ K))

        self.X = X
        self.W = W

        self.tau = tau

    def classify(self, state=None):

        if state is None:
            print("Must supply a state to the policy.")
            return None

        T = np.array(state)

        K = self.kernel_fn(self.X, T)

        C = np.diagonal((1 / len(X)) * K.T @ self.W @ K)

        return np.all(C >= 1 - self.tau, axis=1)
