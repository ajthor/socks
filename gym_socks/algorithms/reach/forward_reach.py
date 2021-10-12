from functools import partial

import gym_socks
from gym_socks.algorithms.algorithm import AlgorithmInterface

import gym_socks.kernel.metrics as kernel

from gym_socks.utils import normalize, indicator_fn, generate_batches

import numpy as np


class KernelForwardReachClassifier(object):
    """
    Stochastic reachability using kernel distribution embeddings (RFF).

    References
    ----------
    .. [1] `Learning Approximate Forward Reachable Sets Using Separating Kernels, 2021
           Adam J. Thorpe, Kendric R. Ortiz, Meeko M. K. Oishi
           Learning for Dynamics and Control,
           <https://arxiv.org/abs/2011.09678>`_
    """

    def __init__(self, kernel_fn=None, l=None, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        if kernel_fn is None:
            kernel_fn = partial(kernel.abel_kernel, sigma=0.1)

        if l is None:
            l = 1

        self.kernel_fn = kernel_fn
        self.l = l

    def _validate_inputs(
        self,
        S: "State sample." = None,
    ):

        if S is None:
            raise ValueError("Must supply a sample.")

    def train(
        self,
        S: "State sample." = None,
    ):
        """
        Run the algorithm.
        """

        self._validate_inputs(
            S=S,
        )

        kernel_fn = self.kernel_fn
        l = self.l

        X = np.array(S)

        K = kernel_fn(X)

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        tau = 1 - np.min(np.diagonal((1 / len(X)) * K.T @ W @ K))

        self.X = X
        self.W = W

        self.tau = tau

    def classify(self, state=None):

        if state is None:
            print("Must supply a state to the classifier.")
            return None

        T = np.array(state)

        K = self.kernel_fn(self.X, T)

        C = np.diagonal((1 / len(self.X)) * K.T @ self.W @ K)

        return C >= 1 - self.tau
