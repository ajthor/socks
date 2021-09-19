from functools import partial

from algorithms.algorithm import AlgorithmInterface

import kernel.metrics as kernel

import numpy as np
from numpy.linalg import norm


def normalize(v):
    return v / np.sum(v, axis=0)


class KernelControlFwd(AlgorithmInterface):
    """
    Stochastic optimal control forward in time.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        # Global algorithm parameters go here.

    def run(cls, system=None, sample=None, test_points=None, constraint_tube=None):

        if system is None:
            print("Must supply a system.")

        if sample is None:
            print("Must supply a sample.")

        if test_points is None:
            print("Must supply test points.")

        kernel_fn = getattr(cls, "kernel_fn", None)
        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        l = getattr(cls, "l", None)
        if l is None:
            l = 1

        Xt = np.array(test_points)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(test_points)

        # make sure shape of sample is (:, 2, :)

        S = np.array(sample)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        CXY = kernel_fn(X, Y)
        CXXt = kernel_fn(X, Xt)

        betaXY = normalize(np.matmul(W, CXY))
        betaXXt = normalize(np.matmul(W, CXXt))

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Y],
            dtype=np.float32,
        )

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(Xt)))

        Pr[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Xt],
            dtype=np.float32,
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            Vt_betaXY = np.matmul(Vt[t + 1, :], betaXY)
            Vt_betaXXt = np.matmul(Vt[t + 1, :], betaXXt)

            # compute value functions
            Y_in_safe_set = np.array(
                [constraint_tube[t].contains(np.array(point)) for point in Y],
                dtype=np.float32,
            )

            Vt[t, :] = Y_in_safe_set * Vt_betaXY

            # compute safety probabilities
            Xt_in_safe_set = np.array(
                [constraint_tube[t].contains(np.array(point)) for point in Xt],
                dtype=np.float32,
            )

            Pr[t, :] = Xt_in_safe_set * Vt_betaXXt

        return Pr, Vt


class KernelControlBwd(AlgorithmInterface):
    """
    Stochastic optimal control backward in time.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        # Global algorithm parameters go here.

    def run(cls, system=None, sample=None, test_points=None, constraint_tube=None):

        if system is None:
            print("Must supply a system.")

        if sample is None:
            print("Must supply a sample.")

        if test_points is None:
            print("Must supply test points.")

        kernel_fn = getattr(cls, "kernel_fn", None)
        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        l = getattr(cls, "l", None)
        if l is None:
            l = 1

        Xt = np.array(test_points)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(test_points)

        # make sure shape of sample is (:, 2, :)

        S = np.array(sample)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        CXY = kernel_fn(X, Y)
        CXXt = kernel_fn(X, Xt)

        betaXY = normalize(np.matmul(W, CXY))
        betaXXt = normalize(np.matmul(W, CXXt))

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Y],
            dtype=np.float32,
        )

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(Xt)))

        Pr[num_time_steps, :] = np.array(
            [constraint_tube[num_time_steps].contains(np.array(point)) for point in Xt],
            dtype=np.float32,
        )

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            Vt_betaXY = np.matmul(Vt[t + 1, :], betaXY)
            Vt_betaXXt = np.matmul(Vt[t + 1, :], betaXXt)

            # compute value functions
            Y_in_safe_set = np.array(
                [constraint_tube[t].contains(np.array(point)) for point in Y],
                dtype=np.float32,
            )

            Vt[t, :] = Y_in_safe_set * Vt_betaXY

            # compute safety probabilities
            Xt_in_safe_set = np.array(
                [constraint_tube[t].contains(np.array(point)) for point in Xt],
                dtype=np.float32,
            )

            Pr[t, :] = Xt_in_safe_set * Vt_betaXXt

        return Pr, Vt
