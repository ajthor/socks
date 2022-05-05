import unittest

from functools import partial

import gym

from gym_socks.kernel.metrics import euclidean_distance
from gym_socks.kernel.metrics import rbf_kernel as rbf_kernel_socks
from gym_socks.kernel.metrics import regularized_inverse
from gym_socks.kernel.metrics import delta_kernel
from gym_socks.kernel.probability import probability_matrix

import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel

from sklearn.metrics.pairwise import euclidean_distances

sklearn_kernel_list = [linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel]


class TestEuclideanDistance(unittest.TestCase):
    def test_euclidean_distance(cls):
        """Test that euclidean distance computes correctly."""

        X = np.arange(4).reshape((2, 2))
        Y = np.arange(4).reshape((2, 2))

        groundTruth = np.array([[0.0, 8.0], [8.0, 0.0]], dtype=np.float32)

        distance = euclidean_distance(Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))

        distance = euclidean_distance(X, Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))

        groundTruth = euclidean_distances(X, Y, squared=True)

        distance = euclidean_distance(Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))

        distance = euclidean_distance(X, Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))


class TestRBFKernel(unittest.TestCase):
    def test_rbf_kernel_same_size(cls):
        """Test that RBF kernel computes correctly with same size inputs."""

        Y = np.arange(4).reshape((2, 2))

        groundTruth = np.array(
            [[1.0, 0.77880078], [0.77880078, 1.0]],
            dtype=np.float32,
        )

        K = rbf_kernel_socks(Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = rbf_kernel_socks(Y, Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = rbf_kernel_socks(Y, distance_fn=euclidean_distances)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = rbf_kernel_socks(Y, Y, distance_fn=euclidean_distances)
        cls.assertTrue(np.allclose(K, groundTruth))

    def test_rbf_kernel_different_size(cls):
        """Test that RBF kernel computes correctly with different-sized inputs."""

        X = np.arange(4).reshape((2, 2))
        Y = np.arange(6).reshape((3, 2))

        groundTruth = np.array(
            [
                [
                    [1.0, 0.93941306, 0.77880078],
                    [0.93941306, 1.0, 0.93941306],
                    [0.77880078, 0.93941306, 1.0],
                ],
            ],
            dtype=np.float32,
        )

        K = rbf_kernel_socks(Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = rbf_kernel_socks(Y, Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        groundTruth = np.array(
            [[1.0, 0.93941306, 0.77880078], [0.93941306, 1.0, 0.93941306]],
            dtype=np.float32,
        )

        K = rbf_kernel_socks(X, Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = rbf_kernel_socks(X, Y, distance_fn=euclidean_distances)
        cls.assertTrue(np.allclose(K, groundTruth))

    def test_close_to_sklearn(cls):
        """Assert that result is close to sklearn."""

        Y = np.arange(4).reshape((2, 2))

        D = euclidean_distance(Y, squared=True)

        groundTruth = rbf_kernel(Y, gamma=1 / (2 * np.median(D) ** 2))

        K = rbf_kernel_socks(Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = rbf_kernel_socks(Y, Y)
        cls.assertTrue(np.allclose(K, groundTruth))


class TestRegularizedInverse(unittest.TestCase):
    def test_regularized_inverse(cls):
        """Test that regularized inverse computes correctly."""

        Y = np.arange(4).reshape((2, 2))

        groundTruth = np.array(
            [[0.66676608, -0.0081415], [-0.0081415, 0.66676608]],
            dtype=np.float32,
        )

        W = regularized_inverse(Y)
        cls.assertTrue(np.allclose(W, groundTruth))

    def test_sklearn_kernels(cls):
        """Sklearn kernels should work with regularized_inverse."""

        Y = np.arange(4).reshape((2, 2))

        for kernel_fn in sklearn_kernel_list:
            with cls.subTest(msg=f"Testing with {type(kernel_fn)}."):

                try:

                    W = regularized_inverse(Y, Y, kernel_fn=kernel_fn)

                except Exception as e:
                    cls.fail(f"Kernel {type(kernel_fn)} raised an exception.")


class TestDeltaKernel(unittest.TestCase):
    def test_delta_kernel(cls):
        """Test that delta kernel computes correctly."""

        X = np.array(
            [
                [0, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=int,
        )

        G = delta_kernel(X)

        cls.assertTrue(np.allclose(G, np.identity(3)))

        Y = np.array(
            [
                [1, 1],
                [2, 0],
                [1, 0],
                [1, 3],
            ],
            dtype=int,
        )

        G = delta_kernel(X, Y)

        groundTruth = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=int)

        cls.assertTrue(np.allclose(G, groundTruth))


class TestProbabilityMatrix(unittest.TestCase):
    def test_probability_matrix_1D(cls):
        """Test that the probability matrix is correct for 1D discrete data."""

        # A list of all possible states.
        states = np.array([[0], [1], [2]], dtype=int)

        # Mock data. X represents the FROM state, Y represents the TO state.
        X = np.array([[0], [0], [0], [1], [2]], dtype=int)
        Y = np.array([[1], [1], [2], [1], [2]], dtype=int)

        P = probability_matrix(states, X, Y)

        groundTruth = np.array(
            [
                [0, 2 / 3, 1 / 3],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        cls.assertTrue(np.allclose(P, groundTruth))

    def test_probability_matrix_multi_dim(cls):
        """Test that the probability matrix is correct for 2D discrete data."""

        # A list of all possible states.
        states = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=int,
        )

        # Mock data. X represents the FROM state, Y represents the TO state.
        X = np.array([[0, 0], [0, 0], [0, 1], [1, 0], [1, 0], [1, 1]], dtype=int)
        Y = np.array([[0, 1], [1, 1], [0, 1], [1, 0], [1, 1], [1, 1]], dtype=int)

        groundTruth = np.array(
            [
                [0, 0.5, 0, 0.5],
                [0, 1, 0, 0],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Compute manually for sanity.
        Gx = delta_kernel(states, X)
        Gy = delta_kernel(states, Y)

        Sx = np.sum(Gx, axis=1, keepdims=True)

        transitions = Gx @ Gy.T
        totals = Sx * transitions

        P = totals / np.sum(totals, axis=1, keepdims=True)

        cls.assertTrue(np.allclose(P, groundTruth))

        P = probability_matrix(states, X, Y)

        cls.assertTrue(np.allclose(P, groundTruth))
