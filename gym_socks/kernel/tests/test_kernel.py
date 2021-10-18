import unittest

from functools import partial

import gym

import gym_socks.kernel.metrics

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

        distance = gym_socks.kernel.metrics.euclidean_distance(Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))

        distance = gym_socks.kernel.metrics.euclidean_distance(X, Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))

        groundTruth = euclidean_distances(X, Y, squared=True)

        distance = gym_socks.kernel.metrics.euclidean_distance(Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))

        distance = gym_socks.kernel.metrics.euclidean_distance(X, Y, squared=True)
        cls.assertTrue(np.allclose(distance, groundTruth))


class TestRBFKernel(unittest.TestCase):
    def test_rbf_kernel_same_size(cls):
        """Test that RBF kernel computes correctly with same size inputs."""

        Y = np.arange(4).reshape((2, 2))

        groundTruth = np.array(
            [[1.0, 0.77880078], [0.77880078, 1.0]],
            dtype=np.float32,
        )

        K = gym_socks.kernel.metrics.rbf_kernel(Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = gym_socks.kernel.metrics.rbf_kernel(Y, Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = gym_socks.kernel.metrics.rbf_kernel(Y, distance_fn=euclidean_distances)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = gym_socks.kernel.metrics.rbf_kernel(Y, Y, distance_fn=euclidean_distances)
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

        K = gym_socks.kernel.metrics.rbf_kernel(Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = gym_socks.kernel.metrics.rbf_kernel(Y, Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        groundTruth = np.array(
            [[1.0, 0.93941306, 0.77880078], [0.93941306, 1.0, 0.93941306]],
            dtype=np.float32,
        )

        K = gym_socks.kernel.metrics.rbf_kernel(X, Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = gym_socks.kernel.metrics.rbf_kernel(X, Y, distance_fn=euclidean_distances)
        cls.assertTrue(np.allclose(K, groundTruth))

    def test_close_to_sklearn(cls):
        """Assert that result is close to sklearn."""

        Y = np.arange(4).reshape((2, 2))

        D = gym_socks.kernel.metrics.euclidean_distance(Y, squared=True)

        groundTruth = rbf_kernel(Y, gamma=1 / (2 * np.median(D) ** 2))

        K = gym_socks.kernel.metrics.rbf_kernel(Y)
        cls.assertTrue(np.allclose(K, groundTruth))

        K = gym_socks.kernel.metrics.rbf_kernel(Y, Y)
        cls.assertTrue(np.allclose(K, groundTruth))


class TestRegularizedInverse(unittest.TestCase):
    def test_regularized_inverse(cls):
        """Test that regularized inverse computes correctly."""

        Y = np.arange(4).reshape((2, 2))

        groundTruth = np.array(
            [[0.66676608, -0.0081415], [-0.0081415, 0.66676608]],
            dtype=np.float32,
        )

        W = gym_socks.kernel.metrics.regularized_inverse(Y)
        cls.assertTrue(np.allclose(W, groundTruth))

    def test_sklearn_kernels(cls):
        """Sklearn kernels should work with regularized_inverse."""

        Y = np.arange(4).reshape((2, 2))

        for kernel_fn in sklearn_kernel_list:
            with cls.subTest(msg=f"Testing with {type(kernel_fn)}."):

                try:

                    W = gym_socks.kernel.metrics.regularized_inverse(
                        Y, Y, kernel_fn=kernel_fn
                    )

                except Exception as e:
                    cls.fail(f"Kernel {type(kernel_fn)} raised an exception.")
