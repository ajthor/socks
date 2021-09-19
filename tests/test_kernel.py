import unittest

from functools import partial

import gym

import gym_socks.kernel.metrics

import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel

sklearn_kernel_list = [linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel]


class TestEuclideanDistance(unittest.TestCase):
    def test_euclidean_distance(cls):
        """
        Test that euclidean distance computes correctly.
        """

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[2, 2], [3, 3]])

        distance = gym_socks.kernel.metrics.euclidean_distance(X, Y, squared=True)

        groundTruth = np.array([[2, 8], [2, 8]], dtype=np.float32)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(distance, groundTruth),
            "Pairwise Euclidean distance matrix should match known ground truth.",
        )


class TestRBFKernel(unittest.TestCase):
    def test_rbf_kernel_same_size(cls):
        """
        Test that RBF kernel computes correctly with same size inputs.
        """

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[2, 2], [3, 3]])

        K = gym_socks.kernel.metrics.rbf_kernel(X, Y, sigma=1)

        groundTruth = np.array(
            [
                [0.367879441171442, 0.018315638888734],
                [0.367879441171442, 0.018315638888734],
            ],
            dtype=np.float32,
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )

        K = gym_socks.kernel.metrics.rbf_kernel(Y, Y, sigma=1)

        groundTruth = np.array(
            [
                [
                    [1.000000000000000, 0.367879441171442],
                    [0.367879441171442, 1.000000000000000],
                ],
            ],
            dtype=np.float32,
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )

    def test_rbf_kernel_different_size(cls):
        """
        Test that RBF kernel computes correctly with different-sized inputs.
        """

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[1, 1], [2, 2], [3, 3]])

        K = gym_socks.kernel.metrics.rbf_kernel(X, Y, sigma=1)

        groundTruth = np.array(
            [
                [
                    [1.0, 0.367879441171442, 0.018315638888734],
                    [1.0, 0.367879441171442, 0.018315638888734],
                ],
            ],
            dtype=np.float32,
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        K = gym_socks.kernel.metrics.rbf_kernel(X, Y, sigma=1)

        groundTruth = np.array(
            [
                [
                    [1.0, 0.367879441171442, 0.018315638888734],
                    [1.0, 0.367879441171442, 0.018315638888734],
                ],
            ],
            dtype=np.float32,
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )

    def test_close_to_sklearn(cls):
        """
        Assert that result is close to sklearn.
        """

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[2, 2], [3, 3]])

        K = gym_socks.kernel.metrics.rbf_kernel(X, Y, sigma=1)

        groundTruth = rbf_kernel(X, Y, gamma=1 / 2)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )


class TestRegularizedInverse(unittest.TestCase):
    def test_regularized_inverse(cls):
        """
        Test that regularized inverse computes correctly.
        """

        Y = np.array([[2, 2], [3, 3]])

        kernel_fn = partial(gym_socks.kernel.metrics.rbf_kernel, sigma=1)

        W = gym_socks.kernel.metrics.regularized_inverse(Y, Y, kernel_fn=kernel_fn)

        groundTruth = np.array(
            [
                [
                    [0.709332306019573, -0.173965848228887],
                    [-0.173965848228887, 0.709332306019573],
                ],
            ],
            dtype=np.float32,
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(W, groundTruth),
            "Regularized inverse should match known ground truth.",
        )

    def test_should_complain(cls):
        """
        Assert that the function should complain if we pass improper values.
        """

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        kernel_fn = partial(gym_socks.kernel.metrics.rbf_kernel, sigma=1)

        with cls.assertRaises(AssertionError) as exception_context:
            W = gym_socks.kernel.metrics.regularized_inverse(X, Y, kernel_fn=kernel_fn)

    def test_sklearn_kernels(cls):
        """
        Assert that sklearn kernels should work with regularized_inverse.
        """

        Y = np.array([[2, 2], [3, 3]])

        for kernel_fn in sklearn_kernel_list:
            with cls.subTest(msg=f"Testing with {type(kernel_fn)}."):

                W = gym_socks.kernel.metrics.regularized_inverse(
                    Y, Y, kernel_fn=kernel_fn
                )
