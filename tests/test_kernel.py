import unittest

from functools import partial

import gym

import kernel_basic.kernel

import numpy as np


class TestEuclideanDistance(unittest.TestCase):
    def test_euclidean_distance(cls):
        """
        Test that euclidean distance computes correctly.
        """

        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[2, 2], [3, 3]])

        distance = kernel_basic.kernel.euclidean_distance(X, Y, squared=True)

        groundTruth = np.array([[2, 8], [2, 8]], dtype=np.float32)

        # print(distance)
        # print(groundTruth)

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

        K = kernel_basic.kernel.rbf_kernel(X, Y, sigma=1)

        groundTruth = np.array(
            [
                [0.367879441171442, 0.018315638888734],
                [0.367879441171442, 0.018315638888734],
            ],
            dtype=np.float32,
        )

        # print(K)
        # print(groundTruth)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )

        K = kernel_basic.kernel.rbf_kernel(Y, Y, sigma=1)

        groundTruth = np.array(
            [
                [
                    [1.000000000000000, 0.367879441171442],
                    [0.367879441171442, 1.000000000000000],
                ],
            ],
            dtype=np.float32,
        )

        # print(K)
        # print(groundTruth)

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
        Y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        K = kernel_basic.kernel.rbf_kernel(X, Y, sigma=1)

        groundTruth = np.array(
            [
                [
                    [1.0, 0.367879441171442, 0.018315638888734],
                    [1.0, 0.367879441171442, 0.018315638888734],
                ],
            ],
            dtype=np.float32,
        )

        # print(K)
        # print(groundTruth)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(K, groundTruth),
            "Kernel matrix should match known ground truth.",
        )

    def test_regularized_inverse(cls):
        """
        Test that regularized inverse computes correctly.
        """

        Y = np.array([[2, 2], [3, 3]])

        kernel = partial(kernel_basic.kernel.rbf_kernel, sigma=1)

        W = kernel_basic.kernel.regularized_inverse(Y, Y, kernel=kernel)

        groundTruth = np.array(
            [
                [
                    [0.709332306019573, -0.173965848228887],
                    [-0.173965848228887, 0.709332306019573],
                ],
            ],
            dtype=np.float32,
        )

        # print(W)
        # print(groundTruth)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(W, groundTruth),
            "Regularized inverse should match known ground truth.",
        )
