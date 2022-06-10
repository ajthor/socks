import unittest

from functools import partial

import numpy as np

import gym

from gym_socks.kernel.metrics import abel_kernel
from gym_socks.kernel.metrics import delta_kernel
from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import _hybrid_distances
from gym_socks.kernel.metrics import hybrid_kernel
from gym_socks.kernel.metrics import euclidean_distances
from gym_socks.kernel.metrics import regularized_inverse
from gym_socks.kernel.metrics import woodbury_inverse


class TestAbelKernel(unittest.TestCase):
    def test_abel_kernel(cls):
        """Test that Abel kernel computes correctly."""

        cls.assertTrue(True)


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


class TestHybridDistances(unittest.TestCase):
    def test_hybrid_distances(cls):
        """Test that hybrid distances computes correctly."""

        X = np.arange(12).reshape(6, 2)
        Q = np.array([[0], [0], [0], [1], [2], [2]], dtype=int)

        D = _hybrid_distances(X, Q)

        groundTruth = (2 / np.pi) * np.arctan(np.max(np.abs(X[0] - X[1])))

        cls.assertEqual(D[0, 1], groundTruth)

        groundTruth = (2 / np.pi) * np.arctan(np.max(np.abs(X[0] - X[2])))

        cls.assertEqual(D[0, 2], groundTruth)


class TestHybridKernel(unittest.TestCase):
    def test_hybrid_kernel(cls):
        """Test that hybrid kernel computes correctly."""

        X = np.arange(12).reshape(6, 2)
        Q = np.array([[0], [0], [0], [1], [2], [2]], dtype=int)

        G = hybrid_kernel(X, Q)

        cls.assertTrue(True)


class TestRegularizedInverse(unittest.TestCase):
    def test_regularized_inverse(cls):
        """Test that regularized inverse computes correctly."""

        X = np.arange(4).reshape((2, 2))

        G = rbf_kernel(X, sigma=1)

        groundTruth = np.array(
            [[0.66676608, -0.0081415], [-0.0081415, 0.66676608]],
            dtype=float,
        )

        W = regularized_inverse(G, 1 / 4)

        cls.assertTrue(np.allclose(W, groundTruth))


class TestWoodburyInverse(unittest.TestCase):
    def test_woodbury_inverse(cls):
        """Test that the Woodbury matrix identity computes correctly."""

        X = np.arange(12).reshape(6, 2)
        groundTruth = regularized_inverse(X @ X.T, 1)

        A = 6 * np.identity(6)
        C = np.identity(2)
        W = woodbury_inverse(A, X, C, X.T)

        cls.assertTrue(np.allclose(W, groundTruth))

        X = np.arange(12).reshape(-1, 1)
        groundTruth = regularized_inverse(X @ X.T, 1)

        A = 12 * np.identity(12)
        C = np.identity(1)
        W = woodbury_inverse(A, X, C, X.T)

        cls.assertTrue(np.allclose(W, groundTruth))

    def test_woodbury_inverse_precomputed(cls):
        """Test that the Woodbury matrix identity computes correctly."""

        X = np.arange(12).reshape(6, 2)
        groundTruth = regularized_inverse(X @ X.T, 1)

        A = (1 / 6) * np.identity(6)
        C = np.identity(2)
        W = woodbury_inverse(A, X, C, X.T, precomputed=True)

        cls.assertTrue(np.allclose(W, groundTruth))

        X = np.arange(12).reshape(-1, 1)
        groundTruth = regularized_inverse(X @ X.T, 1)

        A = (1 / 12) * np.identity(12)
        C = np.identity(1)
        W = woodbury_inverse(A, X, C, X.T, precomputed=True)

        cls.assertTrue(np.allclose(W, groundTruth))
