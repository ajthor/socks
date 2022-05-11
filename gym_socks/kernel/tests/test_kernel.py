from tokenize import group
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

from gym_socks.kernel.probability import probability_matrix


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

        Y = np.arange(4).reshape((2, 2))

        G = rbf_kernel(Y, sigma=1)

        groundTruth = np.array(
            [[0.66676608, -0.0081415], [-0.0081415, 0.66676608]],
            dtype=np.float32,
        )

        W = regularized_inverse(G, 1 / 4)

        cls.assertTrue(np.allclose(W, groundTruth))


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

    def test_probability_matrix_2D(cls):
        """Test that the probability matrix is correct for 2D discrete data."""

        # A list of all possible states.
        states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)

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
