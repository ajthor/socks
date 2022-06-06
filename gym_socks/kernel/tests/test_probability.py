import unittest

import numpy as np

from gym_socks.kernel.metrics import delta_kernel
from gym_socks.kernel.probability import probability_matrix


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
