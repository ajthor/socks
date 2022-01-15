import unittest

import gym

import gym_socks.utils

import numpy as np

from gym_socks.utils.grid import make_grid_from_ranges
from gym_socks.utils.grid import make_grid_from_space
from gym_socks.utils.grid import grid_size_from_ranges
from gym_socks.utils.grid import grid_size_from_space


class TestGrid(unittest.TestCase):
    def test_grid_from_ranges(cls):
        """Should generate proper grid from ranges."""

        xi = [np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)]
        result = make_grid_from_ranges(xi)

        groundTruth = [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]

        cls.assertTrue(np.array_equiv(result, groundTruth))

    def test_grid_from_space(cls):
        """Should generate proper grid from space."""

        sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        groundTruth = [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]

        result = make_grid_from_space(sample_space=sample_space, resolution=3)
        cls.assertTrue(np.array_equiv(result, groundTruth))

        result = make_grid_from_space(sample_space=sample_space, resolution=[3, 3])
        cls.assertTrue(np.array_equiv(result, groundTruth))

    def test_grid_size_from_space(cls):
        """Should generate proper grid size from ranges."""

        xi = [np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)]
        cls.assertEqual(grid_size_from_ranges(xi), 9)

    def test_grid_size_from_space(cls):
        """Should generate proper grid size from space."""

        sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        cls.assertEqual(
            grid_size_from_space(sample_space=sample_space, resolution=3), 9
        )
        cls.assertEqual(
            grid_size_from_space(sample_space=sample_space, resolution=[3, 3]), 9
        )
