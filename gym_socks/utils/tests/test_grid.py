import unittest

from gym_socks.envs.spaces import Box
import gym_socks.utils

import numpy as np

from gym_socks.utils.grid import boxgrid
from gym_socks.utils.grid import cartesian


class TestGrid(unittest.TestCase):
    def test_cartesian(cls):
        """Should generate all combinations of points (i.e. Cartesian product)."""
        result = cartesian(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))

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

    def test_boxgrid(cls):
        """Should generate proper grid from space."""

        sample_space = Box(low=-1, high=1, shape=(2,), dtype=float)

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

        result = boxgrid(space=sample_space, resolution=[3, 3])
        cls.assertTrue(np.array_equiv(result, groundTruth))

        result = boxgrid(space=sample_space, resolution=3)
        cls.assertTrue(np.array_equiv(result, groundTruth))
