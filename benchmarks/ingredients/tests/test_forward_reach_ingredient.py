from typing import ForwardRef
import unittest
from unittest import mock
from unittest.mock import patch

import gym

from gym_socks.envs import NDIntegratorEnv

import numpy as np

from examples.ingredients.forward_reach_ingredient import forward_reach_ingredient
from examples.ingredients.forward_reach_ingredient import compute_test_point_ranges


class TestForwardReachIngredient(unittest.TestCase):
    def test_forward_reach_config(cls):
        """Test the ingredient config."""
        ingredient = forward_reach_ingredient
        cls.assertEqual(len(ingredient.configurations), 1)


class TestComputeTestPointRanges(unittest.TestCase):
    def test_compute_test_point_ranges(cls):
        """Test that compute_test_point_ranges gives correct ranges."""

        test_points = {
            "lower_bound": -1,
            "upper_bound": 1,
            "grid_resolution": 25,
        }

        result = compute_test_point_ranges(shape=(2,), test_points=test_points)

        groundTruth = [np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)]

        cls.assertTrue(np.array_equal(result, groundTruth))
