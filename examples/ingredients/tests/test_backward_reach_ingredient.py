import unittest
from unittest import mock
from unittest.mock import patch

import gym

from gym_socks.envs import NDIntegratorEnv

import numpy as np

from examples.ingredients.backward_reach_ingredient import backward_reach_ingredient
from examples.ingredients.backward_reach_ingredient import generate_tube
from examples.ingredients.backward_reach_ingredient import compute_test_point_ranges


class TestBackwardReachIngredient(unittest.TestCase):
    def test_backward_reach_config(cls):
        """Test the ingredient config."""
        ingredient = backward_reach_ingredient
        cls.assertEqual(len(ingredient.configurations), 1)


class TestGenerateTube(unittest.TestCase):
    def test_generate_tube_scalar_bounds(cls):
        """Test that generate_tube creates a list of spaces when given scalar bounds."""

        env = NDIntegratorEnv(2)
        time_horizon = 5

        tube_cfg = {
            "lower_bound": -1,
            "upper_bound": 1,
        }

        tube = generate_tube(
            time_horizon=time_horizon, shape=env.state_space.shape, bounds=tube_cfg
        )

        cls.assertEqual(len(tube), time_horizon)

        for item in tube:

            cls.assertIsInstance(item, gym.spaces.Box)
            cls.assertTrue(np.array_equal(item.low, [-1, -1]))
            cls.assertTrue(np.array_equal(item.high, [1, 1]))
            cls.assertEqual(item.shape, (2,))

    def test_generate_tube_list_bounds(cls):
        """Test that generate_tube creates a list of spaces when given list bounds."""

        env = NDIntegratorEnv(2)
        time_horizon = 5

        lower_bound = (-np.arange(time_horizon)).tolist()
        upper_bound = np.arange(time_horizon).tolist()

        tube_cfg = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

        tube = generate_tube(
            time_horizon=time_horizon, shape=env.state_space.shape, bounds=tube_cfg
        )

        cls.assertEqual(len(tube), time_horizon)

        for i, item in enumerate(tube):

            cls.assertIsInstance(item, gym.spaces.Box)
            cls.assertTrue(
                np.array_equal(
                    item.low, np.full(item.shape, lower_bound[i], dtype=np.float32)
                )
            )
            cls.assertTrue(
                np.array_equal(
                    item.high, np.full(item.shape, upper_bound[i], dtype=np.float32)
                )
            )
            cls.assertEqual(item.shape, (2,))

    def test_generate_tube_matrix_bounds(cls):
        """Test that generate_tube creates a list of spaces when given matrix bounds."""

        env = NDIntegratorEnv(2)
        time_horizon = 5

        lower_bound = np.full((time_horizon, 2), -1, dtype=np.float32).tolist()
        upper_bound = np.full((time_horizon, 2), 1, dtype=np.float32).tolist()

        tube_cfg = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

        tube = generate_tube(
            time_horizon=time_horizon, shape=env.state_space.shape, bounds=tube_cfg
        )

        cls.assertEqual(len(tube), time_horizon)

        for i, item in enumerate(tube):

            cls.assertIsInstance(item, gym.spaces.Box)
            cls.assertTrue(np.array_equal(item.low, [-1, -1]))
            cls.assertTrue(np.array_equal(item.high, [1, 1]))
            cls.assertEqual(item.shape, (2,))


class TestComputeTestPointRanges(unittest.TestCase):
    def test_compute_test_point_ranges(cls):
        """Test that compute_test_point_ranges gives correct ranges."""

        env = NDIntegratorEnv(2)

        test_points = {
            "lower_bound": -1,
            "upper_bound": 1,
            "grid_resolution": 25,
        }

        result = compute_test_point_ranges(env=env, test_points=test_points)

        groundTruth = [np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)]

        cls.assertTrue(np.array_equal(result, groundTruth))
