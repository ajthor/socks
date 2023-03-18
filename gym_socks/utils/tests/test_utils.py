import unittest

import gym

import gym_socks.utils

import numpy as np


class TestNormalize(unittest.TestCase):
    def test_normalize(cls):
        """Test normalize function."""

        # Single point.
        points = [1.0, 2.0]
        groundTruth = [0.33333333, 0.66666667]
        normalize_result = gym_socks.utils.normalize(points)
        cls.assertTrue(np.allclose(normalize_result, groundTruth))

        # Multiple points.
        points = [[1.0, 2.0], [0.5, 1.0]]
        groundTruth = [[0.66666667, 0.66666667], [0.33333333, 0.33333333]]
        normalize_result = gym_socks.utils.normalize(points)
        cls.assertTrue(np.allclose(normalize_result, groundTruth))

        points = [[1.0, 0.5], [2.0, 1.0]]
        groundTruth = [[0.33333333, 0.33333333], [0.66666667, 0.66666667]]
        normalize_result = gym_socks.utils.normalize(points)
        cls.assertTrue(np.allclose(normalize_result, groundTruth))


class TestIndicatorFunction(unittest.TestCase):
    def test_indicator_function_on_box(cls):
        """Test indicator function on box."""

        interval = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)

        # One point inside.
        points = [[0.1, 0.1]]
        groundTruth = np.array([True], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, space=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

        # Multiple points inside.
        points = [[0.1, 0.1], [0.2, -0.2]]
        groundTruth = np.array([True, True], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, space=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

        # One point in, one point out.
        points = [[0.1, 0.1], [1.2, -0.2]]
        groundTruth = np.array([True, False], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, space=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))
