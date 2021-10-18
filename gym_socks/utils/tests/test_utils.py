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

        interval = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # One point inside.
        points = [[0.1, 0.1]]
        groundTruth = np.array([True], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, set=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

        # Multiple points inside.
        points = [[0.1, 0.1], [0.2, -0.2]]
        groundTruth = np.array([True, True], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, set=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

        # One point in, one point out.
        points = [[0.1, 0.1], [1.2, -0.2]]
        groundTruth = np.array([True, False], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, set=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

    def test_indicator_function_on_function(cls):
        """Test indicator function on function."""

        def interval(points):
            return np.array(np.all(np.abs(points) <= 1, axis=1), dtype=bool)

        # One point inside.
        points = [[0.1, 0.1]]
        groundTruth = np.array([True], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, set=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

        # Multiple points inside.
        points = [[0.1, 0.1], [0.2, -0.2]]
        groundTruth = np.array([True, True], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, set=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))

        # One point in, one point out.
        points = [[0.1, 0.1], [1.2, -0.2]]
        groundTruth = np.array([True, False], dtype=bool)
        indicator_result = gym_socks.utils.indicator_fn(points=points, set=interval)
        cls.assertTrue(np.all(np.equal(indicator_result, groundTruth)))


class TestGenerateBatches(unittest.TestCase):
    """Test generate_batches."""

    def test_generate_batches(cls):
        """Test generate batches."""
        idx = np.arange(10)

        batches = gym_socks.utils.generate_batches(num_elements=10, batch_size=2)
        for i, batch in enumerate(batches):
            cls.assertTrue(len(idx[batch]) == 2)
            cls.assertTrue(np.all(np.equal(idx[batch], idx[i * 2 : i * 2 + 2])))

        batches = gym_socks.utils.generate_batches(num_elements=10, batch_size=5)
        for i, batch in enumerate(batches):
            cls.assertTrue(len(idx[batch]) == 5)
            cls.assertTrue(np.all(np.equal(idx[batch], idx[i * 5 : i * 5 + 5])))

    def test_batch_larger_than_set(cls):
        """Ensure batches past the maximum size work properly."""

        idx = np.arange(10)

        batches = gym_socks.utils.generate_batches(num_elements=14, batch_size=7)
        batches.send(None)
        batch = batches.send(None)
        cls.assertTrue(len(idx[batch]) == 3)
        cls.assertTrue(np.all(np.equal(idx[batch], idx[-3:])))

        batches = gym_socks.utils.generate_batches(num_elements=10, batch_size=14)
        batch = batches.send(None)
        cls.assertTrue(len(idx[batch]) == 10)
        cls.assertTrue(np.all(np.equal(idx[batch], idx[:])))
