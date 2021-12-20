import unittest
from unittest import mock
from unittest.mock import patch

import gym

import numpy as np

from examples.ingredients.common import parse_array
from examples.ingredients.common import box_factory
from examples.ingredients.common import grid_ranges
from examples.ingredients.common import grid_sample_size


class TestParseConfigFunctions(unittest.TestCase):
    def test_parse_array_scalar(cls):
        """Test the parse_array function for scalar inputs."""

        groundTruth = [5, 5]

        result = parse_array(value=5, shape=(2,), dtype=np.float32)
        cls.assertTrue(np.array_equal(result, groundTruth))
        cls.assertTrue(np.array_equal(result, np.array(groundTruth)))
        cls.assertTrue(np.array_equal(result, np.array(groundTruth, dtype=np.float32)))

        groundTruth = [[5, 5], [5, 5]]

        result = parse_array(value=5, shape=(2, 2), dtype=np.float32)
        cls.assertTrue(np.array_equal(result, groundTruth))
        cls.assertTrue(np.array_equal(result, np.array(groundTruth)))
        cls.assertTrue(np.array_equal(result, np.array(groundTruth, dtype=np.float32)))

    def test_parse_array_list(cls):
        """Test the parse_array function for list inputs."""

        result = parse_array(value=[1, 2], shape=(2,), dtype=np.float32)
        cls.assertTrue(np.array_equal(result, [1, 2]))
        cls.assertTrue(np.array_equal(result, np.array([1, 2])))
        cls.assertTrue(np.array_equal(result, np.array([1, 2], dtype=np.float32)))

        result = parse_array(value=[[1, 2], [3, 4]], shape=(2, 2), dtype=np.float32)
        cls.assertTrue(np.array_equal(result, [[1, 2], [3, 4]]))
        cls.assertTrue(np.array_equal(result, np.array([[1, 2], [3, 4]])))
        cls.assertTrue(
            np.array_equal(result, np.array([[1, 2], [3, 4]], dtype=np.float32))
        )


class TestBoxFactory(unittest.TestCase):
    def test_box_factory(cls):
        """Test box factory produces correct spaces."""

        result = box_factory(
            lower_bound=-1, upper_bound=1, shape=(2,), dtype=np.float32
        )

        cls.assertTrue(np.array_equal(result.low, [-1, -1]))
        cls.assertTrue(np.array_equal(result.high, [1, 1]))

        cls.assertEqual(result.shape, (2,))

        cls.assertIsInstance(result, gym.spaces.Box)

        result = box_factory(
            lower_bound=[-1, -2], upper_bound=[1, 2], shape=(2,), dtype=np.float32
        )

        cls.assertTrue(np.array_equal(result.low, [-1, -2]))
        cls.assertTrue(np.array_equal(result.high, [1, 2]))

        cls.assertEqual(result.shape, (2,))

        cls.assertIsInstance(result, gym.spaces.Box)


class TestGridRanges(unittest.TestCase):
    def test_grid_ranges(cls):
        """Test that grid_ranges produces the correct ranges."""

        space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        result = grid_ranges(space=space, grid_resolution=5)

        groundTruth = [np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)]

        cls.assertTrue(np.array_equal(result, groundTruth))

        groundTruth = np.array([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)])

        cls.assertTrue(np.array_equal(result, groundTruth))


class TestGridSampleSize(unittest.TestCase):
    def test_grid_sample_size(cls):
        """Test that grid_sample_size computes the correct sample size."""

        space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        result = grid_sample_size(space=space, grid_resolution=5)

        groundTruth = 25

        cls.assertEqual(result, groundTruth)
