import unittest

import gym

from gym_socks.utils.batch import generate_batches

import numpy as np


class TestGenerateBatches(unittest.TestCase):
    """Test generate_batches."""

    def test_generate_batches(cls):
        """Test generate batches."""
        idx = np.arange(10)

        batches = generate_batches(num_elements=10, batch_size=2)
        for i, batch in enumerate(batches):
            cls.assertTrue(len(idx[batch]) == 2)
            cls.assertTrue(np.all(np.equal(idx[batch], idx[i * 2 : i * 2 + 2])))

        batches = generate_batches(num_elements=10, batch_size=5)
        for i, batch in enumerate(batches):
            cls.assertTrue(len(idx[batch]) == 5)
            cls.assertTrue(np.all(np.equal(idx[batch], idx[i * 5 : i * 5 + 5])))

    def test_batch_larger_than_set(cls):
        """Ensure batches past the maximum size work properly."""

        idx = np.arange(10)

        batches = generate_batches(num_elements=14, batch_size=7)
        batches.send(None)
        batch = batches.send(None)
        cls.assertTrue(len(idx[batch]) == 3)
        cls.assertTrue(np.all(np.equal(idx[batch], idx[-3:])))

        batches = generate_batches(num_elements=10, batch_size=14)
        batch = batches.send(None)
        cls.assertTrue(len(idx[batch]) == 10)
        cls.assertTrue(np.all(np.equal(idx[batch], idx[:])))
