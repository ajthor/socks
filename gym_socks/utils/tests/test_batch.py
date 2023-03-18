import unittest

from gym_socks.utils.batch import batch_generator

import numpy as np


class TestGenerateBatches(unittest.TestCase):
    """Test batch_generator."""

    def test_batch_generator(cls):
        """Test generate batches."""
        idx = np.arange(10)

        batches = batch_generator(s=idx, size=2)
        for i, batch in enumerate(batches):
            cls.assertTrue(len(idx[batch]) == 2)
            cls.assertTrue(np.all(np.equal(idx[batch], idx[i * 2 : i * 2 + 2])))

        batches = batch_generator(s=idx, size=5)
        for i, batch in enumerate(batches):
            cls.assertTrue(len(idx[batch]) == 5)
            cls.assertTrue(np.all(np.equal(idx[batch], idx[i * 5 : i * 5 + 5])))

    def test_batch_larger_than_set(cls):
        """Ensure batches past the maximum size work properly."""

        idx = np.arange(10)

        batches = batch_generator(s=idx, size=7)
        batches.send(None)
        batch = batches.send(None)
        cls.assertTrue(len(idx[batch]) == 3)
        cls.assertTrue(np.all(np.equal(idx[batch], idx[-3:])))

        batches = batch_generator(s=idx, size=14)
        batch = batches.send(None)
        cls.assertTrue(len(idx[batch]) == 10)
        cls.assertTrue(np.all(np.equal(idx[batch], idx[:])))
