import unittest

import numpy as np

from gym_socks.envs.spaces import Space
from gym_socks.envs.spaces import Box


class TestSpace(unittest.TestCase):
    def test_shape_is_not_none(self):
        with self.assertRaises(ValueError):
            Space()
        with self.assertRaises(ValueError):
            Space(shape=None)

    def test_dtype_is_not_none(self):
        with self.assertRaises(ValueError):
            Space(shape=(1,))
        with self.assertRaises(ValueError):
            Space(shape=(1,), dtype=None)

    def test_dtype_is_valid(self):
        with self.assertRaises(ValueError):
            Space(shape=(1,), dtype="invalid")

    def test_seed_is_integer_or_none(self):
        with self.assertRaises(ValueError):
            Space(shape=(1,), dtype=float, seed="invalid")
        with self.assertRaises(ValueError):
            Space(shape=(1,), dtype=float, seed=1.0)


# test the Box class
class TestBox(unittest.TestCase):
    def test_low_is_valid(self):
        with self.assertRaises(ValueError):
            Box(low="invalid", high=1)
        with self.assertRaises(ValueError):
            Box(low=[1, 2], high=1)
        with self.assertRaises(ValueError):
            Box(low=(1, 2), high=1)
        with self.assertRaises(ValueError):
            Box(low=0, high=-1)

    def test_high_is_valid(self):
        with self.assertRaises(ValueError):
            Box(low=0, high="invalid")
        with self.assertRaises(ValueError):
            Box(low=0, high=[1, 2])
        with self.assertRaises(ValueError):
            Box(low=0, high=(1, 2))
        with self.assertRaises(ValueError):
            Box(low=0, high=-1)

    def test_low_and_high_are_same_shape(self):
        with self.assertRaises(ValueError):
            Box(low=[1, 2], high=[1])
        with self.assertRaises(ValueError):
            Box(low=(1, 2), high=(1,))

    def test_contains_is_valid(self):
        box = Box(low=0, high=1)
        self.assertTrue(box.contains(0))
        self.assertTrue(box.contains(0.5))
        self.assertTrue(box.contains(1))
        self.assertFalse(box.contains(-1))
        self.assertFalse(box.contains(2))

        box = Box(low=[0, 0], high=[1, 1])
        self.assertTrue(box.contains([0, 0]))
        self.assertTrue(box.contains([0.5, 0.5]))
        self.assertTrue(box.contains([1, 1]))
        self.assertFalse(box.contains([-1, 0]))
        self.assertFalse(box.contains([2, 0]))
        self.assertFalse(box.contains([0, -1]))
        self.assertFalse(box.contains([0, 2]))

        box = Box(low=(0, 0), high=(1, 1))
        self.assertTrue(box.contains((0, 0)))
        self.assertTrue(box.contains((0.5, 0.5)))
        self.assertTrue(box.contains((1, 1)))
        self.assertFalse(box.contains((-1, 0)))
        self.assertFalse(box.contains((2, 0)))
        self.assertFalse(box.contains((0, -1)))
        self.assertFalse(box.contains((0, 2)))
