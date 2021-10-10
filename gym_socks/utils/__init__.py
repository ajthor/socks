__all__ = ["logging"]

import gym

import numpy as np


def normalize(v):
    return v / np.sum(v, axis=0)


def _gym_space_contains(points, set):
    """Lightweight indicator for gym.spaces.Box sets."""

    _l = set.low
    _h = set.high

    return np.array(
        np.all(points >= _l, axis=1) & np.all(points <= _h, axis=1), dtype=bool
    )


def indicator_fn(points, set):
    """
    Indicator function for generic sets.

    If the set is not an instance of gym.spaces.Box, then 'set' should be a callable function.

    Any easy way to enable this is to create a wrapper. The function should return a 0
    or 1 if the points are outside or inside the set, respectively.
    """

    if isinstance(set, gym.spaces.Box):
        return _gym_space_contains(points, set)

    return np.array(set(points), dtype=bool)


def generate_batches(num_elements=1, batch_size=1):
    """
    Generate batches.
    """

    start = 0

    for _ in range(num_elements // batch_size):
        end = start + batch_size

        yield slice(start, end)

        start = end

    if start < num_elements:
        yield slice(start, num_elements)
