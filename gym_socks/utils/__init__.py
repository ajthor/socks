__all__ = ["logging", "normalize", "indicator_fn", "generate_batches"]

from typing import Generator
import gym

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize.

    Small utility function for normalizing a matrix or a vector. Divides the matrix
    (vector) by the sum of the matrix rows (vector). Used primarily by the stochastic
    reachability algorithms.

    Args:
        v: Matrix or vector to normalize.

    Returns:
        Normalized matrix or vector.

    """
    return v / np.sum(v, axis=0)


def _gym_space_contains(points: np.ndarray, set: gym.spaces.Box) -> np.ndarray:
    """Lightweight indicator for gym.spaces.Box sets.

    Provides a lightweight proxy for the `contains` function in `gym.spaces.Box`. The
    original function does type checking, casting, and shape checking, which is slow.

    Args:
        points: Points to evaluate the indicator at.
        set: Set over which the indicator function is defined. Should be either a
            `gym.spaces.Box` or a function which returns 0 or 1 depending on whether the
            points are inside or outside the set.

    Returns:
        Boolean labels of whether the points are inside or outside the set.

    """

    _l = set.low
    _h = set.high

    return np.array(
        np.all(points >= _l, axis=1) & np.all(points <= _h, axis=1), dtype=bool
    )


def indicator_fn(points: np.ndarray, set: any) -> np.ndarray:
    """Indicator function for generic sets.

    If the set is not an instance of `gym.spaces.Box`, then `set` should be a callable
    function. Any easy way to enable this is to create a wrapper. The function should
    return a 0 or 1 if the points are outside or inside the set, respectively.

    Args:
        points: Points to evaluate the indicator at.
        set: Set over which the indicator function is defined. Should be either a
            `gym.spaces.Box` or a function which returns 0 or 1 depending on whether the
            points are inside or outside the set.

    Returns:
        Boolean labels of whether the points are inside or outside the set.

    """

    if isinstance(set, gym.spaces.Box):
        return _gym_space_contains(points, set)

    return np.array(set(points), dtype=bool)


def generate_batches(num_elements, batch_size) -> Generator[slice, None, None]:
    """Generate batches.

    Batch generation function to split a list into smaller batches (slices). Generates
    `slice` objects, which can be iterated over in a for loop.

    Args:
        num_elements: Length of the list.
        batch_size: Maximum size of the batches. If the last batch is smaller than
            `batch_size`, then the final batch will be the length of the remaining
            elements.

    Yields:
        A slice object.

    """

    start = 0

    for _ in range(num_elements // batch_size):
        end = start + batch_size

        yield slice(start, end)

        start = end

    if start < num_elements:
        yield slice(start, num_elements)


def save_mat_file(filename: str, data: dict):
    from scipy.io import savemat

    with open(filename, "wb") as f:
        savemat(f, data)
