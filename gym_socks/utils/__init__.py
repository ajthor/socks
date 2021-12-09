__all__ = ["batch", "grid", "logging"]

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


def indicator_fn(points: np.ndarray, space: any) -> np.ndarray:
    """Lightweight indicator for gym.spaces.Box sets.

    Provides a lightweight proxy for the `contains` function in `gym.spaces.Box`. The
    original function does type checking, casting, and shape checking, which is slow.

    Args:
        points: Points to evaluate the indicator at.
        space: Space over which the indicator function is defined. Should be either a
            `gym.spaces.Box` or a function which returns 0 or 1 depending on whether the
            points are inside or outside the space.

    Returns:
        Boolean labels of whether the points are inside or outside the space.

    """

    _l = space.low
    _h = space.high

    return np.array(
        np.all(points >= _l, axis=1) & np.all(points <= _h, axis=1), dtype=bool
    )


def save_mat_file(filename: str, data: dict):
    from scipy.io import savemat

    with open(filename, "wb") as f:
        savemat(f, data)
