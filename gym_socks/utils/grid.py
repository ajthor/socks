import gym

import numpy as np

from functools import reduce
from operator import mul


def make_grid_from_ranges(xi: list) -> list:
    """Create a grid of points from a list of ranges.

    Args:
        xi: List of ranges.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> import numpy as np
        >>> from gym_socks.utils.grid import make_grid_from_ranges
        >>> grid = make_grid_from_ranges([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)])

    """

    len_xi = len(xi)

    dtype = np.result_type(*xi)
    result = np.empty([len(a) for a in xi] + [len_xi], dtype=dtype)

    for i, x in enumerate(np.ix_(*xi)):
        result[..., i] = x

    return result.reshape(-1, len_xi)


def make_grid_from_space(sample_space: gym.spaces.Box, resolution: int) -> list:
    """Create a grid from a bounded sample space.

    Args:
        sample_space: Bounded sample space.
        resolution: The gid resolution, specified either as an integer for all
            dimensions or as a list of integers, one per dimension.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> from gym.spaces import Box
        >>> import numpy as np
        >>> from gym_socks.utils.grid import make_grid_from_space
        >>> sample_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        >>> grid = make_grid_from_space(sample_space, [3, 2])

    """

    assert sample_space.is_bounded(), "space must be bounded."

    low = sample_space.low
    high = sample_space.high

    if np.isscalar(resolution):
        xi = np.linspace(low, high, resolution, axis=-1)
        grid = make_grid_from_ranges(xi)

    else:
        assert (
            np.shape(resolution) == sample_space.shape
        ), "resolution.shape doesn't match sample_space.shape"
        xi = []
        for i, value in enumerate(resolution):
            xi.append(np.linspace(low[i], high[i], value))

        grid = make_grid_from_ranges(xi)

    return grid


def grid_size_from_ranges(xi: list) -> int:
    """Returns the size of a grid sample based on ranges.

    Args:
        xi: List of ranges.

    Returns:
        The product of the lengths of the ranges.

    """

    return reduce(mul, map(len, xi), 1)


def grid_size_from_space(sample_space: gym.spaces.Box, resolution: int) -> int:
    """Returns the size of a grid sample based on sample space and resolution.

    Args:
        sample_space: The sample space.
        resolution: The gid resolution, specified either as an integer for all
            dimensions or as a list of integers, one per dimension.

    Returns:
        The product of the lengths of the ranges.

    """

    assert sample_space.is_bounded(), "space must be bounded."

    low = sample_space.low
    high = sample_space.high

    if np.isscalar(resolution):
        resolution = np.full(shape=sample_space.shape, fill_value=resolution)

    else:
        assert (
            np.shape(resolution) == sample_space.shape
        ), "resolution.shape doesn't match sample_space.shape"

    return reduce(mul, resolution, 1)
