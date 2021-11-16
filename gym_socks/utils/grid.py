import numpy as np

from itertools import islice
from functools import reduce
from operator import mul


def make_grid(xi):
    """Create a uniform grid from a list of ranges.

    Args:
        xi: List of ranges.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> import numpy as np
        >>> from gym_socks.utils.grid import make_grid
        >>> grid = uniform_grid([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)])

    """

    len_xi = len(xi)

    dtype = np.result_type(*xi)
    result = np.empty([len(a) for a in xi] + [len_xi], dtype=dtype)

    for i, x in enumerate(np.ix_(*xi)):
        result[..., i] = x

    return result.reshape(-1, len_xi)


def grid_size(xi):
    return reduce(mul, map(len, xi), 1)


def grid_ranges(low, high, shape, resolution, dtype):

    for item in resolution:
        np.linspace([-1, -1], [1, 1], 5, dtype=dtype, axis=-1)
    # return xi
    pass
