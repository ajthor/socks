import numpy as np

from itertools import islice, product


def uniform_grid(xi):
    """Create a uniform grid from a list of ranges.

    Args:
        xi: List of ranges.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> import numpy as np
        >>> from gym_socks.envs.sample import uniform_grid
        >>> grid = uniform_grid([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)])

    """

    # return list(product(*xi))

    len_xi = len(xi)

    dtype = np.result_type(*xi)
    result = np.empty([len(a) for a in xi] + [len_xi], dtype=dtype)

    for i, x in enumerate(np.ix_(*xi)):
        result[..., i] = x

    return result.reshape(-1, len_xi)
