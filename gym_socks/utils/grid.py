import numpy as np

from functools import reduce
from operator import mul

from gym_socks.envs.spaces import Box


def cartesian(*xi, out=None):
    """Create a grid of points from ranges.

    Args:
        xi: Ranges.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> import numpy as np
        >>> from gym_socks.utils.grid import cartesian
        >>> grid = cartesian(np.linspace(1, 3, 3), np.linspace(4, 6, 3))
        array([[1.0, 4.0],
               [1.0, 5.0],
               [1.0, 6.0],
               [2.0, 4.0],
               [2.0, 5.0],
               [2.0, 6.0],
               [3.0, 4.0],
               [3.0, 5.0],
               [3.0, 6.0]])

    """

    len_xi = len(xi)
    dtype = np.result_type(*xi)

    if out is None:
        out = np.empty([len(a) for a in xi] + [len_xi], dtype=dtype)

    for i, x in enumerate(np.ix_(*xi)):
        out[..., i] = x

    out = out.reshape(-1, len_xi)

    return out


def boxgrid(space: Box, resolution: tuple, out=None):
    r"""Returns a coarse grid from a bounded Box space.

    Constructs a coarse grid from a bounded box in :math:`\mathbb{R}^n`. A
    :py:class:`Box` is the Cartesian product of :math:`n` intervals, and we create a
    coarse grid of those intervals using a specified :py:obj:`resolution`.

    Args:
        space: The Box space to create a grid over. Must be bounded.
        resolution: The grid resolution. Can either be a single integer (applies to all
            dimensions) or a tuple of integers, which is of length :math:`n`.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> import numpy as np
        >>> from gym_socks.envs.spaces import Box
        >>> from gym_socks.utils.grid import boxgrid
        >>> space = Box(low=-1, high=1, shape=(2,), dtype=float)
        >>> boxgrid(space, (3, 5))
        array([[-1. , -1. ],
               [-1. , -0.5],
               [-1. ,  0. ],
               [-1. ,  0.5],
               [-1. ,  1. ],
               [ 0. , -1. ],
               [ 0. , -0.5],
               [ 0. ,  0. ],
               [ 0. ,  0.5],
               [ 0. ,  1. ],
               [ 1. , -1. ],
               [ 1. , -0.5],
               [ 1. ,  0. ],
               [ 1. ,  0.5],
               [ 1. ,  1. ]])

    """

    assert space.is_bounded(), "space must be bounded."

    if np.isscalar(resolution):
        xi = np.linspace(space.low, space.high, resolution, axis=-1, dtype=space.dtype)
        out = cartesian(*xi, out=out)

    else:
        assert (
            np.shape(resolution) == space.shape
        ), "resolution shape doesn't match space shape"

        xi = []
        for i, value in enumerate(resolution):
            xi.append(
                np.linspace(space.low[i], space.high[i], value, dtype=space.dtype)
            )

        out = cartesian(*xi, out=out)

    return out
