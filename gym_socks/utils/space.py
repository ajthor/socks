import gym

import numpy as np
from numpy.core.numeric import isscalar


def subspace(
    space: gym.spaces.Box, low, high, shape: tuple = None, seed: int = None
) -> gym.spaces.Box:

    # if shape is not None:
    #     shape = tuple(shape)

    #     # assert same as space or less than
    #     assert space.shape == shape, "shape must match space.shape"

    # elif not np.isscalar(low):
    #     shape = low.shape

    # elif not np.isscalar(high):
    #     shape = high.shape

    # if np.isscalar(low):
    #     low = np.full(shape, low, dtype=space.dtype)

    # if np.isscalar(high):
    #     high = np.full(shape, high, dtype=space.dtype)

    _space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=space.dtype)

    if seed is not None:
        _space.seed(seed=seed)

    return _space
