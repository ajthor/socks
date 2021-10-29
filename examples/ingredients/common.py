"""Common utility functions for ingredients."""

import gym

import numpy as np


def assert_config_has_key(_config, key):
    # if key not in _config:
    #     raise KeyError(f"Configuration {_config} missing entry: '{key}'.")
    msg = f"Configuration {_config} missing entry: '{key}'."
    assert key in _config, msg


def parse_array(value: float or list, shape: tuple, dtype: type) -> list:
    """Utility function for parsing configuration variables.

    Parses values passed as either a scalar or list into an array of type `dtype`.

    Args:
        value: The value input to the config.
        shape: The shape of the resulting array.
        dtype: The type to cast to.

    Returns:
        An array of a particular length.

    """

    result = value

    if np.isscalar(value):
        result = np.full(shape, value, dtype=dtype)

    for i, item in enumerate(result):
        if item == "inf":
            result[i] = np.inf
        elif item == "-inf":
            result[i] = -np.inf

    result = np.asarray(result, dtype=dtype)

    return result


def box_factory(lower_bound, upper_bound, shape: tuple, dtype: type) -> gym.spaces.Box:
    """Box space factory.

    Creates a `gym.spaces.Box` to be used by sampling functions.

    """

    lower_bound = parse_array(lower_bound, shape=shape, dtype=dtype)
    upper_bound = parse_array(upper_bound, shape=shape, dtype=dtype)

    _space = gym.spaces.Box(
        low=lower_bound,
        high=upper_bound,
        shape=shape,
        dtype=dtype,
    )

    return _space


def grid_ranges(
    space: gym.Space,
    grid_resolution: list[int] or int,
) -> list:
    """Compute grid ranges."""

    lower_bound = space.low
    upper_bound = space.high

    grid_resolution = parse_array(grid_resolution, shape=space.shape, dtype=int)

    xi = []
    for i in range(space.shape[0]):
        points = np.linspace(lower_bound[i], upper_bound[i], grid_resolution[i])
        xi.append(points.tolist())

    return xi


def grid_sample_size(
    space: gym.Space,
    grid_resolution: list[int] or int,
) -> int:
    """Compute grid sample size."""
    grid_resolution = parse_array(grid_resolution, shape=space.shape, dtype=int)
    return int(np.product(grid_resolution))
