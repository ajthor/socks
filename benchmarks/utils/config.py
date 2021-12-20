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

    if value is str:
        if value == "inf":
            value = np.inf
        elif value == "-inf":
            value = -np.inf

    elif value is list:
        for i, item in enumerate(value):
            if item == "inf":
                value[i] = np.inf
            elif item == "-inf":
                value[i] = -np.inf

    result = np.asarray(value)

    if np.isscalar(value):
        result = np.full(shape, value, dtype=dtype)

    result = np.asarray(result, dtype=dtype)

    return result
