import numpy as np


def transpose_sample(sample):
    """Transpose the sample.

    By default, a sample should be a list of tuples of the form::

        S = [(x_1, y_1), ..., (x_n, y_n)]

    For most algorithms, we need to isolate the sample components (e.g. all x's).
    This function converts a sample from a list of tuples to a tuple of lists::

        S_T = ([x_1, ..., x_n], [y_1, ..., y_n])

    This can then be unpacked as: ``X, Y = S_T``

    Args:
        sample: list of tuples

    Returns:
        tuple of lists

    """

    return tuple(map(list, zip(*sample)))


def flatten_sample(sample):
    """Reshapes trajectory samples.

    Often, trajectory samples are organized such that the "trajectory" components are a
    2D array of points indexed by time. However, for kernel methods, we typically
    require that the trajectories be concatenated into a single vector (1D array)::

        [[x1], [x2], ..., [xn]] -> [x1, x2, ..., xn]

    This function converts the sample so that the trajectories are 1D arrays.

    Args:
        sample: list of tuples

    Returns:
        List of tuples, where the components of the tuples are flattened.

    """
    sample_size = len(sample)
    _S = transpose_sample(sample)

    return zip(*[np.reshape(item, (sample_size, -1)) for item in _S])
