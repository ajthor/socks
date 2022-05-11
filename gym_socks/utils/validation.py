import numpy as np


def check_array(
    array: np.ndarray,
    dtype: np.dtype = np.float64,
    order=None,
    copy: bool = False,
    ensure_2d: bool = True,
    ensure_finite: bool = True,
):
    """Validate array.

    Performs checks to ensure that the array is valid.

    Note:

        This function is intended as a simple replacement for the
        :py:func:`sklearn.metrics.pairwise.check_array` function. Unlike
        :py:mod:`sklearn`, this function does not check sparse input data, and does not
        do sophisticated type checking or upcasting.

    Args:
        array: The array to be validated.
        dtype: The data type of the resulting array.
        order: The memory layout of the resulting array.
        copy: Whether to create a forced copy of ``array``.
        ensure_2d: Whether to raise an error if the array is not 2D.
        ensure_finite: Whether to raise an error if the array is not finite.

    Returns:
        The validated array.

    """

    array_orig = array

    # Convert array to numpy array.
    array = np.asarray(array, dtype=dtype, order=order)

    # Raise an error if the array is not 2D.
    if ensure_2d is True:

        if array.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead.\n"
                "Reshape the data using array.reshape(-1, 1) "
                "if the data has only a single dimension "
                "or array.reshape(1, -1) if there is only a single sample."
            )
        elif array.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead.\n"
                "Reshape the data using array.reshape(-1, 1) "
                "if the data has only a single dimension "
                "or array.reshape(1, -1) if there is only a single sample."
            )

    # Raise an error if the array elements are not finite.
    if ensure_finite is True:
        if array.dtype.kind in "fc":
            if not np.isfinite(array).all():
                raise ValueError("Input contains infinity or NaN.")

    # Create a copy if requested.
    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array


def check_gram_matrix(
    G: np.ndarray,
    ensure_square: bool = True,
    copy: bool = False,
):
    """Validate the Gram (kernel) matrix.

    Performs checks to ensure that the matrix is valid.

    Args:
        G: The Gram (kernel) matrix.
        ensure_square: Whether to raise an error if the kernel matrix is not square.
        copy: Whether to create a forced copy of ``G``.

    Returns:
        The validated matrix.

    """

    G = check_array(G, copy=copy)

    if ensure_square:
        m, n = np.shape(G)
        if G.ndim != 2 and m != n:
            raise ValueError("Gram matrix must be square.")

    return G
