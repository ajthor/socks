import numpy as np
from sklearn.utils import check_array


def check_gram_matrix(
    G: np.ndarray,
    ensure_square: bool = True,
    copy: bool = False,
):
    """Validate the Gram matrix.

    Args:
        G: The Gram (kernel) matrix.
        ensure_square: Whether to raise an error if the kernel matrix is not square.
        copy: Whether to create a forced copy of `G`.

    Returns:
        The validated matrix.

    """

    G = check_array(G, copy=copy)

    if ensure_square:
        m, n = np.shape(G)
        if G.ndim != 2 and m != n:
            raise ValueError("Gram matrix must be square.")

    return G
