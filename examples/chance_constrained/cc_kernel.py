import numpy as np

from sklearn.metrics.pairwise import rbf_kernel


def cc_kernel(X: np.ndarray, Y: np.ndarray = None, gamma: float = None) -> np.ndarray:
    """CC kernel function."""

    if len(np.shape(X)) == 2:
        return rbf_kernel(X, Y, gamma=gamma)

    N = np.shape(X)[1]

    K = np.ones((N, N), dtype=np.float32)
    for i in range(N):
        K *= rbf_kernel(X[:, i, :], Y[:, i, :], gamma=gamma)

    # K /= N

    return K
