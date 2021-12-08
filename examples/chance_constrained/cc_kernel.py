import numpy as np

from sklearn.metrics.pairwise import rbf_kernel


def normalize_data(X):
    if X is not None:
        return X / np.sum(X, axis=1)

    return None


def cc_kernel(X: np.ndarray, Y: np.ndarray = None, gamma: float = None) -> np.ndarray:
    """CC kernel function."""

    print(f"kernel shape: {np.shape(X)}, len: {len(np.shape(X))}")
    if len(np.shape(X)) == 2:
        return rbf_kernel(X, Y, gamma=gamma)

    M = np.shape(X)[0]
    N = np.shape(X)[1]

    print(f"kernel X shape: {np.shape(X)}")
    print(f"kernel Y shape: {np.shape(Y)}")

    K = np.zeros((M, M), dtype=np.float32)
    for i in range(N):
        if Y is None:
            K += rbf_kernel(X[:, i, :], gamma=gamma)
        else:
            K += rbf_kernel(X[:, i, :], Y[:, i, :], gamma=gamma)

    # K /= N

    return K
