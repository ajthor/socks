import unittest

from functools import partial

import numpy as np

from scipy.linalg import cholesky

from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.kernel.update import add_sample
from gym_socks.kernel.update import remove_first_sample
from gym_socks.kernel.update import remove_last_sample
from gym_socks.kernel.update import remove_regularization
from gym_socks.kernel.update import cho_add_rc
from gym_socks.kernel.update import cho_del_rc


class TestMatrixInverseUpdate(unittest.TestCase):
    def test_add_sample(cls):
        """Matrix inverse should update correctly when a sample is added."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        # Full sample of observations.
        Z = np.arange(8).reshape(4, 2)
        # Extract reduced set.
        X = Z[:-1]
        y = Z[-1].reshape(1, -1)

        # Compute the regularized inverse of the reduced set.
        W_reduced = np.linalg.inv(
            kernel_fn(X) + regularization_param * np.identity(len(X))
        )

        # Compute the regularized inverse of the full set.
        groundTruth = np.linalg.inv(
            kernel_fn(Z) + regularization_param * np.identity(len(Z))
        )

        # Compute the modified inverse after adding a sample.
        W = add_sample(W_reduced, X, y, kernel_fn, regularization_param)

        cls.assertTrue(np.allclose(W, groundTruth))

    def test_remove_first_sample(cls):
        """Matrix inverse should update correctly when the first sample is removed."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        # Full sample of observations.
        Z = np.arange(8).reshape(4, 2)
        # Extract reduced set.
        X = Z[1:]

        # Compute the regularized inverse of the reduced set.
        groundTruth = np.linalg.inv(
            kernel_fn(X) + regularization_param * np.identity(len(X))
        )

        # Compute the regularized inverse of the full set.
        W_full = np.linalg.inv(
            kernel_fn(Z) + regularization_param * np.identity(len(Z))
        )

        # Compute the modified inverse after removing a sample.
        W = remove_first_sample(W_full)

        cls.assertTrue(np.allclose(W, groundTruth))

    def test_remove_last_sample(cls):
        """Matrix inverse should update correctly when the last sample is removed."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        # Full sample of observations.
        Z = np.arange(8).reshape(4, 2)
        # Extract reduced set.
        X = Z[:-1]

        # Compute the regularized inverse of the reduced set.
        groundTruth = np.linalg.inv(
            kernel_fn(X) + regularization_param * np.identity(len(X))
        )

        # Compute the regularized inverse of the full set.
        W_full = np.linalg.inv(
            kernel_fn(Z) + regularization_param * np.identity(len(Z))
        )

        # Compute the modified inverse after removing a sample.
        W = remove_last_sample(W_full)

        cls.assertTrue(np.allclose(W, groundTruth))

    def test_remove_regularization(cls):
        """Test remove regularization updates matrix inverse correctly."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        Z = np.arange(8).reshape(4, 2)

        # Compute the inverse without regularization.
        G = kernel_fn(Z)
        groundTruth = np.linalg.inv(G)

        # Compute the regularized inverse.
        W_reg = np.linalg.inv(G + regularization_param * len(G) * np.identity(len(G)))

        # Compute the modified inverse after removing regularization.
        W = remove_regularization(G, W_reg, regularization_param)

        cls.assertTrue(np.allclose(W, groundTruth))


class TestCholeskyFactorizationUpdate(unittest.TestCase):
    def test_add_sample(cls):
        """Matrix inverse should update correctly when a sample is added."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        # Full sample of observations.
        Z = np.arange(8).reshape(4, 2)
        # Extract reduced set.
        X = Z[:-1]
        y = Z[-1].reshape(1, -1)

        # Compute the Cholesky factorization of the reduced set.
        L_reduced = cholesky(
            kernel_fn(X) + regularization_param * np.identity(len(X)),
        )

        # Compute the Cholesky factorization of the full set.
        groundTruth = cholesky(
            kernel_fn(Z) + regularization_param * np.identity(len(Z)),
        )

        # Compute the modified Cholesky factorization after adding a sample.
        L = cho_add_rc((L_reduced, False), X, y, kernel_fn, regularization_param)

        cls.assertTrue(np.allclose(L, groundTruth))

    def test_remove_first_sample(cls):
        """Matrix inverse should update correctly when the first sample is removed."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        # Full sample of observations.
        Z = np.arange(8).reshape(4, 2)
        # Extract reduced set.
        X = Z[1:]

        # Compute the Cholesky factorization of the reduced set.
        groundTruth = cholesky(
            kernel_fn(X) + regularization_param * np.identity(len(X)),
        )

        # Compute the Cholesky factorization of the full set.
        L_full = cholesky(
            kernel_fn(Z) + regularization_param * np.identity(len(Z)),
        )

        # Compute the modified Cholesky factorization after removing a sample.
        L = cho_del_rc((L_full, False), last=False)

        cls.assertTrue(np.allclose(L, groundTruth))

    def test_remove_last_sample(cls):
        """Matrix inverse should update correctly when the last sample is removed."""

        regularization_param = 1e-1
        kernel_fn = partial(rbf_kernel, sigma=1)

        # Full sample of observations.
        Z = np.arange(8).reshape(4, 2)
        # Extract reduced set.
        X = Z[:-1]

        # Compute the Cholesky factorization of the reduced set.
        groundTruth = cholesky(
            kernel_fn(X) + regularization_param * np.identity(len(X)),
        )

        # Compute the Cholesky factorization of the full set.
        L_full = cholesky(
            kernel_fn(Z) + regularization_param * np.identity(len(Z)),
        )

        # Compute the modified Cholesky factorization after removing a sample.
        L = cho_del_rc((L_full, False), last=True)

        cls.assertTrue(np.allclose(L, groundTruth))
