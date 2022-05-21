import unittest
from unittest import mock
from unittest.mock import patch

import gym

import numpy as np

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import regularized_inverse


class TestEinsumInverse(unittest.TestCase):
    def test_einsum_inverse(cls):

        X = np.random.randn(10, 2)

        G = rbf_kernel(X)
        W = regularized_inverse(G)

        groundTruth = W @ G

        A1 = np.einsum("ii,ij->ij", W, G)
        A2 = np.einsum("ij,jk->ik", W, G)

        # cls.assertTrue(np.allclose(groundTruth, C))

        print(groundTruth)
        print(A1)
        print(A2)
        print(np.diag(np.diag(W)) @ G)
        print(G)
