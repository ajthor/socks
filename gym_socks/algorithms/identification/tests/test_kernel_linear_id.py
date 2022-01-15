import unittest
from unittest import mock
from unittest.mock import patch

import gym

import numpy as np

from gym_socks.algorithms.identification.kernel_linear_id import KernelLinearId
from gym_socks.envs import CWH4DEnv
from gym_socks.envs import CWH6DEnv
from gym_socks.policies import RandomizedPolicy

from gym_socks.sampling import sample
from gym_socks.sampling import sample_generator


def dummy_sampler(env, policy, sample_space):
    @sample_generator
    def _sample_generator():
        state = sample_space.sample()
        action = policy(time=0, state=state)

        env.state = state
        next_state, *_ = env.step(action=action)

        yield (state, action, next_state)

    return _sample_generator


class TestKernelLinearIdentificationAlgorithm(unittest.TestCase):
    def test_linear_identification_CWH4D(cls):
        """Should identify linear dynamics of CWH4D system within a tolerance."""

        env = CWH4DEnv()
        env.seed(seed=0)

        S = sample(
            sampler=dummy_sampler(
                env=env,
                policy=RandomizedPolicy(action_space=env.action_space),
                sample_space=env.state_space,
            ),
            sample_size=1000,
        )

        regularization_param = 1e-9

        alg = KernelLinearId(regularization_param=regularization_param, verbose=False)
        alg.fit(S=S)

        cls.assertTrue(np.allclose(alg.state_matrix, env.state_matrix, atol=1e-2))
        cls.assertTrue(np.allclose(alg.input_matrix, env.input_matrix, atol=1e-2))

    def test_linear_identification_CWH6D(cls):
        """Should identify linear dynamics of CWH6D system within a tolerance."""

        env = CWH6DEnv()
        env.seed(seed=0)

        S = sample(
            sampler=dummy_sampler(
                env=env,
                policy=RandomizedPolicy(action_space=env.action_space),
                sample_space=env.state_space,
            ),
            sample_size=1000,
        )

        regularization_param = 1e-9

        alg = KernelLinearId(regularization_param=regularization_param, verbose=False)
        alg.fit(S=S)

        cls.assertTrue(np.allclose(alg.state_matrix, env.state_matrix, atol=1e-2))
        cls.assertTrue(np.allclose(alg.input_matrix, env.input_matrix, atol=1e-2))
