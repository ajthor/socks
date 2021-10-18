import unittest
from unittest.mock import patch

import gym

import gym_socks.envs

from gym_socks.envs.policy import BasePolicy, ZeroPolicy

import numpy as np


class TestBasePolicy(unittest.TestCase):
    @classmethod
    @patch("gym_socks.envs.policy.BasePolicy.__abstractmethods__", set())
    def setUpClass(cls):
        cls.policy = BasePolicy()

    def test_default_policy_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.policy()


class TestConstantPolicy(unittest.TestCase):
    @classmethod
    @patch("gym_socks.envs.dynamical_system.DynamicalSystem.__abstractmethods__", set())
    def setUpClass(cls):
        cls.system = gym_socks.envs.dynamical_system.DynamicalSystem(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    def test_constant_policy_returns_constants(cls):
        policy = gym_socks.envs.policy.ConstantPolicy(system=cls.system, constant=1)
        cls.assertEqual(policy(), [1])

        policy = gym_socks.envs.policy.ConstantPolicy(system=cls.system, constant=5.0)
        cls.assertEqual(policy(), [5.0])


class TestZeroPolicy(unittest.TestCase):
    @classmethod
    @patch("gym_socks.envs.dynamical_system.DynamicalSystem.__abstractmethods__", set())
    def setUpClass(cls):
        cls.system = gym_socks.envs.dynamical_system.DynamicalSystem(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    def test_zero_policy_returns_zero(cls):
        policy = gym_socks.envs.policy.ZeroPolicy(system=cls.system)
        cls.assertEqual(policy(), [0])
