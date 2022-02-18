import unittest
from unittest.mock import patch

import gym

import gym_socks.envs

from gym_socks.policies import BasePolicy

from gym_socks.policies import ConstantPolicy
from gym_socks.policies import RandomizedPolicy
from gym_socks.policies import ZeroPolicy

import numpy as np


class TestBasePolicy(unittest.TestCase):
    @classmethod
    @patch("gym_socks.policies.policy.BasePolicy.__abstractmethods__", set())
    def setUpClass(cls):
        cls.policy = BasePolicy()

    def test_default_policy_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.policy()


class TestConstantPolicy(unittest.TestCase):
    @classmethod
    @patch("gym_socks.envs.dynamical_system.DynamicalSystem.__abstractmethods__", set())
    def setUpClass(cls):
        cls.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def test_constant_policy_returns_constants(cls):
        policy = ConstantPolicy(action_space=cls.action_space, constant=1)
        cls.assertEqual(policy(), [1])

        policy = ConstantPolicy(action_space=cls.action_space, constant=5.0)
        cls.assertEqual(policy(), [5.0])


class TestZeroPolicy(unittest.TestCase):
    @classmethod
    @patch("gym_socks.envs.dynamical_system.DynamicalSystem.__abstractmethods__", set())
    def setUpClass(cls):
        cls.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def test_zero_policy_returns_zero(cls):
        policy = ZeroPolicy(action_space=cls.action_space)
        cls.assertEqual(policy(), [0])
