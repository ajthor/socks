import unittest
from unittest.mock import patch

import gym

import gym_socks.envs

import numpy as np


class TestNonholonomicSystem(unittest.TestCase):
    def test_corrects_angle(cls):
        system = gym_socks.envs.NonholonomicVehicleEnv()
        system.disturbance_space = gym.spaces.Box(
            low=0, high=0, shape=(3,), dtype=np.float32
        )

        system.state = np.array([0, 0, 4 * np.pi])
        action = np.array([0, 0], dtype=np.float32)
        obs, cost, done, _ = system.step(action)

        cls.assertTrue((system.state[2] <= 2 * np.pi) & (system.state[2] >= -2 * np.pi))
