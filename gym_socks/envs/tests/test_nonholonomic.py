import unittest
from unittest.mock import patch

from gym_socks.envs.spaces import Box
from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv

import numpy as np


class TestNonholonomicSystem(unittest.TestCase):
    def test_corrects_angle(cls):
        system = NonholonomicVehicleEnv()
        system.disturbance_space = Box(low=0, high=0, shape=(3,), dtype=float)

        system.state = np.array([0, 0, 4 * np.pi])
        action = np.array([0, 0], dtype=float)
        obs, cost, done, _ = system.step(action)

        cls.assertTrue((system.state[2] <= 2 * np.pi) & (system.state[2] >= -2 * np.pi))
