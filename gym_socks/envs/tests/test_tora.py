import unittest
from unittest.mock import patch

import gym

import gym_socks.envs

import numpy as np


class TestToraSystem(unittest.TestCase):
    def test_set_damping_coefficient(cls):
        system = gym_socks.envs.TORAEnv()

        system.damping_coefficient = 0.5
        cls.assertEqual(system.damping_coefficient, 0.5)
