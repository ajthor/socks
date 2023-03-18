import unittest
from unittest.mock import patch

from gym_socks.envs.tora import TORAEnv

import numpy as np


class TestToraSystem(unittest.TestCase):
    def test_set_damping_coefficient(cls):
        env = TORAEnv()

        cls.assertEqual(TORAEnv._damping_coefficient, 0.1)
        cls.assertEqual(env.damping_coefficient, 0.1)

        env.damping_coefficient = 0.5
        cls.assertEqual(env.damping_coefficient, 0.5)
