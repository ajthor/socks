import unittest

import numpy as np

from gym_socks.envs.aero.f16 import MorelliF16Env
from gym_socks.envs.aero.f16 import StevensF16Env


class TestStevensF16Dynamics(unittest.TestCase):
    def test_stevens_f16_dynamics(cls):
        env = StevensF16Env()
        env._xcg = 0.40

        dx = env.dynamics(
            time=0,
            state=[
                500,  # VT
                0.5,  # alpha
                -0.2,  # beta
                -1,  # phi
                1,  # theta
                -1,  # psi
                0.7,  # p
                -0.8,  # q
                0.9,  # r
                1000,  # x
                900,  # y
                10000,  # altitude
                90,  # power
            ],
            action=[
                0.9,  # throttle
                20,  # elevator
                -15,  # aileron
                -20,  # rudder
            ],
            disturbance=0,
        )

        GroundTruth = [
            -75.23724,
            -0.8813491,
            -0.4759990,
            2.505734,
            0.3250820,
            2.145926,
            12.62679,
            0.9649671,
            0.5809759,
            342.4439,
            -266.7707,
            248.1241,
            -58.6899,
        ]

        np.testing.assert_allclose(dx, GroundTruth, atol=0.5)

        # Assert that the difference between each element is less than 0.05
        # assert abs(dx[0] - GroundTruth[0]) < 0.05
        # assert abs(dx[1] - GroundTruth[1]) < 0.05
        # assert abs(dx[2] - GroundTruth[2]) < 0.05
        # assert abs(dx[3] - GroundTruth[3]) < 0.05
        # assert abs(dx[4] - GroundTruth[4]) < 0.05
        # assert abs(dx[5] - GroundTruth[5]) < 0.05
        # assert abs(dx[6] - GroundTruth[6]) < 0.05
        # assert abs(dx[7] - GroundTruth[7]) < 0.05
        # assert abs(dx[8] - GroundTruth[8]) < 0.05
        # assert abs(dx[9] - GroundTruth[9]) < 0.05
        # assert abs(dx[10] - GroundTruth[10]) < 0.05
        # assert abs(dx[11] - GroundTruth[11]) < 0.05
        # assert abs(dx[12] - GroundTruth[12]) < 0.05


class TestMorelliF16Dynamics(unittest.TestCase):
    def test_morelli_f16_dynamics(cls):
        env = MorelliF16Env()
        env._xcg = 0.40

        dx = env.dynamics(
            time=0,
            state=[
                500,  # VT
                0.5,  # alpha
                -0.2,  # beta
                -1,  # phi
                1,  # theta
                -1,  # psi
                0.7,  # p
                -0.8,  # q
                0.9,  # r
                1000,  # x
                900,  # y
                10000,  # altitude
                90,  # power
            ],
            action=[
                0.9,  # throttle
                20,  # elevator
                -15,  # aileron
                -20,  # rudder
            ],
            disturbance=0,
        )

        GroundTruth = [
            -77.57521,
            -0.88123,
            -0.45276,
            0.70000,
            0.32508,
            2.14593,
            12.91108,
            0.97006,
            -0.55450,
            342.44390,
            -266.77068,
            248.12412,
            -58.6900,
        ]

        np.testing.assert_allclose(dx, GroundTruth, atol=0.5)
