import unittest
from unittest.mock import patch

import gym

from gym_socks.envs.integrator import NDIntegratorEnv

from gym_socks.policies import ZeroPolicy
from gym_socks.envs.dynamical_system import simulate

import numpy as np


class TestIntegratorSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NDIntegratorEnv(2)

    @patch.object(NDIntegratorEnv, "generate_disturbance")
    def test_known_trajectory(cls, mock_generate_disturbance):
        """Test against specific known trajectory. Sanity check."""
        mock_generate_disturbance.return_value = np.zeros((2,))

        env = cls.env
        env.sampling_time = 0.25

        # env.reset([0.1, 0.1])
        # action = np.array([0], dtype=np.float32)

        # trajectory = []
        # trajectory.append(env.state)

        # for t in range(16):
        #     obs, cost, done, _ = env.step(time=t, action=action)
        #     trajectory.append(obs)

        # trajectory = np.array(trajectory)

        policy = ZeroPolicy(env.action_space)
        trajectory, _ = simulate(env, 16, [0.1, 0.1], policy, True)

        groundTruth = np.array(
            [
                [0.100, 0.1],
                [0.125, 0.1],
                [0.150, 0.1],
                [0.175, 0.1],
                [0.200, 0.1],
                [0.225, 0.1],
                [0.250, 0.1],
                [0.275, 0.1],
                [0.300, 0.1],
                [0.325, 0.1],
                [0.350, 0.1],
                [0.375, 0.1],
                [0.400, 0.1],
                [0.425, 0.1],
                [0.450, 0.1],
                [0.475, 0.1],
                [0.500, 0.1],
            ]
        )

        cls.assertTrue(np.allclose(trajectory, groundTruth))

    @patch.object(NDIntegratorEnv, "generate_disturbance")
    def test_euler_approximation(cls, mock_generate_disturbance):
        """Test against specific known trajectory (Euler). Sanity check."""
        mock_generate_disturbance.return_value = np.zeros((2,))

        env = cls.env
        env._euler = True
        env.sampling_time = 0.25

        # env.reset([0.1, 0.1])
        # action = np.array([0], dtype=np.float32)

        # trajectory = []
        # trajectory.append(env.state)

        # for t in range(16):
        #     obs, cost, done, _ = env.step(time=t, action=action)
        #     trajectory.append(obs)

        # trajectory = np.array(trajectory)

        policy = ZeroPolicy(env.action_space)
        trajectory, _ = simulate(env, 16, [0.1, 0.1], policy, True)

        groundTruth = np.array(
            [
                [0.100, 0.1],
                [0.125, 0.1],
                [0.150, 0.1],
                [0.175, 0.1],
                [0.200, 0.1],
                [0.225, 0.1],
                [0.250, 0.1],
                [0.275, 0.1],
                [0.300, 0.1],
                [0.325, 0.1],
                [0.350, 0.1],
                [0.375, 0.1],
                [0.400, 0.1],
                [0.425, 0.1],
                [0.450, 0.1],
                [0.475, 0.1],
                [0.500, 0.1],
            ]
        )

        cls.assertTrue(np.allclose(trajectory, groundTruth))
