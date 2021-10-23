import unittest
from unittest.mock import patch

import gym

import gym_socks.envs

import numpy as np


class TestIntegratorSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.NDIntegratorEnv(2)

    @patch.object(gym_socks.envs.NDIntegratorEnv, "generate_disturbance")
    def test_known_trajectory(cls, mock_generate_disturbance):
        """Test against specific known trajectory. Sanity check."""
        mock_generate_disturbance.return_value = np.zeros((2,))

        env = cls.env

        env.state = np.array([0.1, 0.1], dtype=np.float32)
        action = np.array([0], dtype=np.float32)

        trajectory = []
        trajectory.append(env.state)

        for i in range(env.num_time_steps):
            obs, cost, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

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

    @patch.object(gym_socks.envs.NDIntegratorEnv, "generate_disturbance")
    def test_euler_approximation(cls, mock_generate_disturbance):
        """Test against specific known trajectory (Euler). Sanity check."""
        mock_generate_disturbance.return_value = np.zeros((2,))

        env = cls.env
        env._euler = True

        env.state = np.array([0.1, 0.1], dtype=np.float32)
        action = np.array([0], dtype=np.float32)

        trajectory = []
        trajectory.append(env.state)

        for i in range(env.num_time_steps):
            obs, cost, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

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
