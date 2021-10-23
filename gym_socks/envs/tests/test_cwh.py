import unittest
from unittest import mock
from unittest.mock import patch

import gym

import gym_socks.envs

import numpy as np

from scipy.constants import gravitational_constant


class Test4DCWHSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.cwh.CWH4DEnv()

    def test_set_parameters(cls):

        state_matrix = cls.env.state_matrix
        input_matrix = cls.env.input_matrix

        # system parameters
        cls.env.orbital_radius = 850 + 6378.1
        cls.env.gravitational_constant = gravitational_constant
        cls.env.celestial_body_mass = 5.9472e24
        cls.env.chief_mass = 300

        cls.assertTrue(np.allclose(state_matrix, cls.env.state_matrix))
        cls.assertTrue(np.allclose(input_matrix, cls.env.input_matrix))

    @patch.object(gym_socks.envs.cwh.CWH4DEnv, "generate_disturbance")
    def test_known_trajectory(cls, mock_generate_disturbance):
        """Test against specific known trajectory. Sanity check."""
        mock_generate_disturbance.return_value = np.zeros((4,))

        env = cls.env

        env.state = np.array([-0.1, -0.1, 0, 0], dtype=np.float32)
        action = np.array([0, 0], dtype=np.float32)

        trajectory = []
        trajectory.append(env.state)

        for i in range(3):
            obs, cost, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

        n = env.angular_velocity
        nt = n * env.sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        A = [
            [4 - 3 * cos_nt, 0, (1 / n) * sin_nt, (2 / n) * (1 - cos_nt)],
            [
                6 * (sin_nt - nt),
                1,
                -(2 / n) * (1 - cos_nt),
                (1 / n) * (4 * sin_nt - 3 * nt),
            ],
            [3 * n * sin_nt, 0, cos_nt, 2 * sin_nt],
            [-6 * n * (1 - cos_nt), 0, -2 * sin_nt, 4 * cos_nt - 3],
        ]

        state = np.array([-0.1, -0.1, 0, 0])

        groundTruth = []
        groundTruth.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            groundTruth.append(state)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )

        state = np.array([-0.1, 0.1, 0, 0])

        falseTrajectory = []
        falseTrajectory.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            falseTrajectory.append(state)

        cls.assertFalse(
            np.allclose(trajectory, falseTrajectory),
            "Generated trajectory should not match known false trajectory generated using dynamics.",
        )


class Test6DCWHSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.cwh.CWH6DEnv()

    @patch.object(gym_socks.envs.cwh.CWH6DEnv, "generate_disturbance")
    def test_known_trajectory(cls, mock_generate_disturbance):
        """Test against specific known trajectory. Sanity check."""
        mock_generate_disturbance.return_value = np.zeros((6,))

        env = cls.env

        env.state = np.array([-0.1, -0.1, 0, 0, 0, 0], dtype=np.float32)
        action = np.array([0, 0, 0], dtype=np.float32)

        trajectory = []
        trajectory.append(env.state)

        for i in range(3):
            obs, cost, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

        n = env.angular_velocity
        nt = n * env.sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        A = [
            [4 - 3 * cos_nt, 0, 0, (1 / n) * sin_nt, (2 / n) * (1 - cos_nt), 0],
            [
                6 * (sin_nt - nt),
                1,
                0,
                -(2 / n) * (1 - cos_nt),
                (1 / n) * (4 * sin_nt - 3 * nt),
                0,
            ],
            [0, 0, cos_nt, 0, 0, (1 / n) * sin_nt],
            [3 * n * sin_nt, 0, 0, cos_nt, 2 * sin_nt, 0],
            [-6 * n * (1 - cos_nt), 0, 0, -2 * sin_nt, 4 * cos_nt - 3, 0],
            [0, 0, -n * sin_nt, 0, 0, cos_nt],
        ]

        state = np.array([-0.1, -0.1, 0, 0, 0, 0])

        groundTruth = []
        groundTruth.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            groundTruth.append(state)

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )

        state = np.array([-0.1, 0.1, 0, 0, 0, 0])

        falseTrajectory = []
        falseTrajectory.append(state)

        for i in range(3):
            state = np.matmul(A, state.T)
            falseTrajectory.append(state)

        cls.assertFalse(
            np.allclose(trajectory, falseTrajectory),
            "Generated trajectory should not match known false trajectory generated using dynamics.",
        )
