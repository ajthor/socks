import unittest

import gym

import gym_basic.envs

import numpy as np

from gym_basic.envs.sample import generate_sample
from gym_basic.envs.sample import generate_sample_trajectories


class Test4DCWHSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_basic.envs.cwh.CWH4DEnv()

    def test_known_trajectory(cls):
        """
        Test against specific known trajectory. Sanity check.
        """

        env = cls.env

        env.state = np.array([-0.1, -0.1, 0, 0])
        action = np.array([0, 0])

        trajectory = []
        trajectory.append(env.state)

        for i in range(3):
            obs, reward, done, _ = env.step(action)
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
        cls.env = gym_basic.envs.cwh.CWH6DEnv()

    def test_known_trajectory(cls):
        """
        Test against specific known trajectory. Sanity check.
        """

        env = cls.env

        env.state = np.array([-0.1, -0.1, 0, 0, 0, 0])
        action = np.array([0, 0, 0])

        trajectory = []
        trajectory.append(env.state)

        for i in range(3):
            obs, reward, done, _ = env.step(action)
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
