import unittest

import gym

import gym_socks.envs

import numpy as np

from gym_socks.envs.sample import sample
from gym_socks.envs.sample import sample_trajectories


class TestIntegratorSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.NDIntegratorEnv(2)

    def test_known_trajectory(cls):
        """
        Test against specific known trajectory. Sanity check.
        """

        env = cls.env

        env.state = np.array([0.1, 0.1])
        action = np.array([0])

        trajectory = []
        trajectory.append(env.state)

        for i in range(cls.env.num_time_steps):
            obs, reward, done, _ = env.step(action)
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

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )
