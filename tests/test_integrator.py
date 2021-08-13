import unittest

import gym

import gym_basic.envs

import numpy as np

from gym_basic.envs.sample import generate_sample
from gym_basic.envs.sample import generate_sample_trajectories


class TestIntegratorSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_basic.envs.integrator.NDIntegratorEnv(2)

    def test_known_trajectory(cls):
        """
        Test against specific known trajectory.
        """

        env = cls.env

        env.state = np.array([0.1, 0.1])

        trajectory = []
        trajectory.append(env.state)

        for i in range(10):
            action = np.array([0])
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs)

        trajectory = np.array(trajectory)

        groundTruth = np.array(
            [
                [
                    [0.1, 0.1],
                    [0.11, 0.1],
                    [0.12, 0.1],
                    [0.13, 0.1],
                    [0.14, 0.1],
                    [0.15, 0.1],
                    [0.16, 0.1],
                    [0.17, 0.1],
                    [0.18, 0.1],
                    [0.19, 0.1],
                    [0.2, 0.1],
                ]
            ]
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(
            np.allclose(trajectory, groundTruth),
            "Generated trajectory should match known trajectory generated using dynamics.",
        )
