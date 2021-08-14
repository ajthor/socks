import unittest

import gym

import gym_basic.envs

import numpy as np

from gym_basic.envs.sample import generate_sample
from gym_basic.envs.sample import generate_sample_trajectories


class TestGenerateSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [
            gym_basic.envs.integrator.NDIntegratorEnv(1),
            gym_basic.envs.integrator.NDIntegratorEnv(2),
            gym_basic.envs.integrator.NDIntegratorEnv(3),
            gym_basic.envs.integrator.NDIntegratorEnv(4),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(1),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(2),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(3),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(4),
            gym_basic.envs.point_mass.NDPointMassEnv(1),
            gym_basic.envs.point_mass.NDPointMassEnv(2),
            gym_basic.envs.point_mass.NDPointMassEnv(3),
            gym_basic.envs.point_mass.NDPointMassEnv(4),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(1),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(2),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(3),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(4),
            gym_basic.envs.nonholonomic.NonholonomicVehicleEnv(),
            gym_basic.envs.cwh.CWH4DEnv(),
            gym_basic.envs.cwh.CWH6DEnv(),
            gym_basic.envs.QUAD20.QuadrotorEnv(),
            gym_basic.envs.QUAD20.StochasticQuadrotorEnv(),
        ]

    def test_sample_is_ndarray(cls):
        """
        Assert that generate_sample generates a list of state observations.
        """

        for env in cls.envs:
            with cls.subTest(f"Testing with {type(env)}."):

                sample_space = gym.spaces.Box(
                    low=-0.1,
                    high=0.1,
                    shape=env.observation_space.shape,
                    dtype=np.float32,
                )

                S = generate_sample(sample_space, env, 5)

                cls.assertIsInstance(S, np.ndarray, "Should be an ndarray.")
                cls.assertEqual(
                    np.shape(S),
                    (5, 2, env.observation_space.shape[0]),
                    "Should return the correct dimensions.",
                )

    def test_incorrect_sample_space_dimensionality(cls):
        """
        Test for failure if sample_space is incorrect dimensionality.
        """

        for env in cls.envs:
            with cls.subTest(f"Testing with {type(env)}."):

                sample_space = gym.spaces.Box(
                    low=-0.1,
                    high=0.1,
                    shape=(env.observation_space.shape[0] + 1,),
                    dtype=np.float32,
                )

                cls.assertNotEqual(env.observation_space.shape, sample_space.shape)
                with cls.assertRaises(AssertionError) as exception_context:
                    S = generate_sample(sample_space, env, 5)


class TestGenerateSampleTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [
            gym_basic.envs.integrator.NDIntegratorEnv(1),
            gym_basic.envs.integrator.NDIntegratorEnv(2),
            gym_basic.envs.integrator.NDIntegratorEnv(3),
            gym_basic.envs.integrator.NDIntegratorEnv(4),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(1),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(2),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(3),
            gym_basic.envs.integrator.StochasticNDIntegratorEnv(4),
            gym_basic.envs.point_mass.NDPointMassEnv(1),
            gym_basic.envs.point_mass.NDPointMassEnv(2),
            gym_basic.envs.point_mass.NDPointMassEnv(3),
            gym_basic.envs.point_mass.NDPointMassEnv(4),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(1),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(2),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(3),
            gym_basic.envs.point_mass.StochasticNDPointMassEnv(4),
            gym_basic.envs.nonholonomic.NonholonomicVehicleEnv(),
            gym_basic.envs.cwh.CWH4DEnv(),
            gym_basic.envs.cwh.CWH6DEnv(),
            gym_basic.envs.QUAD20.QuadrotorEnv(),
            gym_basic.envs.QUAD20.StochasticQuadrotorEnv(),
        ]

    def test_sample_trajectories(cls):
        """
        Assert that generate_state_trajectory generates a list of state observations.
        """

        for env in cls.envs:
            with cls.subTest(f"Testing with {type(env)}."):

                sample_space = gym.spaces.Box(
                    low=-0.1,
                    high=0.1,
                    shape=env.observation_space.shape,
                    dtype=np.float32,
                )

                S = generate_sample_trajectories(sample_space, env, 5)

                len = int(np.floor(env.time_horizon / env.sampling_time)) + 1

                cls.assertIsInstance(S, np.ndarray, "Should be an ndarray.")
                cls.assertEqual(
                    np.shape(S),
                    (5, len, env.observation_space.shape[0]),
                    "Should return the correct dimensions.",
                )

    def test_incorrect_sample_space_dimensionality(cls):
        """
        Test for failure if sample_space is incorrect dimensionality.
        """

        for env in cls.envs:
            with cls.subTest(f"Testing with {type(env)}."):

                sample_space = gym.spaces.Box(
                    low=-0.1,
                    high=0.1,
                    shape=(env.observation_space.shape[0] + 1,),
                    dtype=np.float32,
                )

                cls.assertNotEqual(env.observation_space.shape, sample_space.shape)
                with cls.assertRaises(AssertionError) as exception_context:
                    S = generate_sample_trajectories(sample_space, env, 5)

    def test_known_trajectory(cls):
        """
        Test against specific known trajectory.
        """

        env = gym_basic.envs.integrator.NDIntegratorEnv(2)
        env.action_space = gym.spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)

        sample_space = gym.spaces.Box(low=0.1, high=0.1, shape=(2,), dtype=np.float32)

        S = generate_sample_trajectories(sample_space, env, 1)

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
        cls.assertTrue(np.allclose(S, groundTruth))


if __name__ == "__main__":
    unittest.main()
