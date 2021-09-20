import unittest

import gym

import gym_socks.envs

import numpy as np

from gym_socks.envs.sample import random_initial_conditions
from gym_socks.envs.sample import uniform_initial_conditions

from gym_socks.envs.sample import generate_sample
from gym_socks.envs.sample import generate_sample_trajectories

from gym_socks.envs.policy import ZeroPolicy

system_list = [
    gym_socks.envs.NDIntegratorEnv(1),
    gym_socks.envs.NDIntegratorEnv(2),
    gym_socks.envs.NDIntegratorEnv(3),
    gym_socks.envs.NDIntegratorEnv(4),
    gym_socks.envs.StochasticNDIntegratorEnv(1),
    gym_socks.envs.StochasticNDIntegratorEnv(2),
    gym_socks.envs.StochasticNDIntegratorEnv(3),
    gym_socks.envs.StochasticNDIntegratorEnv(4),
    gym_socks.envs.NDPointMassEnv(1),
    gym_socks.envs.NDPointMassEnv(2),
    gym_socks.envs.NDPointMassEnv(3),
    gym_socks.envs.NDPointMassEnv(4),
    gym_socks.envs.StochasticNDPointMassEnv(1),
    gym_socks.envs.StochasticNDPointMassEnv(2),
    gym_socks.envs.StochasticNDPointMassEnv(3),
    gym_socks.envs.StochasticNDPointMassEnv(4),
    gym_socks.envs.NonholonomicVehicleEnv(),
    gym_socks.envs.CWH4DEnv(),
    gym_socks.envs.CWH6DEnv(),
    gym_socks.envs.QuadrotorEnv(),
    gym_socks.envs.StochasticQuadrotorEnv(),
]


class TestGenerateInitialConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = system_list

    def test_incorrect_sample_space_dimensionality(cls):
        """
        Test for failure if sample_space is incorrect dimensionality.
        """

        for env in cls.envs:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                sample_space = gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(env.observation_space.shape[0] + 1,),
                    dtype=np.float32,
                )

                cls.assertNotEqual(env.observation_space.shape, sample_space.shape)
                with cls.assertRaises(AssertionError) as exception_context:
                    ic = random_initial_conditions(
                        sample_space=sample_space, system=env, n=5
                    )

                cls.assertNotEqual(env.observation_space.shape, sample_space.shape)
                with cls.assertRaises(AssertionError) as exception_context:
                    ic = uniform_initial_conditions(
                        sample_space=sample_space, system=env, n=5
                    )


class TestGenerateSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = system_list

    def test_generate_sample(cls):
        """
        Assert that generate_sample generates a list of state observations.
        """

        for env in cls.envs:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                obs_shape = env.observation_space.shape

                sample_space = gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=obs_shape,
                    dtype=np.float32,
                )

                initial_conditions = (
                    [[1] * obs_shape[0]],
                    [[1] * obs_shape[0], [2] * obs_shape[0]],
                    random_initial_conditions(system=env, sample_space=sample_space),
                    uniform_initial_conditions(system=env, sample_space=sample_space),
                )

                for ic in initial_conditions:
                    with cls.subTest(msg=f"Testing with different ICs."):

                        S, U = generate_sample(system=env, initial_conditions=ic)

                        cls.assertIsInstance(S, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(S),
                            (np.array(ic).shape[0], 2, obs_shape[0]),
                            "Should return the correct dimensions.",
                        )

                        cls.assertIsInstance(U, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(U),
                            (np.array(ic).shape[0], 1, env.action_space.shape[0]),
                            "Should return the correct dimensions.",
                        )

    def test_known_sample(cls):
        """
        Test against specific known sample.
        """

        env = gym_socks.envs.integrator.NDIntegratorEnv(2)

        policy = ZeroPolicy(env)

        S, U = generate_sample(
            system=env,
            initial_conditions=[[0.1, 0.1], [0.125, 0.1]],
            policy=policy,
        )

        groundTruth = np.array(
            [
                [
                    [0.100, 0.1],
                    [0.125, 0.1],
                ],
                [
                    [0.125, 0.1],
                    [0.150, 0.1],
                ],
            ]
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(np.allclose(S, groundTruth))


class TestGenerateSampleTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = system_list

    def test_generate_sample_trajectories(cls):
        """
        Assert that generate_state_trajectory generates a list of state observations.
        """

        for env in cls.envs:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                obs_shape = env.observation_space.shape

                sample_space = gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=obs_shape,
                    dtype=np.float32,
                )

                initial_conditions = (
                    [[1] * obs_shape[0]],
                    [[1] * obs_shape[0], [2] * obs_shape[0]],
                    random_initial_conditions(system=env, sample_space=sample_space),
                    uniform_initial_conditions(system=env, sample_space=sample_space),
                )

                for ic in initial_conditions:
                    with cls.subTest(msg=f"Testing with different ICs."):

                        S, U = generate_sample_trajectories(
                            system=env, initial_conditions=ic
                        )

                        len = env.num_time_steps + 1

                        cls.assertIsInstance(S, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(S),
                            (np.array(ic).shape[0], len, obs_shape[0]),
                            "Should return the correct dimensions.",
                        )

                        cls.assertIsInstance(U, np.ndarray, "Should be an ndarray.")
                        cls.assertEqual(
                            np.shape(U),
                            (np.array(ic).shape[0], len - 1, env.action_space.shape[0]),
                            "Should return the correct dimensions.",
                        )

    def test_known_trajectory(cls):
        """
        Test against specific known trajectory.
        """

        env = gym_socks.envs.integrator.NDIntegratorEnv(2)

        policy = ZeroPolicy(env)

        S, U = generate_sample_trajectories(
            system=env, initial_conditions=[[0.1, 0.1]], policy=policy
        )

        groundTruth = np.array(
            [
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
            ]
        )

        # tests if the two arrays are equivalent, within tolerance
        cls.assertTrue(np.allclose(S, groundTruth))
