import unittest
from unittest.mock import patch

import gym

import gym_socks.envs

import numpy as np

system_list = [
    gym_socks.envs.NDIntegratorEnv(1),
    gym_socks.envs.NDIntegratorEnv(2),
    gym_socks.envs.NDIntegratorEnv(3),
    gym_socks.envs.NDIntegratorEnv(4),
    gym_socks.envs.NDPointMassEnv(1),
    gym_socks.envs.NDPointMassEnv(2),
    gym_socks.envs.NDPointMassEnv(3),
    gym_socks.envs.NDPointMassEnv(4),
    gym_socks.envs.NonholonomicVehicleEnv(),
    gym_socks.envs.CWH4DEnv(),
    gym_socks.envs.CWH6DEnv(),
    gym_socks.envs.QuadrotorEnv(),
    gym_socks.envs.TORAEnv(),
]


class TestEnvironmentsRun(unittest.TestCase):
    def test_envs_run(cls):
        """Assert that environments can run."""

        for env in system_list:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                try:
                    obs = env.reset()

                    for i in range(env.num_time_steps):

                        # get action
                        action = env.action_space.sample()

                        # apply action
                        obs, cost, done, _ = env.step(action)

                except Exception as e:
                    cls.fail(f"Simulating system {type(env)} raised an exception.")


class TestDynamicalSystem(unittest.TestCase):
    """Dynamical system tests."""

    @patch("gym_socks.envs.dynamical_system.DynamicalSystem.__abstractmethods__", set())
    def setUp(cls):
        cls.env = gym_socks.envs.dynamical_system.DynamicalSystem(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            state_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    @patch("gym_socks.envs.dynamical_system.DynamicalSystem.__abstractmethods__", set())
    def test_should_fail_without_spaces(cls):
        """Test that system throws an error if spaces are not specified."""
        with cls.assertRaises(ValueError):
            system = gym_socks.envs.dynamical_system.DynamicalSystem(
                observation_space=None, action_space=None
            )

        with cls.assertRaises(ValueError):
            system = gym_socks.envs.dynamical_system.DynamicalSystem(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                action_space=None,
            )

        with cls.assertRaises(ValueError):
            system = gym_socks.envs.dynamical_system.DynamicalSystem(
                observation_space=None,
                action_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            )

    def test_system_warns_time_step(cls):
        """
        Test that system throws a warning if sampling time is greater than time horizon.
        """

        with cls.assertWarns(Warning):
            cls.env.time_horizon = 1
            cls.env.sampling_time = 2

        with cls.assertWarns(Warning):
            cls.env.sampling_time = 1
            cls.env.time_horizon = 0.1

    def test_system_num_time_steps(cls):
        """System returns correct number of time steps."""

        cls.env.time_horizon = 1
        cls.env.sampling_time = 0.1

        # Note that this is not an error, and has to do with floating point precision.
        # See: https://docs.python.org/3/tutorial/floatingpoint.html
        cls.assertEqual(cls.env.num_time_steps, 9)
        cls.assertEqual(cls.env.time_horizon, 1)
        cls.assertEqual(cls.env.sampling_time, 0.1)

        cls.env.time_horizon = 10.0
        cls.env.sampling_time = 1.0

        # Note that this is not an error, and has to do with floating point precision.
        # See: https://docs.python.org/3/tutorial/floatingpoint.html
        cls.assertEqual(cls.env.num_time_steps, 10)
        cls.assertEqual(cls.env.time_horizon, 10)
        cls.assertEqual(cls.env.sampling_time, 1.0)

        cls.env.time_horizon = 5
        cls.env.sampling_time = 1

        cls.assertEqual(cls.env.num_time_steps, 5)
        cls.assertEqual(cls.env.time_horizon, 5)
        cls.assertEqual(cls.env.sampling_time, 1)

    def test_dims(cls):
        """State and action dims should match spaces."""
        cls.assertEqual(cls.env.state_dim, (1,))
        cls.assertEqual(cls.env.action_dim, (1,))

        cls.env.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        cls.assertEqual(cls.env.state_dim, (1,))

    def test_cannot_change_dims(cls):
        """Cannot change the state and action dims."""

        with cls.assertRaises(AttributeError):
            cls.env.state_dim = 1

        with cls.assertRaises(AttributeError):
            cls.env.action_dim = 1

    def test_reset_returns_valid_state(cls):
        """Reset should return a valid state."""
        cls.env.reset()
        cls.assertTrue(cls.env.observation_space.contains(cls.env.state))

    def test_default_dynamics_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            state = cls.env.reset()
            action = cls.env.action_space.sample()
            disturbance = cls.env.generate_disturbance(0, state, action)
            cls.env.dynamics(0, state, action, disturbance)

    def test_default_close_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.env.close()

    def test_default_render_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.env.render()
