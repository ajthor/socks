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
    gym_socks.envs.StochasticNonholonomicVehicleEnv(),
    gym_socks.envs.CWH4DEnv(),
    gym_socks.envs.CWH6DEnv(),
    gym_socks.envs.StochasticCWH4DEnv(),
    gym_socks.envs.StochasticCWH6DEnv(),
    gym_socks.envs.QuadrotorEnv(),
    gym_socks.envs.StochasticQuadrotorEnv(),
    gym_socks.envs.TORAEnv(),
    gym_socks.envs.StochasticTORAEnv(),
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
        cls.system = gym_socks.envs.dynamical_system.DynamicalSystem(
            observation_space=gym.spaces.Box(
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
            cls.system.time_horizon = 1
            cls.system.sampling_time = 2

        with cls.assertWarns(Warning):
            cls.system.sampling_time = 1
            cls.system.time_horizon = 0.1

    def test_system_num_time_steps(cls):
        """System returns correct number of time steps."""

        cls.system.time_horizon = 1
        cls.system.sampling_time = 0.1

        # Note that this is not an error, and has to do with floating point precision.
        # See: https://docs.python.org/3/tutorial/floatingpoint.html
        cls.assertEqual(cls.system.num_time_steps, 9)
        cls.assertEqual(cls.system.time_horizon, 1)
        cls.assertEqual(cls.system.sampling_time, 0.1)

        cls.system.time_horizon = 10.0
        cls.system.sampling_time = 1.0

        # Note that this is not an error, and has to do with floating point precision.
        # See: https://docs.python.org/3/tutorial/floatingpoint.html
        cls.assertEqual(cls.system.num_time_steps, 10)
        cls.assertEqual(cls.system.time_horizon, 10)
        cls.assertEqual(cls.system.sampling_time, 1.0)

        cls.system.time_horizon = 5
        cls.system.sampling_time = 1

        cls.assertEqual(cls.system.num_time_steps, 5)
        cls.assertEqual(cls.system.time_horizon, 5)
        cls.assertEqual(cls.system.sampling_time, 1)

    def test_dims(cls):
        """State and action dims should match spaces."""
        cls.assertEqual(cls.system.state_dim, (1,))
        cls.assertEqual(cls.system.action_dim, (1,))

        cls.system.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        cls.assertEqual(cls.system.state_dim, (2,))

    def test_cannot_change_dims(cls):
        """Cannot change the state and action dims."""

        with cls.assertRaises(AttributeError):
            cls.system.state_dim = 1

        with cls.assertRaises(AttributeError):
            cls.system.action_dim = 1

    def test_reset_returns_valid_state(cls):
        """Reset should return a valid state."""
        cls.system.reset()
        cls.assertTrue(cls.system.observation_space.contains(cls.system.state))

    def test_default_dynamics_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            state = cls.system.reset()
            action = cls.system.action_space.sample()
            cls.system.dynamics(0, state, action)

    def test_default_close_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.system.close()

    def test_default_render_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            cls.system.render()


class TestStochasticDynamicalSystem(unittest.TestCase):
    """Stochastic dynamical system tests."""

    @patch(
        "gym_socks.envs.dynamical_system.StochasticMixin.__abstractmethods__",
        set(),
    )
    def setUp(cls):
        cls.system = gym_socks.envs.dynamical_system.StochasticMixin(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            disturbance_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            seed=1,
        )

    @patch(
        "gym_socks.envs.dynamical_system.StochasticMixin.__abstractmethods__",
        set(),
    )
    def test_should_fail_without_spaces(cls):
        """Test that system throws an error without disturbance space."""
        with cls.assertRaises(ValueError):
            system = gym_socks.envs.dynamical_system.StochasticMixin(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                action_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                disturbance_space=None,
            )

    def test_default_dynamics_not_implemented(cls):

        with cls.assertRaises(NotImplementedError):
            state = cls.system.reset()
            action = cls.system.action_space.sample()
            cls.system.dynamics(0, state, action)
