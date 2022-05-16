import unittest
from unittest import mock
from unittest.mock import patch

import gym
import gym_socks

from scipy.constants import unit

from gym_socks.envs.integrator import NDIntegratorEnv
from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.policies import ZeroPolicy
from gym_socks.policies import RandomizedPolicy

from gym_socks.sampling import sample_fn
from gym_socks.sampling import space_sampler
from gym_socks.sampling import trajectory_sampler
from gym_socks.sampling.transform import transpose_sample

from gym_socks.algorithms.reach.common import _tht_step, _fht_step
from gym_socks.algorithms.reach.monte_carlo import MonteCarloSR
from gym_socks.algorithms.reach.monte_carlo import monte_carlo_sr
from gym_socks.algorithms.reach.monte_carlo import _trajectory_indicator

import numpy as np


def make_tube(
    time_horizon: int, shape: tuple, lower_bound: float, upper_bound: float
) -> list:
    """Creates a tube for testing."""

    tube = []
    for i in range(time_horizon):
        tube_t = gym.spaces.Box(
            lower_bound,
            upper_bound,
            shape=shape,
            dtype=np.float32,
        )
        tube.append(tube_t)

    return tube


@sample_fn
def zero_sampler(sample_space):
    yield np.zeros(shape=sample_space.shape, dtype=sample_space.dtype)


class TestTrajectoryIndicator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = NDIntegratorEnv(2)

        cls.time_horizon = 20

        cls.target_tube = make_tube(
            time_horizon=cls.time_horizon + 1,
            shape=cls.env.state_space.shape,
            lower_bound=-0.25,
            upper_bound=0.25,
        )
        cls.constraint_tube = make_tube(
            time_horizon=cls.time_horizon + 1,
            shape=cls.env.state_space.shape,
            lower_bound=-1,
            upper_bound=1,
        )

    @patch.object(gym_socks.envs.NDIntegratorEnv, "generate_disturbance")
    def test_trajectory_indicator_inside(cls, mock_generate_disturbance):
        """Indicator should return one if trajectory satisfies constraints."""
        mock_generate_disturbance.return_value = np.zeros((2,))

        env = NDIntegratorEnv(2)

        # Test inside.

        # The trajectory of the system with no disturbance starts at (-0.1, 0.1) and
        # ends at (0.1, 0.1) after 20 time steps.

        sample_space = gym.spaces.Box(
            low=np.array([-0.1, 0.1], dtype=np.float32),
            high=np.array([-0.1, 0.1], dtype=np.float32),
            dtype=np.float32,
        )

        sample_size = 5
        state_sampler = space_sampler(space=sample_space)
        action_sampler = zero_sampler(sample_space=env.action_space)
        S = trajectory_sampler(
            env, state_sampler, action_sampler, cls.time_horizon
        ).sample(size=sample_size)

        # S = sample(
        #     sampler=default_trajectory_sampler(
        #         state_sampler=random_sampler(sample_space=sample_space),
        #         action_sampler=zero_sampler(sample_space=env.action_space),
        #         env=env,
        #         time_horizon=cls.time_horizon,
        #     ),
        #     sample_size=sample_size,
        # )

        X, _, Y = transpose_sample(S)
        full_trajectories = [(X[i], *Y[i]) for i in range(sample_size)]

        result = _trajectory_indicator(
            trajectories=full_trajectories,
            time_horizon=cls.time_horizon,
            constraint_tube=cls.constraint_tube,
            target_tube=cls.target_tube,
            step_fn=_tht_step,
        )

        cls.assertTrue(np.all(result))

    @patch.object(gym_socks.envs.NDIntegratorEnv, "generate_disturbance")
    def test_trajectory_indicator_outside(cls, mock_generate_disturbance):
        """Indicator should return zero if trajectory does not satisfy constraints."""
        mock_generate_disturbance.return_value = np.zeros((2,))

        env = NDIntegratorEnv(2)

        # Test not inside.

        # The trajectory of the system with no disturbance starts at (0.2, 0.1) and
        # ends at (0.4, 0.1) after 20 time steps.

        sample_space = gym.spaces.Box(
            low=np.array([0.2, 0.1], dtype=np.float32),
            high=np.array([0.2, 0.1], dtype=np.float32),
            dtype=np.float32,
        )

        sample_size = 5

        state_sampler = space_sampler(space=sample_space)
        action_sampler = zero_sampler(sample_space=env.action_space)
        S = trajectory_sampler(
            env, state_sampler, action_sampler, cls.time_horizon
        ).sample(size=sample_size)

        # S = sample(
        #     sampler=default_trajectory_sampler(
        #         state_sampler=random_sampler(sample_space=sample_space),
        #         action_sampler=zero_sampler(sample_space=env.action_space),
        #         env=env,
        #         time_horizon=cls.time_horizon,
        #     ),
        #     sample_size=sample_size,
        # )

        X, _, Y = transpose_sample(S)
        full_trajectories = [(X[i], *Y[i]) for i in range(sample_size)]

        result = _trajectory_indicator(
            trajectories=full_trajectories,
            time_horizon=cls.time_horizon,
            constraint_tube=cls.constraint_tube,
            target_tube=cls.target_tube,
            step_fn=_tht_step,
        )

        cls.assertTrue(~np.any(result))
