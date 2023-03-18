import unittest
from unittest.mock import patch

from gym_socks.envs.spaces import Box

from gym_socks.envs import NDIntegratorEnv
from gym_socks.policies import ConstantPolicy
from gym_socks.policies import RandomizedPolicy

from gym_socks.sampling.sample import sample_fn
from gym_socks.sampling.sample import space_sampler
from gym_socks.sampling.sample import grid_sampler
from gym_socks.sampling.sample import transition_sampler
from gym_socks.sampling.sample import trajectory_sampler

from gym_socks.sampling.transform import transpose_sample
from gym_socks.sampling.transform import flatten_sample

from gym_socks.utils.grid import boxgrid
from gym_socks.utils.grid import cartesian

import numpy as np


class TestSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NDIntegratorEnv(2)
        cls.policy = RandomizedPolicy(action_space=cls.env.action_space)

        cls.sample_space = Box(low=-1, high=1, shape=(2,), dtype=float)
        cls.sample_space.seed(1)

    def test_random_sampler(cls):
        """Random sampler should output correct sample size."""

        sample_size = 10

        state_sampler = space_sampler(space=cls.env.state_space)
        action_sampler = space_sampler(space=cls.env.action_space)

        S = transition_sampler(
            env=cls.env, state_sampler=state_sampler, action_sampler=action_sampler
        ).sample(size=sample_size)

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.shape(X), (sample_size, 2))
        cls.assertEqual(np.shape(U), (sample_size, 1))
        cls.assertEqual(np.shape(Y), (sample_size, 2))

    def test_grid_sampler(cls):
        """Grid sampler should output items on a grid of the correct size."""

        sample_size = 27

        state_sample_space = Box(
            low=-1,
            high=1,
            shape=cls.env.state_space.shape,
            dtype=cls.env.state_space.dtype,
        )

        state_grid = boxgrid(space=state_sample_space, resolution=3)
        action_grid = cartesian(np.linspace(-1, 1, 3))

        state_sampler = grid_sampler(grid_points=state_grid).repeat(num=3)
        action_sampler = grid_sampler(grid_points=action_grid)

        S = transition_sampler(
            env=cls.env, state_sampler=state_sampler, action_sampler=action_sampler
        ).sample(size=sample_size)

        X, U, Y = transpose_sample(S)

        groundTruth = [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]

        cls.assertTrue(np.array_equiv(X, groundTruth))

        groundTruth = [
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
            [-1.0],
            [0.0],
            [1.0],
        ]

        cls.assertTrue(np.array_equiv(U, groundTruth))

        cls.assertEqual(np.shape(X), (sample_size, 2))
        cls.assertEqual(np.shape(U), (sample_size, 1))
        cls.assertEqual(np.shape(Y), (sample_size, 2))

    def test_default_sampler(cls):
        """Default sampler should output sample of the correct size."""

        sample_size = 10

        state_sampler = space_sampler(space=cls.env.state_space)
        action_sampler = space_sampler(space=cls.env.action_space)

        S = transition_sampler(
            env=cls.env, state_sampler=state_sampler, action_sampler=action_sampler
        ).sample(size=sample_size)

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))

    def test_trajectory_sampler(cls):
        """Default trajectory sampler should output sample of the correct size."""

        sample_size = 10
        time_horizon = 5

        state_sampler = space_sampler(space=cls.env.state_space)
        action_sampler = space_sampler(space=cls.env.action_space)

        S = trajectory_sampler(
            env=cls.env,
            state_sampler=state_sampler,
            action_sampler=action_sampler,
            time_horizon=5,
        ).sample(size=sample_size)

        cls.assertEqual(np.array(S[0][0]).shape, (2,))
        cls.assertEqual(np.array(S[0][1]).shape, (5, 1))
        cls.assertEqual(np.array(S[0][2]).shape, (5, 2))

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 5, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 5, 2))

        ST = flatten_sample(S)
        X, U, Y = transpose_sample(ST)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 5))
        cls.assertEqual(np.array(Y).shape, (sample_size, 10))

    def test_custom_sampler_as_function(cls):
        sample_size = 10

        @sample_fn
        def custom_sampler():
            state = cls.sample_space.sample()
            action = cls.env.action_space.sample()

            cls.env.reset(state)
            next_state, *_ = cls.env.step(action=action)

            return state, action, next_state

        #   ^^^^^^

        S = custom_sampler().sample(size=sample_size)

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))

    def test_custom_sampler_as_generator(cls):
        sample_size = 10

        @sample_fn
        def custom_sampler():
            state = cls.sample_space.sample()
            action = cls.env.action_space.sample()

            cls.env.reset(state)
            next_state, *_ = cls.env.step(action=action)

            yield (state, action, next_state)

        #   ^^^^^

        S = custom_sampler().sample(size=sample_size)

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))

    def test_custom_policy_sampler(cls):
        sample_size = 10

        state_sampler = space_sampler(space=cls.env.state_space)
        policy = ConstantPolicy(action_space=cls.env.action_space, constant=-1.0)

        @sample_fn
        def custom_sampler():
            state = next(state_sampler)
            action = policy()

            cls.env.reset(state)
            next_state, *_ = cls.env.step(action=action)

            yield (state, action, next_state)

        S = custom_sampler().sample(size=sample_size)

        X, U, Y = transpose_sample(S)

        cls.assertTrue(np.array_equiv(U, [-1.0] * sample_size))

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))
