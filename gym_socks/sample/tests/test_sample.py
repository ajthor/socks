import unittest
from unittest.mock import patch

import gym

from gym_socks.envs import NDIntegratorEnv
from gym_socks.envs.policy import ConstantPolicy
from gym_socks.envs.policy import RandomizedPolicy

from gym_socks.sample.sample import sample_generator
from gym_socks.sample.sample import sample
from gym_socks.sample.sample import default_sampler
from gym_socks.sample.sample import default_trajectory_sampler
from gym_socks.sample.sample import random_sampler
from gym_socks.sample.sample import grid_sampler
from gym_socks.sample.sample import repeat

from gym_socks.sample.transform import transpose_sample
from gym_socks.sample.transform import flatten_sample

from gym_socks.utils.grid import make_grid_from_ranges
from gym_socks.utils.grid import make_grid_from_space

import numpy as np


class TestSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NDIntegratorEnv(2)
        cls.policy = RandomizedPolicy(action_space=cls.env.action_space)

        cls.sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        cls.sample_space.seed(1)

    def test_random_sampler(cls):
        """Random sampler should output correct sample size."""

        sample_size = 10

        state_sampler = random_sampler(sample_space=cls.env.state_space)
        action_sampler = random_sampler(sample_space=cls.env.action_space)

        S = sample(
            sampler=default_sampler(
                state_sampler=state_sampler,
                action_sampler=action_sampler,
                env=cls.env,
            ),
            sample_size=sample_size,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.shape(X), (sample_size, 2))
        cls.assertEqual(np.shape(U), (sample_size, 1))
        cls.assertEqual(np.shape(Y), (sample_size, 2))

    def test_grid_sampler(cls):
        """Grid sampler should output items on a grid of the correct size."""

        # print(np.repeat(np.linspace(-1, 1, 5), 2, axis=0))

        sample_size = 27

        state_sampler = repeat(
            grid_sampler(
                make_grid_from_space(
                    sample_space=gym.spaces.Box(
                        low=-1,
                        high=1,
                        shape=cls.env.state_space.shape,
                        dtype=cls.env.state_space.dtype,
                    ),
                    resolution=3,
                )
            ),
            num=3,
        )

        action_sampler = grid_sampler(make_grid_from_ranges([np.linspace(-1, 1, 3)]))

        S = sample(
            sampler=default_sampler(
                state_sampler=state_sampler,
                action_sampler=action_sampler,
                env=cls.env,
            ),
            sample_size=sample_size,
        )

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

        state_sampler = random_sampler(sample_space=cls.env.state_space)
        action_sampler = random_sampler(sample_space=cls.env.action_space)

        S = sample(
            sampler=default_sampler(
                state_sampler=state_sampler,
                action_sampler=action_sampler,
                env=cls.env,
            ),
            sample_size=sample_size,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))

    def test_trajectory_sampler(cls):
        """Default trajectory sampler should output sample of the correct size."""

        sample_size = 10
        time_horizon = 5

        state_sampler = random_sampler(sample_space=cls.env.state_space)
        action_sampler = random_sampler(sample_space=cls.env.action_space)

        S = sample(
            sampler=default_trajectory_sampler(
                state_sampler=state_sampler,
                action_sampler=action_sampler,
                env=cls.env,
                time_horizon=5,
            ),
            sample_size=sample_size,
        )

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

        @sample_generator
        def custom_sampler():
            state = cls.sample_space.sample()
            action = cls.env.action_space.sample()

            cls.env.state = state
            next_state, cost, done, _ = cls.env.step(action=action)

            return (state, action, next_state)

        #   ^^^^^^

        S = sample(
            sampler=custom_sampler,
            sample_size=sample_size,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))

    def test_custom_sampler_as_generator(cls):

        sample_size = 10

        @sample_generator
        def custom_sampler():
            state = cls.sample_space.sample()
            action = cls.env.action_space.sample()

            cls.env.state = state
            next_state, cost, done, _ = cls.env.step(action=action)

            yield (state, action, next_state)

        #   ^^^^^

        S = sample(
            sampler=custom_sampler,
            sample_size=sample_size,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))

    def test_custom_policy_sampler(cls):

        sample_size = 10

        state_sampler = random_sampler(sample_space=cls.env.state_space)
        policy = ConstantPolicy(action_space=cls.env.action_space, constant=-1.0)

        @sample_generator
        def custom_sampler():
            state = next(state_sampler)
            action = policy()

            cls.env.state = state
            next_state, *_ = cls.env.step(action=action)

            yield (state, action, next_state)

        S = sample(
            sampler=custom_sampler,
            sample_size=sample_size,
        )

        X, U, Y = transpose_sample(S)

        cls.assertTrue(np.array_equiv(U, [-1.0] * sample_size))

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))
