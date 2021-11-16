import unittest
from unittest.mock import patch

import itertools

import gym

import gym_socks.envs

from gym_socks.sample.sample import sample
from gym_socks.sample.sample import step_sampler
from gym_socks.sample.sample import uniform_grid_step_sampler
from gym_socks.sample.sample import trajectory_sampler
from gym_socks.sample.sample import sample_generator

from gym_socks.sample.transform import transpose_sample

from gym_socks.utils.grid import make_grid
from gym_socks.utils.grid import grid_size

import numpy as np


class TestSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym_socks.envs.NDIntegratorEnv(2)
        cls.policy = gym_socks.envs.policy.RandomizedPolicy(
            action_space=cls.env.action_space
        )

        cls.sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        cls.sample_space.seed(1)

    def test_step_sampler(cls):

        S = sample(
            sampler=step_sampler(
                env=cls.env,
                policy=cls.policy,
                sample_space=cls.sample_space,
            ),
            sample_size=10,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (10, 2))
        cls.assertEqual(np.array(U).shape, (10, 1))
        cls.assertEqual(np.array(Y).shape, (10, 2))

    def test_uniform_step_sampler(cls):

        ranges = [np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)]

        S = sample(
            sampler=uniform_grid_step_sampler(
                ranges,
                env=cls.env,
                policy=cls.policy,
                sample_space=cls.sample_space,
            ),
            sample_size=25,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (25, 2))
        cls.assertEqual(np.array(U).shape, (25, 1))
        cls.assertEqual(np.array(Y).shape, (25, 2))

        groundTruth = list(itertools.product(*ranges))
        cls.assertTrue(np.all(np.equal(np.array(X), groundTruth)))

        # Sample more than ranges.
        S = sample(
            sampler=uniform_grid_step_sampler(
                ranges,
                env=cls.env,
                policy=cls.policy,
                sample_space=cls.sample_space,
            ),
            sample_size=50,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (50, 2))
        cls.assertEqual(np.array(U).shape, (50, 1))
        cls.assertEqual(np.array(Y).shape, (50, 2))

        cls.assertTrue(np.all(np.equal(np.array(X[:25]), np.array(X[25:]))))

    def test_trajectory_sampler(cls):

        S = sample(
            sampler=trajectory_sampler(
                time_horizon=16,
                env=cls.env,
                policy=cls.policy,
                sample_space=cls.sample_space,
            ),
            sample_size=10,
        )

        cls.assertEqual(np.array(S[0][0]).shape, (2,))
        cls.assertEqual(np.array(S[0][1]).shape, (16, 1))
        cls.assertEqual(np.array(S[0][2]).shape, (16, 2))

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (10, 2))
        cls.assertEqual(np.array(U).shape, (10, 16, 1))
        cls.assertEqual(np.array(Y).shape, (10, 16, 2))

        ST = gym_socks.envs.sample.reshape_trajectory_sample(S)
        X, U, Y = transpose_sample(ST)

        cls.assertEqual(np.array(X).shape, (10, 2))
        cls.assertEqual(np.array(U).shape, (10, 16))
        cls.assertEqual(np.array(Y).shape, (10, 32))

    def test_custom_sampler_as_function(cls):
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
            sample_size=10,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (10, 2))
        cls.assertEqual(np.array(U).shape, (10, 1))
        cls.assertEqual(np.array(Y).shape, (10, 2))

    def test_custom_sampler_as_generator(cls):
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
            sample_size=10,
        )

        X, U, Y = transpose_sample(S)

        cls.assertEqual(np.array(X).shape, (10, 2))
        cls.assertEqual(np.array(U).shape, (10, 1))
        cls.assertEqual(np.array(Y).shape, (10, 2))

    def test_custom_generator_sampler(cls):
        @sample_generator
        def random_sampler(sample_space: gym.Space):
            yield sample_space.sample()

        @sample_generator
        def grid_sampler(sample_space: gym.spaces.Box, resolution):
            if not sample_space.is_bounded():
                raise ValueError("space must be bounded.")

            low = sample_space.low
            high = sample_space.high

            if np.isscalar(resolution):
                xi = np.linspace(low, high, resolution, axis=-1)
                grid = make_grid(xi)

            else:
                assert (
                    np.shape(resolution) == sample_space.shape
                ), "resolution.shape doesn't match provided shape"
                xi = []
                for i, value in enumerate(resolution):
                    xi.append(np.linspace(low[i], high[i], value))

                grid = make_grid(xi)

            for item in grid:
                yield item

        @sample_generator
        def repeat(_sample_generator, num: int):
            for item in _sample_generator:
                for i in range(num):
                    yield item

        # print(np.repeat(np.linspace(-1, 1, 5), 2, axis=0))

        state_sampler = repeat(
            grid_sampler(
                sample_space=gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=cls.env.state_space.shape,
                    dtype=cls.env.state_space.dtype,
                ),
                resolution=3,
            ),
            num=3,
        )

        action_sampler = grid_sampler(
            sample_space=gym.spaces.Box(
                low=-1,
                high=1,
                shape=cls.env.action_space.shape,
                dtype=cls.env.action_space.dtype,
            ),
            resolution=3,
        )
        # action_sampler = repeat_sampler(grid=np.linspace(-1, 1, 3), num=3)

        # state_sampler.send(None)
        # action_sampler.send(None)

        @sample_generator
        def custom_sampler():
            state = next(state_sampler)
            action = next(action_sampler)

            # print(state)
            # print(action)

            cls.env.state = state
            next_state, *_ = cls.env.step(action=np.asarray(action, dtype=np.float32))

            yield (state, action, next_state)

        S = sample(
            sampler=custom_sampler,
            sample_size=25,
        )

        X, U, Y = transpose_sample(S)

        print(np.array(X))
        print(np.array(U))

        # cls.assertEqual(np.array(X).shape, (10, 2))
        # cls.assertEqual(np.array(U).shape, (10, 1))
        # cls.assertEqual(np.array(Y).shape, (10, 2))
