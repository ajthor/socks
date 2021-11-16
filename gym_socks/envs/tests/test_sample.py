import unittest
from unittest.mock import patch

import itertools

import gym

import gym_socks.envs

from gym_socks.envs.sample import (
    sample,
    step_sampler,
    uniform_grid_step_sampler,
    trajectory_sampler,
    sample_generator,
    transpose_sample,
)

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
