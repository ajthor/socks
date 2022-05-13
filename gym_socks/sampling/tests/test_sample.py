import unittest
from unittest.mock import patch

import gym

from gym_socks.envs import NDIntegratorEnv
from gym_socks.policies import ConstantPolicy
from gym_socks.policies import RandomizedPolicy

# from gym_socks.sampling.sample import sample_generator
# from gym_socks.sampling.sample import sample
# from gym_socks.sampling.sample import default_sampler
# from gym_socks.sampling.sample import default_trajectory_sampler
# from gym_socks.sampling.sample import random_sampler
# from gym_socks.sampling.sample import grid_sampler
# from gym_socks.sampling.sample import repeat

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

# class TestSampleClasses(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.env = NDIntegratorEnv(2)

#     def test_callable(cls):
#         @SampleGenerator
#         def sampler():
#             return cls.env.state_space.sample()

#         result = sampler()

#         cls.assertIsInstance(result, np.ndarray)
#         cls.assertEqual(len(result), 2)

#     # def test_generator_callable(cls):
#     #     @SampleGenerator
#     #     def sampler():
#     #         yield cls.env.state_space.sample()

#     #     result = sampler()

#     #     cls.assertIsInstance(result, np.ndarray)
#     #     cls.assertEqual(len(result), 2)

#     def test_callable_with_parameters(cls):
#         @SampleGenerator
#         def sampler(val):
#             return val

#         result = sampler(5)

#         cls.assertIsInstance(result, int)
#         cls.assertEqual(result, 5)

#         result = sampler("hello")

#         cls.assertIsInstance(result, str)
#         cls.assertEqual(result, "hello")

#     # def test_generator_callable_with_parameters(cls):
#     #     val = 0

#     #     @SampleGenerator
#     #     def sampler(dummy):
#     #         val = dummy
#     #         print(dummy)
#     #         cls.assertEqual(val, 5)
#     #         yield cls.env.state_space.sample()

#     #     result = sampler(5)

#     #     cls.assertIsInstance(result, np.ndarray)
#     #     cls.assertEqual(len(result), 2)

#     def test_sample(cls):
#         @SampleGenerator
#         def sampler():
#             return cls.env.state_space.sample()

#         result = sampler.sample(10)

#         cls.assertIsInstance(result, list)
#         cls.assertEqual(len(result), 10)

#     def test_generator_sample(cls):
#         @SampleGenerator
#         def sampler():
#             yield cls.env.state_space.sample()

#         result = sampler.sample(10)

#         cls.assertIsInstance(result, list)
#         cls.assertEqual(len(result), 10)

#     def test_iterable(cls):
#         @SampleGenerator
#         def sampler():
#             return cls.env.state_space.sample()

#         result = next(sampler)

#         cls.assertIsInstance(result, np.ndarray)
#         cls.assertEqual(len(result), 2)

#     def test_generator_iterable(cls):
#         @SampleGenerator
#         def sampler():
#             yield cls.env.state_space.sample()

#         result = next(sampler)

#         cls.assertIsInstance(result, np.ndarray)
#         cls.assertEqual(len(result), 2)

#     def test_sample_generator(cls):
#         @SampleGenerator
#         def state_sampler():
#             return cls.env.state_space.sample()

#         @SampleGenerator
#         def action_sampler():
#             return cls.env.action_space.sample()

#         @SampleGenerator
#         def sampler():
#             state = state_sampler()
#             action = action_sampler()

#             cls.env.reset(state)
#             next_state, *_ = cls.env.step(action=action)

#             return (state, action, next_state)

#         result = sampler.sample(10)

#         cls.assertIsInstance(result, list)
#         cls.assertEqual(len(result), 10)


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

        state_sample_space = gym.spaces.Box(
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

        # state_sampler = repeat(
        #     grid_sampler(
        #         boxgrid(
        #             space=gym.spaces.Box(
        #                 low=-1,
        #                 high=1,
        #                 shape=cls.env.state_space.shape,
        #                 dtype=cls.env.state_space.dtype,
        #             ),
        #             resolution=3,
        #         )
        #     ),
        #     num=3,
        # )

        # action_sampler = grid_sampler(cartesian(np.linspace(-1, 1, 3)))

        # S = sample(
        #     sampler=default_sampler(
        #         state_sampler=state_sampler,
        #         action_sampler=action_sampler,
        #         env=cls.env,
        #     ),
        #     sample_size=sample_size,
        # )

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

        # state_sampler = random_sampler(sample_space=cls.env.state_space)
        # action_sampler = random_sampler(sample_space=cls.env.action_space)

        # S = sample(
        #     sampler=default_sampler(
        #         state_sampler=state_sampler,
        #         action_sampler=action_sampler,
        #         env=cls.env,
        #     ),
        #     sample_size=sample_size,
        # )

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

        # state_sampler = random_sampler(sample_space=cls.env.state_space)
        # action_sampler = random_sampler(sample_space=cls.env.action_space)

        # S = sample(
        #     sampler=default_trajectory_sampler(
        #         state_sampler=state_sampler,
        #         action_sampler=action_sampler,
        #         env=cls.env,
        #         time_horizon=5,
        #     ),
        #     sample_size=sample_size,
        # )

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

        # S = sample(
        #     sampler=custom_sampler,
        #     sample_size=sample_size,
        # )

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

        # S = sample(
        #     sampler=custom_sampler,
        #     sample_size=sample_size,
        # )

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

        # S = sample(
        #     sampler=custom_sampler,
        #     sample_size=sample_size,
        # )

        X, U, Y = transpose_sample(S)

        cls.assertTrue(np.array_equiv(U, [-1.0] * sample_size))

        cls.assertEqual(np.array(X).shape, (sample_size, 2))
        cls.assertEqual(np.array(U).shape, (sample_size, 1))
        cls.assertEqual(np.array(Y).shape, (sample_size, 2))


class TestDummy(unittest.TestCase):
    def test_dummy(cls):
        """Dummy tests."""

        cls.assertTrue(True)

        env = NDIntegratorEnv(2)

        state_sampler = space_sampler(env.state_space)
        action_sampler = space_sampler(env.action_space)

        S = transition_sampler(env, state_sampler, action_sampler).sample(size=10)
        X, U, Y = transpose_sample(S)

        A = action_sampler.sample(5)

        x1 = np.linspace(-1, 1, 3)
        x2 = np.linspace(-1, 1, 3)
        T = cartesian(x1, x2)

        # print(trajectory_sampler(env, time_horizon=2).sample(5))

        # print("samplable")
        # print(sampler.sample(5))

        # print("loop")
        # count = 0
        # for sample in sampler:
        #     print(sample)

        #     count += 1
        #     if count >= 5:
        #         break
