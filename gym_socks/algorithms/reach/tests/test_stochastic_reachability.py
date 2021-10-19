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


class TestKernelSR(unittest.TestCase):
    def setUp(cls):
        cls.system = gym_socks.envs.NDIntegratorEnv(2)
        cls.policy = gym_socks.envs.policy.RandomizedPolicy(cls.system)

        cls.sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        cls.sample_space.seed(1)

    def test_kernel_sr(cls):

        S = sample(
            sampler=step_sampler(
                system=cls.system,
                policy=cls.policy,
                sample_space=cls.sample_space,
            ),
            sample_size=10,
        )
