import unittest
from unittest.mock import Base, patch

import gym

import gym_socks.envs
from gym_socks.envs.core import BaseDynamicalObject
from gym_socks.envs.policy import RandomizedPolicy
from gym_socks.envs.world import World
from gym_socks.envs.obstacle import BaseObstacle

import numpy as np


class DummyObstacle(BaseObstacle):
    def __init__(self, identifier: str) -> None:
        self._id = identifier

    def intersects(self, state):
        return

    def step(self, time=0):
        # print(f"Hello, {self._id}.")
        pass

    def reset(self):
        pass

    def render(self):
        raise NotImplementedError


class TestWorld(unittest.TestCase):
    def test_world(cls):
        """Test world iteration."""

        env = gym_socks.envs.NDIntegratorEnv()
        policy = RandomizedPolicy(action_space=env.action_space)

        obstacle1 = DummyObstacle(identifier="Larry")
        obstacle2 = DummyObstacle(identifier="Curly")
        obstacle3 = DummyObstacle(identifier="Moe")

        world = World()
        world.time_horizon = 5

        world += [obstacle1, obstacle2, obstacle3]

        world.reset()
        env.reset()

        for t in range(world.time_horizon):
            world.step()

            action = policy(time=t, state=[env.state])
            env.step(time=t, action=action)
