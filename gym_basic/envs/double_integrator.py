import math

import gym
from gym_basic.spaces.real_space import RealSpace

import numpy as np


class DoubleIntegratorEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # system parameters


        self.action_space = RealSpace(1)
        self.observation_space = RealSpace(2)

        self.state = None

    def step(self, action):
        # get the state
        x1, x2 = self.state

        # dynamics

        # set the state
        self.state = (x1, x2)

        reward = 0.0

        done = False
        info = {}
        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = np.random.rand(2);
        return np.array(self.state)

    def render(self, mode="human"):
        ...

    def close(self):
        ...


# def simulate_di(self, t, global):
#     """Dynamical equations of motion for a double integrator system."""
#
#     local_states = self._global_to_local(global_states)
#     return self._local_ds_global_ds(global_states[2], self.simulate(local_states))
#
#     def simulate(self, local_states):
#         """Simulate the system."""
#
#         x1 = local_states[0]
#         x2 = local_states[1]
