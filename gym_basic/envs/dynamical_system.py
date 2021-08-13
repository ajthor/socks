from abc import ABC, abstractmethod

import gym
from gym.utils import seeding

import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystemEnv(gym.Env, ABC):
    """
    DynamicalSystemEnv

    Base class for dynamical system models.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, state_dim, action_dim):
        """Initialize the dynamical system."""

        # time parameters
        self.time_horizon = 1
        self.sampling_time = 0.1

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )

        self.seed()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        def dynamics(t, y):
            """Dynamics for the system."""
            return action

        # solve the initial value problem
        sol = solve_ivp(dynamics, [0, self.sampling_time], self.state)
        *y, self.state = sol.y.T

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.np_random.uniform(low=0, high=1, size=(self.state_dim,))
        return np.array(self.state)

    def render(self, mode="human"):
        ...

    def close(self):
        ...

    def cost(self, action):
        """Cost function for the system."""
        return 0.0
