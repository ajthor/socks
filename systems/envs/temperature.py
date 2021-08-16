from functools import reduce

from systems.envs.dynamical_system import DynamicalSystem
from systems.envs.dynamical_system import StochasticMixin

import gym

import numpy as np


class TemperatureRegEnv(DynamicalSystem):
    """
    Temperature regulation system.
    """

    def __init__(self, dim, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=dim, action_dim=1, *args, **kwargs)

        self.ambient_temperature = 5

        self.ambient_loss = self.np_random.random(size=(dim,))
        self.room_loss = np.multiply(
            self.np_random.random(size=(dim, dim)), (-np.eye(dim) + 1)
        )
        self.heat_rate = self.np_random.random(size=(dim,))

        self.dim = dim  # (ON1, ON2, ..., ONr), r = dim

        self.discrete_observation_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(2) for i in range(self.dim))
        )
        # self.continuous_observation_space = gym.space.Box(low=,high=,shape=(dim,),dtype=np.float32)
        self.continuous_observation_space = self.observation_space

        self.observation_space = gym.spaces.Tuple(
            (self.discrete_observation_space, self.continuous_observation_space)
        )

        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(2) for i in range(self.dim))
        )

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        q, z = self.state

        q = action

        # There is certainly a more elegant way to write this.
        for i in range(self.dim):
            a = 0
            for j in range(self.dim):
                a += self.room_loss[i, j] * (z[j] - z[i])

            z[i] = (
                z[i]
                + a
                + self.ambient_loss[i] * (self.ambient_temperature - z[i])
                + self.heat_rate[i] * np.float32(q[i])
            )

        self.state = (q, np.array(z, dtype=np.float32))

        reward = self.cost(action)

        done = False
        info = {}

        return self.state, reward, done, info

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        ...

    def reset(self):
        continuous_state = self.np_random.uniform(
            low=0, high=35, size=self.continuous_observation_space.shape
        )
        self.state = (tuple(0 for i in range(self.dim)), continuous_state)
        return self.state


class StochasticTemperatureRegEnv(StochasticMixin, TemperatureRegEnv):
    """
    Stochastic temperature regulation system.
    """

    def __init__(self, dim, *args, **kwargs):
        """Initialize the system."""
        super().__init__(dim=dim, disturbance_dim=dim, *args, **kwargs)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        q, z = self.state

        q = action

        w = self.sample_disturbance()

        # There is certainly a more elegant way to write this.
        for i in range(self.dim):
            a = 0
            for j in range(self.dim):
                a += self.room_loss[i, j] * (z[j] - z[i])

            z[i] = (
                z[i]
                + a
                + self.ambient_loss[i] * (self.ambient_temperature - z[i])
                + self.heat_rate[i] * np.float32(q[i])
                + w[i]
            )

        self.state = (q, np.array(z, dtype=np.float32))

        reward = self.cost(action)

        done = False
        info = {}

        return self.state, reward, done, info

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        ...

    def sample_disturbance(self):
        w = self.np_random.standard_normal(size=self.disturbance_space.shape)
        return np.array(w)
