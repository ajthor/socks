import math

from gym_basic.envs.dynamical_system_env import DynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp


class CWH4DEnv(DynamicalSystemEnv):
    """
    Clohessy-Wiltshire-Hill system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initialize the system."""
        super().__init__(state_dim=4, action_dim=2)

        # system parameters
        self.time_horizon = 600  # [s]
        self.sampling_time = 20  # [s]
        self.orbital_radius = 850 + 6378.1  # [m]
        self.gravitational_constant = 6.673e-11
        self.celestial_body_mass = 5.9472e24  # [kg]
        self.chief_mass = 300  # [kg]

        self.mu = self.gravitational_constant * self.celestial_body_mass / 1e9
        self.angular_velocity = np.sqrt(self.mu / (self.orbital_radius ** 3))

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        def dynamics(t, y):
            """Dynamics for the system."""
            x1, x2, x3, x4 = y
            u1, u2 = action

            dx1 = x1
            dx2 = x2
            dx3 = (
                3 * (self.angular_velocity ** 2) * x1
                + 2 * self.angular_velocity * x4
                + u1
            )
            dx4 = -2 * self.angular_velocity * x3 + u2

            return (dx1, dx2, dx3, dx4)

        # solve the initial value problem
        sol = solve_ivp(dynamics, [0, self.sampling_time], self.state)
        *y, self.state = sol.y.T

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.observation_space.shape
        )
        return np.array(self.state)


class CWH6DEnv(DynamicalSystemEnv):
    """
    Clohessy-Wiltshire-Hill system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initialize the system."""
        super().__init__(state_dim=6, action_dim=3)

        # system parameters
        self.time_horizon = 600  # [s]
        self.sampling_time = 20  # [s]
        self.orbital_radius = 850 + 6378.1  # [m]
        self.gravitational_constant = 6.673e-11
        self.celestial_body_mass = 5.9472e24  # [kg]
        self.chief_mass = 300  # [kg]

        self.mu = self.gravitational_constant * self.celestial_body_mass / 1e9
        self.angular_velocity = np.sqrt(self.mu / (self.orbital_radius ** 3))

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        def dynamics(t, y):
            """Dynamics for the system."""
            x1, x2, x3, x4, x5, x6 = y
            u1, u2, u3 = action

            dx1 = x1
            dx2 = x2
            dx3 = x3
            dx4 = (
                3 * (self.angular_velocity ** 2) * x1
                + 2 * self.angular_velocity * x5
                + u1
            )
            dx5 = -2 * self.angular_velocity * x4 + u2
            dx6 = -(self.angular_velocity ** 2) * x3 + u3

            return (dx1, dx2, dx3, dx4, dx5, dx6)

        # solve the initial value problem
        sol = solve_ivp(dynamics, [0, self.sampling_time], self.state)
        *y, self.state = sol.y.T

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.observation_space.shape
        )
        return np.array(self.state)
