import math

from gym_basic.envs.dynamical_system_env import DynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp

class NonholonomicVehicleEnv(DynamicalSystemEnv):
    """
    Nonholonomic vehicle system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initialize the system."""
        super().__init__(state_dim=3, action_dim=2)



    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        def dynamics(t, y):
            """Dynamics for the system."""

            x1, x2, x3 = y
            u1, u2 = action

            dx1 = u1 * np.sin(x3)
            dx2 = u1 * np.cos(x3)
            dx3 = u2

            return (dx1, dx2, dx3)

        # solve the initial value problem
        sol = solve_ivp(dynamics, [0, self.sampling_time], self.state)
        *y, self.state = sol.y.T

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info



    def reset(self):
        self.state = self.np_random.uniform(low=-1, high=1, size=self.observation_space.shape)
        return np.array(self.state)
