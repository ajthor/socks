import math

from gym_basic.envs.dynamical_system_env import DynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp


class TORAEnv(DynamicalSystemEnv):
    """
    TORA (translational oscillation with a rotational actuator) system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initialize the system."""
        super().__init__(state_dim=4, action_dim=1)

        # system parameters
        self.damping_coefficient = 0.1


    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        def dynamics(t, y):
            """Dynamics for the system."""
            x1, x2, x3, x4 = y
            u = action

            dx1 = x2
            dx2 = -x1 + self.damping_coefficient * np.sin(x3)
            dx3 = x4
            dx4 = u

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
