import math

from gym_basic.envs.dynamical_system import DynamicalSystemEnv

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

    def dynamics(self, t, x, u):
        """Dynamics for the system."""

        x1, x2, x3 = x
        u1, u2 = u

        dx1 = u1 * np.sin(x3)
        dx2 = u1 * np.cos(x3)
        dx3 = u2

        return (dx1, dx2, dx3)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)
