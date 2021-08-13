from gym_basic.envs.dynamical_system import DynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp


class NDPointMassEnv(DynamicalSystemEnv):
    """
    ND integrator system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, dim):
        """Initialize the system."""
        super().__init__(state_dim=dim, action_dim=dim)

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        return u

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)
