from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.dynamical_system import StochasticMixin

import numpy as np
from scipy.integrate import solve_ivp


class CartPoleEnv(DynamicalSystem):
    """
    Cart-pole system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=3, action_dim=1, *args, **kwargs)

        # system parameters
        self.damping_coefficient = 0.1

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        x1, x2, x3 = x

        # dx1 = x2
        # dx2 = -x1 + self.damping_coefficient * np.sin(x3)
        # dx3 = x4
        # dx4 = u

        return (dx1, dx2, dx3)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.observation_space.shape
        )
        return np.array(self.state)
