from systems.envs.dynamical_system import DynamicalSystem
from systems.envs.dynamical_system import StochasticMixin

import numpy as np

class NonholonomicVehicleEnv(DynamicalSystem):
    """
    Nonholonomic vehicle system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=3, action_dim=2, *args, **kwargs)

    def dynamics(self, t, x, u):
        """Dynamics for the system."""

        x1, x2, x3 = x
        u1, u2 = u

        dx1 = u1 * np.sin(x3)
        dx2 = u1 * np.cos(x3)
        dx3 = u2

        return np.array([dx1, dx2, dx3], dtype=np.float32)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)


class StochasticNonholonomicVehicleEnv(StochasticMixin, NonholonomicVehicleEnv):
    """
    Stochastic nonholonomic vehicle.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=3, action_dim=2, disturbance_dim=3, *args, **kwargs)

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""

        x1, x2, x3 = x
        u1, u2 = u
        w1, w2, w3 = w

        dx1 = u1 * np.sin(x3) + w1
        dx2 = u1 * np.cos(x3) + w2
        dx3 = u2 + w3

        return np.array([dx1, dx2, dx3], dtype=np.float32)
