from gym_basic.envs.dynamical_system import DynamicalSystem
from gym_basic.envs.dynamical_system import StochasticMixin

import numpy as np

class NDPointMassEnv(DynamicalSystem):
    """
    ND point mass system.
    """

    def __init__(self, dim, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=dim, action_dim=dim, *args, **kwargs)

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        return u

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)


class StochasticNDPointMassEnv(StochasticMixin, NDPointMassEnv):
    """
    Stochastic ND point mass system.
    """

    def __init__(self, dim, *args, **kwargs):
        """Initialize the system."""
        super().__init__(dim=dim, disturbance_dim=dim, *args, **kwargs)

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        return u + w
