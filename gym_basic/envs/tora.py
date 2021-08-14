from gym_basic.envs.dynamical_system import DynamicalSystem
from gym_basic.envs.dynamical_system import StochasticMixin

import numpy as np

class TORAEnv(DynamicalSystem):
    """
    TORA (translational oscillation with rotational actuation) system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=4, action_dim=1, *args, **kwargs)

        # system parameters
        self.damping_coefficient = 0.1

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        x1, x2, x3, x4 = x

        dx1 = x2
        dx2 = -x1 + self.damping_coefficient * np.sin(x3)
        dx3 = x4
        dx4 = u

        return np.array([dx1, dx2, dx3, dx4], dtype=np.float32)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.observation_space.shape
        )
        return np.array(self.state)


class StochasticTORAEnv(StochasticMixin, TORAEnv):
    """
    Stochastic TORA system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=4, action_dim=1, disturbance_dim=4, *args, **kwargs)

        # system parameters
        self.damping_coefficient = 0.1

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        x1, x2, x3, x4 = x
        w1, w2, w3, w4 = w

        dx1 = x2 + w1
        dx2 = -x1 + self.damping_coefficient * np.sin(x3) + w2
        dx3 = x4 + w3
        dx4 = u + w4

        return np.array([dx1, dx2, dx3, dx4], dtype=np.float32)
