from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.dynamical_system import StochasticMixin

import numpy as np


class TORAEnv(DynamicalSystem):
    """
    TORA (translational oscillation with rotational actuation) system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            action_dspace=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            *args,
            **kwargs
        )

        # system parameters
        self._damping_coefficient = 0.1

    @property
    def damping_coefficient(self):
        return self._damping_coefficient

    @damping_coefficient.setter
    def damping_coefficient(self, value):
        self._damping_coefficient = value

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        x1, x2, x3, x4 = x

        dx1 = x2
        dx2 = -x1 + self._damping_coefficient * np.sin(x3)
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
        super().__init__(
            disturbance_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            *args,
            **kwargs
        )

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        x1, x2, x3, x4 = x
        w1, w2, w3, w4 = w

        dx1 = x2 + w1
        dx2 = -x1 + self._damping_coefficient * np.sin(x3) + w2
        dx3 = x4 + w3
        dx4 = u + w4

        return np.array([dx1, dx2, dx3, dx4], dtype=np.float32)
