from gym_basic.envs.dynamical_system import DynamicalSystemEnv
from gym_basic.envs.dynamical_system import StochasticDynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp


class NDIntegratorEnv(DynamicalSystemEnv):
    """
    ND integrator system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, state_dim):
        """Initialize the system."""
        super().__init__(state_dim=state_dim, action_dim=1)

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        _, *x = x
        return np.array([*x, *u], dtype=np.float32)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)


class DoubleIntegratorEnv(NDIntegratorEnv):
    """
    2D integrator system.
    """

    def __init__(self):
        """Initialize the system."""
        super().__init__(state_dim=2)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)


class StochasticNDIntegratorEnv(StochasticDynamicalSystemEnv):
    """
    Stochastic ND integrator system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, state_dim):
        """Initialize the system."""
        super().__init__(state_dim=state_dim, action_dim=1, disturbance_dim=state_dim)

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        _, *x = x
        return np.array([*x, *u], dtype=np.float32) + w

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)
