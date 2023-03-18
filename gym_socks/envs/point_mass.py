"""ND point mass system."""

from gym_socks.envs.spaces import Box
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class NDPointMassEnv(DynamicalSystem):
    """ND point mass system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    A point mass is a very simple system in which the inputs apply directly to the state
    variables. Thus, it is essentially a representation of a particle using Newton's
    equations `F = mA`.

    Args:
        dim: Dimensionality of the point mass system.
        mass: Mass of the particle.

    """

    _mass = 1

    def __init__(self, dim, seed=None, *args, **kwargs):
        """Initialize the system."""
        super().__init__(*args, **kwargs)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=float
        )
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=float)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=float)

        self.state = None

        self.seed(seed=seed)

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = value

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        return (1 / self.mass) * np.array([*action], dtype=float) + disturbance
