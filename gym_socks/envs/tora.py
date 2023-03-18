"""TORA (translational oscillation with rotational actuation) system."""

from gym_socks.envs.spaces import Box
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class TORAEnv(DynamicalSystem):
    """TORA (translational oscillation with rotational actuation) system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    The TORA system is a mass with an attached pendulum (rotational oscillator) attached via a spring to a surface. This system is useful for modeling a variant of the pendulum system or a cart-pole system. The input is to the pendulum.

    """

    # system parameters
    _damping_coefficient = 0.1

    def __init__(self, seed=None):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=float)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=float)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)

        self.state = None

        self.seed(seed=seed)

    @property
    def damping_coefficient(self):
        return self._damping_coefficient

    @damping_coefficient.setter
    def damping_coefficient(self, value):
        self._damping_coefficient = value

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        x1, x2, x3, x4 = state
        w1, w2, w3, w4 = disturbance

        dx1 = x2 + w1
        dx2 = -x1 + self.damping_coefficient * np.sin(x3) + w2
        dx3 = x4 + w3
        dx4 = action + w4

        return np.array([dx1, dx2, dx3, *dx4], dtype=float)
