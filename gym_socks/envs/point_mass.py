"""ND point mass system."""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class NDPointMassEnv(DynamicalSystem):
    """ND point mass system.

    A point mass is a very simple system in which the inputs apply directly to the state
    variables. Thus, it is essentially a representation of a particle using Newton's
    equations `F = mA`.

    Args:
        dim: Dimensionality of the point mass system.
        mass: Mass of the particle.

    """

    def __init__(self, dim, seed=None, mass=1, *args, **kwargs):
        """Initialize the system."""
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

        self.state = None

        self.seed(seed=seed)

        self.mass = mass

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        return (1 / self.mass) * np.array([*action], dtype=np.float32) + disturbance
