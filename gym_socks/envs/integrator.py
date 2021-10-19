import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class NDIntegratorEnv(DynamicalSystem):
    """
    ND integrator system.
    """

    def __init__(self, dim, *args, **kwargs):
        """Initialize the system."""
        super().__init__(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
            ),
            state_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            *args,
            **kwargs
        )

        self.time_horizon = 4
        self.sampling_time = 0.25

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system."""
        _, *x = state
        return np.array([*x, *action], dtype=np.float32) + disturbance

    def reset(self):
        self.state = self.state_space.sample()
