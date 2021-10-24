"""ND Integrator system."""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class NDIntegratorEnv(DynamicalSystem):
    """ND integrator system.

    An integrator system is an extremely simple dynamical system model, typically used
    to model a single variable and its higher order derivatives, where the input is
    applied to the highest derivative term, and is "integrated" upwards.

    A 2D Integrator system, for example, corresponds to the position and velocity
    components of a variable, where the input is applied to the velocity and then
    integrates upward to the position variable. Chaining two 2D integrator systems can
    model a system with x/y position and velocity.

    Args:
        dim: The dimension of the integrator system.

    Example:

        >>> from gym_socks.envs import NDIntegratorEnv
        >>> env = NDIntegratorEnv(dim=2)
        >>> env.reset()
        >>> num_steps = env.num_time_steps
        >>> for i in range(num_steps):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, _ = env.step(action)
        >>> env.close()

    """

    def __init__(self, dim: int = 1, *args, **kwargs):
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
        _, *x = state
        return np.array([*x, *action], dtype=np.float32) + disturbance
