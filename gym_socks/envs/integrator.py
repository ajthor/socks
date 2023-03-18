r"""ND Integrator system.

An integrator system is an extremely simple dynamical system model, typically used
to model a single variable and its higher order derivatives, where the input is
applied to the highest derivative term, and is "integrated" upwards.

.. tab-set::

    .. tab-item:: Continuous Time

        .. math::

            \dot{x} =
            \begin{bmatrix}
                0 & 1 & 0 & \cdots & 0 & 0 \\
                0 & 0 & 1 & \cdots & 0 & 0 \\
                0 & 0 & 0 & \cdots & 0 & 0 \\
                \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                0 & 0 & 0 & \cdots & 0 & 1 \\
                0 & 0 & 0 & \cdots & 0 & 0
            \end{bmatrix} x +
            \begin{bmatrix}
                0 \\
                0 \\
                0 \\
                \vdots \\
                0 \\
                1
            \end{bmatrix} u + w

    .. tab-item:: Discrete Time

        .. math::

            x_{t+1} =
            \begin{bmatrix}
                1 & T & \cdots & \frac{T^{N-2}}{(N-2)!} & \frac{T^{N-1}}{(N-1)!} \\
                0 & 1 & \cdots & \frac{T^{N-3}}{(N-3)!} & \frac{T^{N-2}}{(N-2)!} \\
                \vdots & \vdots & \ddots & \vdots & \vdots \\
                0 & 0 & \cdots & 1 & T \\
                0 & 0 & \cdots & 0 & 1
            \end{bmatrix} x_{t} +
            \begin{bmatrix}
                \frac{T^{N}}{N!} \\
                \frac{T^{N-1}}{(N-1)!} \\
                \vdots \\
                \frac{T^{2}}{2} \\
                T
            \end{bmatrix} u_{t} + w_{t}

A 2D Integrator system, for example, corresponds to the position and velocity
components of a variable, where the input is applied to the velocity and then
integrates upward to the position variable.

Tip:
    Chaining two 2D integrator systems can model a system with x/y position and
    velocity.

    .. tab-set::

        .. tab-item:: Continuous Time

            .. math::

                \dot{x} =
                \begin{bmatrix}
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 0 & 0
                \end{bmatrix} x +
                \begin{bmatrix}
                    0 & 0 \\
                    1 & 0 \\
                    0 & 0 \\
                    0 & 1
                \end{bmatrix} u + w

        .. tab-item:: Discrete Time

            .. math::

                x_{t+1} =
                \begin{bmatrix}
                    1 & T & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & T \\
                    0 & 0 & 0 & 1
                \end{bmatrix} x_{t} +
                \begin{bmatrix}
                    \frac{T^{2}}{2} & 0 \\
                    T & 0 \\
                    0 & \frac{T^{2}}{2} \\
                    0 & T
                \end{bmatrix} u_{t} + w_{t}

"""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class NDIntegratorEnv(DynamicalSystem):
    """ND integrator system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    Args:
        dim: The dimension of the integrator system.

    Example:

        >>> from gym_socks.envs import NDIntegratorEnv
        >>> env = NDIntegratorEnv(dim=2)
        >>> env.reset()
        >>> for i in range(10):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, _ = env.step(action)

    """

    def __init__(self, dim: int = 2, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=float
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=float
        )

        self.state = None

        self.seed(seed=seed)

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        _, *x = state
        return np.array([*x, *action], dtype=float) + disturbance
