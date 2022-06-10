r"""Nonholonomic vehicle system.

.. math::
    :nowrap:

    \begin{align}
        x_{1} &= x & \dot{x}_{1} &= \dot{x} = u_{1} \sin(x_{3}) \\
        x_{2} &= y & \dot{x}_{2} &= \dot{y} = u_{1} \cos(x_{3}) \\
        x_{3} &= \theta & \dot{x}_{3} &= \dot{\theta} = u_{2}
    \end{align}

"""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

from scipy.integrate import solve_ivp


class NonholonomicVehicleEnv(DynamicalSystem):
    """Nonholonomic vehicle system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    A nonholonomic vehicle (car-like) is typically modeled using what are known as
    "unicycle" dynamics. It is useful for modeling vehicles which can move forward and
    backward, and incorporates a steering angle or heading. The inputs are the velocity
    and change in steering angle.

    """

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=float
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=float
        )

        self.state = None

        self.seed(seed=seed)

    def step(self, action, time=0):
        action = np.asarray(action, dtype=float)

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        disturbance = self.generate_disturbance(time, self.state, action)

        # solve the initial value problem
        if self._euler is True:
            next_state = self.state + self.sampling_time * self.dynamics(
                time, self.state, action, disturbance
            )
            self.state = next_state
        else:
            # solve the initial value problem
            sol = solve_ivp(
                self.dynamics,
                [time, time + self.sampling_time],
                self.state,
                args=(
                    action,
                    disturbance,
                ),
            )
            *_, self.state = sol.y.T

        # correct the angle
        if np.abs(self.state[2]) >= 2 * np.pi:
            self.state[2] %= 2 * np.pi

        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = {}

        return observation, cost, done, info

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):

        x1, x2, x3 = state
        u1, u2 = action
        w1, w2, w3 = disturbance

        dx1 = u1 * np.sin(x3) + w1
        dx2 = u1 * np.cos(x3) + w2
        dx3 = u2 + w3

        return np.array([dx1, dx2, dx3], dtype=float)
