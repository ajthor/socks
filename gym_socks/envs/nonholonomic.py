"""Nonholonomic vehicle system."""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

from scipy.integrate import solve_ivp


class NonholonomicVehicleEnv(DynamicalSystem):
    """Nonholonomic vehicle system.

    A nonholonomic vehicle (car-like) is typically modeled using what are known as
    "unicycle" dynamics. It is useful for modeling vehicles which can move forward and
    backward, and incorporates a steering angle or heading. The inputs are the velocity
    and change in steering angle.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            state_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            *args,
            **kwargs
        )

    def step(self, action, time=0):
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

        return np.array([dx1, dx2, dx3], dtype=np.float32)
