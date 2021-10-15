from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.dynamical_system import StochasticMixin

import gym

import numpy as np
from scipy.integrate import solve_ivp


class NonholonomicVehicleEnv(DynamicalSystem):
    """
    Nonholonomic vehicle system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            *args,
            **kwargs
        )

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # solve the initial value problem
        sol = solve_ivp(
            self.dynamics,
            [0, self.sampling_time],
            self.state,
            args=(action,),
        )
        *_, self.state = sol.y.T

        # correct the angle
        if np.abs(self.state[2]) >= 2 * np.pi:
            self.state[2] %= 2 * np.pi

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def dynamics(self, t, x, u):
        """Dynamics for the system."""

        x1, x2, x3 = x
        u1, u2 = u

        dx1 = u1 * np.sin(x3)
        dx2 = u1 * np.cos(x3)
        dx3 = u2

        return np.array([dx1, dx2, dx3], dtype=np.float32)

    def reset(self):
        self.state = self.np_random.uniform(
            low=-1, high=1, size=self.observation_space.shape
        )
        return np.array(self.state)


class StochasticNonholonomicVehicleEnv(StochasticMixin, NonholonomicVehicleEnv):
    """
    Stochastic nonholonomic vehicle.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(
            disturbance_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            *args,
            **kwargs
        )

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        disturbance = self.sample_disturbance()

        # solve the initial value problem
        sol = solve_ivp(
            self.dynamics,
            [0, self.sampling_time],
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

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""

        x1, x2, x3 = x
        u1, u2 = u
        w1, w2, w3 = w

        dx1 = u1 * np.sin(x3) + w1
        dx2 = u1 * np.cos(x3) + w2
        dx3 = u2 + w3

        return np.array([dx1, dx2, dx3], dtype=np.float32)

    def sample_disturbance(self):
        w = self.np_random.standard_normal(size=self.disturbance_space.shape)
        return 1e-2 * np.array(w)
