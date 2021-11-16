"""Planar quadrotor system."""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

from scipy.constants import g
from scipy.integrate import solve_ivp


class PlanarQuadrotorEnv(DynamicalSystem):
    """Planar quadrotor system.

    A planar quadrotor is quadrotor restricted to two dimensions. Similar to the OpenAI gym lunar lander benchmark, the planar quadrotor is a bar with two independent rotors at either end. Inputs are the trust of the rotors, and apply a torque to the bar. The system is also subject to gravitational forces.

    """

    # system parameters
    _gravitational_acceleration = g  # [m/s^2]
    _rotor_distance = 2  # [m]
    _total_mass = 5  # [kg]
    _inertia = 2

    def __init__(self, seed=None, *args, **kwargs):
        """Initialize the system."""
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.state = None

        self.seed(seed=seed)

    @property
    def gravitational_acceleration(self):
        return self._gravitational_acceleration

    @gravitational_acceleration.setter
    def gravitational_acceleration(self, value):
        self._gravitational_acceleration = value

    @property
    def rotor_distance(self):
        return self._rotor_distance

    @rotor_distance.setter
    def rotor_distance(self, value):
        self._rotor_distance = value

    @property
    def total_mass(self):
        return self._total_mass

    @total_mass.setter
    def total_mass(self, value):
        self._total_mass = value

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, value):
        self._inertia = value

    def step(self, action, time=0):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        disturbance = self.generate_disturbance(time, self.state, action)

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
        if np.abs(self.state[5]) >= 2 * np.pi:
            self.state[5] %= 2 * np.pi

        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = {}

        return observation, cost, done, info

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        w = np.multiply([1e-3, 1e-5, 1e-3, 1e-5, 1e-3, 1e-5], w)
        return np.array(w)

    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system."""

        x1, x2, x3, x4, x5, x6 = state
        u1, u2 = action
        w1, w2, w3, w4, w5, w6 = disturbance

        M = u1 + u2

        dx1 = x2 + w1
        dx2 = -(M * np.sin(x5)) / self.total_mass + w2
        dx3 = x4 + w3
        dx4 = (M * np.cos(x5)) / self.total_mass - self.gravitational_acceleration + w4
        dx5 = x5 + w5
        dx6 = (self.rotor_distance * (u1 - u2)) / self.inertia + w6

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=np.float32)
