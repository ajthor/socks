"""Quadrotor system.

This system is taken from `ARCH-COMP20 Category Report: Continuous and Hybrid Systems
with Nonlinear Dynamics <https://easychair.org/publications/open/nrdD>`_.

"""

from gym_socks.envs.spaces import Box
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

from scipy.constants import g


class QuadrotorEnv(DynamicalSystem):
    """Quadrotor system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    The quadrotor system is a high-dimensional (12D) system. The states are the
    position, velocity, and angles of the system, and the inputs are the torques on the
    angles.

    """

    # system parameters
    _gravitational_acceleration = g  # [m/s^2]
    _radius_center_mass = 0.1  # [m]
    _rotor_distance = 0.5  # [m]
    _rotor_mass = 0.1  # [kg]
    _center_mass = 1  # [kg]

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=float)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=float)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=float)

        self.state = None

        self.seed(seed=seed)

        self._compute_total_mass()
        self._compute_inertia()

    @property
    def gravitational_acceleration(self):
        return self._gravitational_acceleration

    @gravitational_acceleration.setter
    def gravitational_acceleration(self, value):
        self._gravitational_acceleration = value

    @property
    def radius_center_mass(self):
        return self._radius_center_mass

    @radius_center_mass.setter
    def radius_center_mass(self, value):
        self._radius_center_mass = value
        self._compute_inertia()

    @property
    def rotor_distance(self):
        return self._rotor_distance

    @rotor_distance.setter
    def rotor_distance(self, value):
        self._rotor_distance = value
        self._compute_inertia()

    @property
    def rotor_mass(self):
        return self._rotor_mass

    @rotor_mass.setter
    def rotor_mass(self, value):
        self._rotor_mass = value
        self._compute_total_mass()
        self._compute_inertia()

    @property
    def center_mass(self):
        return self._center_mass

    @center_mass.setter
    def center_mass(self, value):
        self._center_mass = value
        self._compute_total_mass()
        self._compute_inertia()

    def _compute_total_mass(self):
        self.total_mass = self.center_mass + 4 * self.rotor_mass

    def _compute_inertia(self):
        self.Jx = (2 / 5) * self.center_mass * (self.radius_center_mass**2) + 2 * (
            self.rotor_distance**2
        ) * self.rotor_mass
        self.Jy = self.Jx
        self.Jz = (2 / 5) * self.center_mass * (self.radius_center_mass**2) + 4 * (
            self.rotor_distance**2
        ) * self.rotor_mass

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system."""
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = state
        u1, u2, u3 = action
        w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12 = disturbance

        F = self.total_mass * self.gravitational_acceleration - 10 * (x3 - u1) + 3 * x6
        tau_phi = -(x7 - u2) - x10
        tau_psi = -(x8 - u3) - x11
        tau_theta = 0

        dx1 = (
            np.cos(x8) * np.cos(x9) * x4
            + (np.sin(x7) * np.sin(x8) * np.cos(x9) - np.cos(x7) * np.sin(x9)) * x5
            + (np.cos(x7) * np.sin(x8) * np.cos(x9) + np.sin(x7) * np.sin(x9)) * x6
        ) + w1
        dx2 = (
            np.cos(x8) * np.sin(x9) * x4
            + (np.sin(x7) * np.sin(x8) * np.sin(x9) + np.cos(x7) * np.cos(x9)) * x5
            + (np.cos(x7) * np.sin(x8) * np.sin(x9) - np.sin(x7) * np.cos(x9)) * x6
        ) + w2
        dx3 = (
            np.sin(x8) * x4
            - np.sin(x7) * np.cos(x8) * x5
            - np.cos(x7) * np.cos(x8) * x6
            + w3
        )
        dx4 = x12 * x5 - x11 * x6 - self.gravitational_acceleration * np.sin(x8) + w4
        dx5 = (
            x10 * x6
            - x12 * x4
            + self.gravitational_acceleration * np.cos(x8) * np.sin(x7)
            + w5
        )
        dx6 = (
            x11 * x4
            - x10 * x5
            + self.gravitational_acceleration * np.cos(x8) * np.cos(x7)
            - (F / self.total_mass)
        ) + w6
        dx7 = x10 + np.sin(x7) * np.tan(x8) * x11 + np.cos(x7) * np.tan(x8) * x12 + w7
        dx8 = np.cos(x7) * x11 - np.sin(x7) * x12 + w8
        dx9 = (np.sin(x7) / np.cos(x8)) * x11 + (np.cos(x7) / np.cos(x8)) * x12 + w9
        dx10 = (
            ((self.Jy - self.Jz) / self.Jx) * x11 * x12 + (1 / self.Jx) * tau_phi + w10
        )
        dx11 = (
            ((self.Jz - self.Jx) / self.Jy) * x10 * x12 + (1 / self.Jy) * tau_psi + w11
        )
        dx12 = (
            ((self.Jx - self.Jy) / self.Jz) * x10 * x11
            + (1 / self.Jz) * tau_theta
            + w12
        )

        return np.array(
            [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12],
            dtype=float,
        )
