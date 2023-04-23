import numpy as np

from gym_socks.envs.spaces import Box
from gym_socks.envs.dynamical_system import DynamicalSystem


class ProjectileEnv(DynamicalSystem):
    """Projectile system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    A projectile is a body that is projected into the air by an external force.
    It is typically modeled as a point mass, and the dynamics are governed by
    Newton's second law.

    """

    _available_renderers = ["matplotlib"]

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=float)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=float)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=float)

        self.state = None

        self.seed(seed=seed)

        self._mass = 1.0
        self._inertia_matrix = np.eye(3)

    @property
    def mass(self):
        return self._mass

    @property
    def inertia_matrix(self):
        return self._inertia_matrix

    def dynamics(self, time, state, action, disturbance):
        """Dynamics of the projectile.

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.
            disturbance: Current disturbance of the system.

        Returns:
            The derivative of the state.

        """

        x, y, z, u, v, w, phi, theta, psi, p, q, r = state
        # Aerodynamic surface deflections.
        d1, d2, d3, d4 = action

        w1, w2, w3 = disturbance

        m = self.mass
        I = self.inertia_matrix

        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        dx = (
            c_theta * c_psi * u
            + (s_phi * s_theta * c_psi - c_phi * s_psi) * v
            + (c_phi * s_theta * c_psi + s_phi * s_psi) * w
        )
        dy = (
            c_theta * s_psi * u
            + (s_phi * s_theta * s_psi + c_phi * c_psi) * v
            + (c_phi * s_theta * s_psi - s_phi * c_psi) * w
        )
        dz = -s_theta * u + s_phi * c_theta * v + c_phi * c_theta * w

        du = r * v - q * w - 9.81 * s_theta + (d1 + d2 + d3 + d4) / m + w1 / m
        dv = p * w - r * u + 9.81 * s_phi * c_theta + w2 / m
        dw = q * u - p * v + 9.81 * c_phi * c_theta + w3 / m

        dphi = p + (q * s_phi + r * c_phi) * np.tan(theta)
        dtheta = q * c_phi - r * s_phi
        dpsi = (q * s_phi + r * c_phi) / np.cos(theta)

        dp = (
            (I[1, 1] - I[2, 2]) * q * r
            + (d2 - d4) * 0.5
            + (I[1, 2] - I[2, 1]) * (q**2 - r**2)
            + p * (I[0, 0] - I[1, 1] + I[2, 2])
        ) / I[0, 0]
        dq = (
            (I[2, 2] - I[0, 0]) * p * r
            + (d3 - d1) * 0.5
            + (I[2, 0] - I[0, 2]) * (r**2 - p**2)
            + q * (I[1, 1] - I[2, 2] + I[0, 0])
        ) / I[1, 1]
        dr = (
            (I[0, 0] - I[1, 1]) * p * q
            + (d2 - d4) * 0.5
            + (I[0, 1] - I[1, 0]) * (p**2 - q**2)
            + r * (I[2, 2] - I[0, 0] + I[1, 1])
        ) / I[2, 2]

        return np.array([dx, dy, dz, du, dv, dw, dphi, dtheta, dpsi, dp, dq, dr])
