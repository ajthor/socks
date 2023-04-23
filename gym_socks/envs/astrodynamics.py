import numpy as np

from gym_socks.envs.spaces import Box
from gym_socks.envs.dynamical_system import DynamicalSystem

from scipy.constants import gravitational_constant


def compute_mass_ratio(m1, m2):
    """Compute the mass ratio parameter.

    Args:
        m1: The mass of the primary body.
        m2: The mass of the secondary body.

    Returns:
        The mass ratio parameter.

    """
    return m2 / (m1 + m2)


class CRTBP(DynamicalSystem):
    """Circular Restricted Three Body Problem.

    The CRTBP is a special case of the three-body problem where one of the
    bodies has negligible mass, and the other two bodies are in circular
    orbits around their barycenter.

    """

    _sampling_time = 0.001

    _available_renderers = ["matplotlib"]

    def __init__(self, m1, m2, seed=None, *args, **kwargs):
        """Initialize the CRTBP.

        Args:
            m1: The mass of the primary body.
            m2: The mass of the secondary body.

        """
        super().__init__(*args, **kwargs)

        self.m1 = m1
        self.m2 = m2
        self.mu = compute_mass_ratio(m1, m2)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=float)

        self.state_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=float)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=float)

        self.state = None

        self.seed(seed=seed)

    def generate_disturbance(self, time, state, action):
        """Generate disturbance.

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.

        Returns:
            The disturbance.

        """

        return np.zeros(6)

    def dynamics(self, time, state, action, disturbance=None):
        """Dynamics of the CRTBP.

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.
            disturbance: Current disturbance of the system.

        Returns:
            The derivative of the state.

        """

        x1, x2, x3, x4, x5, x6 = state
        u1, u2, u3 = action
        w1, w2, w3, w4, w5, w6 = disturbance

        mu = self.mu

        r1 = [x1 + mu, x3, x5]
        r2 = [x1 - (1 - mu), x3, x5]

        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)

        dx1 = x2 + w1
        dx2 = (
            2 * x4
            + x1
            - (1 - mu) * (x1 + mu) / r1_norm**3
            - mu * (x1 - (1 - mu)) / r2_norm**3
            + u1
        ) + w2
        dx3 = x4 + w3
        dx4 = (
            -2 * x2
            + x3
            - (1 - mu) * x3 / r1_norm**3
            - mu * x3 / r2_norm**3
            + u2
            + w4
        )
        dx5 = x6 + w5
        dx6 = -(1 - mu) * x5 / r1_norm**3 - mu * x5 / r2_norm**3 + u3 + w6

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def _render_init(self, renderer, *args, **kwargs):
        """Initialize the renderer."""

        ax = renderer.canvas.figure.gca()

        # Set the limits of the plot.
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)

        # Set the labels of the plot.
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Draw the Earth.
        ax.scatter(0, 0, 0, color="blue", s=100, label="Earth")

        # Draw the Moon.
        ax.scatter(1 - self.mu, 0, 0, color="gray", s=20, label="Moon")

        # Draw the spacecraft.
        (self._artist,) = ax.plot(
            self.state[0],
            self.state[2],
            self.state[4],
            color="red",
            marker=".",
            linestyle="None",
            label="Spacecraft",
        )

        return (self._artist,)

    def _render_update(self, renderer, *args, **kwargs):
        """Update the renderer."""

        self._artist.set_data(self.state[0], self.state[2])
        self._artist.set_3d_properties(self.state[4])

        return (self._artist,)


class NBodyProblem(DynamicalSystem):
    """N-Body Problem."""

    _sampling_time = 0.001

    def __init__(self, masses, seed=None, *args, **kwargs):
        """Initialize the N-Body Problem.

        Args:
            masses: The masses of the bodies.

        """
        super().__init__(*args, **kwargs)

        self.masses = masses
        self.n = len(masses)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(6 * self.n,), dtype=float
        )

        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(6 * self.n,), dtype=float
        )
        self.action_space = Box(
            low=-np.inf, high=np.inf, shape=(3 * self.n,), dtype=float
        )

        self.state = None

        self.seed(seed=seed)

    def generate_disturbance(self, time, state, action):
        """Generate disturbance.

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.

        Returns:
            The disturbance.

        """

        return np.zeros(6 * self.n)

    def dynamics(self, time, state, action, disturbance=None):
        """Dynamics of the N-Body Problem.

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.
            disturbance: Current disturbance of the system.

        Returns:
            The derivative of the state.

        """

        x = state
        u = action
        w = disturbance

        n = self.n
        masses = self.masses
        G = gravitational_constant

        dx = np.zeros((6 * n,))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r = x[6 * i : 6 * i + 3] - x[6 * j : 6 * j + 3]
                    r_norm = np.linalg.norm(r)

                    dx[6 * i : 6 * i + 3] = x[6 * i + 3 : 6 * i + 6]
                    dx[6 * i + 3 : 6 * i + 6] = (
                        -G * masses[j] * r / r_norm**3 + u[3 * i : 3 * i + 3]
                    ) + w[6 * i : 6 * i + 6]

        return dx
