from gym_basic.envs.dynamical_system import DynamicalSystem
from gym_basic.envs.dynamical_system import StochasticMixin

import gym

import numpy as np


class CWHBase(DynamicalSystem):
    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(*args, **kwargs)

        self.time_horizon = 600  # [s]
        self.sampling_time = 20  # [s]

        # system parameters
        self._orbital_radius = 850 + 6378.1  # [m]
        self._gravitational_constant = 6.673e-11
        self._celestial_body_mass = 5.9472e24  # [kg]
        self._chief_mass = 300  # [kg]

        self.compute_mu()
        self.compute_angular_velocity()

    @property
    def orbital_radius(self):
        return self._orbital_radius

    @orbital_radius.setter
    def orbital_radius(self, value):
        self._orbital_radius = value
        self.compute_angular_velocity()

    @property
    def gravitational_constant(self):
        return self._gravitational_constant

    @gravitational_constant.setter
    def gravitational_constant(self, value):
        self._gravitational_constant = value
        self.compute_mu()

    @property
    def celestial_body_mass(self):
        return self._celestial_body_mass

    @celestial_body_mass.setter
    def celestial_body_mass(self, value):
        self._celestial_body_mass = value
        self.compute_mu()

    @property
    def chief_mass(self):
        return self._chief_mass

    @chief_mass.setter
    def chief_mass(self, value):
        self._chief_mass = value

    def compute_mu(self):
        self.mu = self.gravitational_constant * self.celestial_body_mass / 1e9
        self.compute_angular_velocity()

    def compute_angular_velocity(self):
        self.angular_velocity = np.sqrt(self.mu / (self.orbital_radius ** 3))
        self.compute_transition_matrix()

    def compute_transition_matrix(self):
        pass


class CWH4DEnv(CWHBase):
    """
    4D Clohessy-Wiltshire-Hill system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=4, action_dim=2, *args, **kwargs)

        self.action_space = gym.spaces.Box(
            low=-0.01, high=0.01, shape=(2,), dtype=np.float32
        )

        self.compute_transition_matrix()

    def compute_transition_matrix(self):
        n = self.angular_velocity
        nt = n * self.sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        self.transition_matrix = [
            [4 - 3 * cos_nt, 0, (1 / n) * sin_nt, (2 / n) * (1 - cos_nt)],
            [
                6 * (sin_nt - nt),
                1,
                -(2 / n) * (1 - cos_nt),
                (1 / n) * (4 * sin_nt - 3 * nt),
            ],
            [3 * n * sin_nt, 0, cos_nt, 2 * sin_nt],
            [-6 * n * (1 - cos_nt), 0, -2 * sin_nt, 4 * cos_nt - 3],
        ]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # use closed-form solution
        self.state = np.matmul(self.transition_matrix, self.state.T)

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        x1, x2, x3, x4 = x
        u1, u2 = u

        dx1 = x1
        dx2 = x2
        dx3 = (
            3 * (self.angular_velocity ** 2) * x1
            + 2 * self.angular_velocity * x4
            + (u1 / self.chief_mass)
        )
        dx4 = -2 * self.angular_velocity * x3 + (u2 / self.chief_mass)

        return np.array([dx1, dx2, dx3, dx4], dtype=np.float32)

    def reset(self):
        self.state = self.np_random.uniform(
            low=[-1, -1, 0, 0], high=[1, 0, 0, 0], size=self.observation_space.shape
        )
        return np.array(self.state)


class CWH6DEnv(CWHBase):
    """
    6D Clohessy-Wiltshire-Hill system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=6, action_dim=3, *args, **kwargs)

        self.action_space = gym.spaces.Box(
            low=-0.01, high=0.01, shape=(3,), dtype=np.float32
        )

        self.compute_transition_matrix()

    def compute_transition_matrix(self):
        n = self.angular_velocity
        nt = n * self.sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        self.transition_matrix = [
            [4 - 3 * cos_nt, 0, 0, (1 / n) * sin_nt, (2 / n) * (1 - cos_nt), 0],
            [
                6 * (sin_nt - nt),
                1,
                0,
                -(2 / n) * (1 - cos_nt),
                (1 / n) * (4 * sin_nt - 3 * nt),
                0,
            ],
            [0, 0, cos_nt, 0, 0, (1 / n) * sin_nt],
            [3 * n * sin_nt, 0, 0, cos_nt, 2 * sin_nt, 0],
            [-6 * n * (1 - cos_nt), 0, 0, -2 * sin_nt, 4 * cos_nt - 3, 0],
            [0, 0, -n * sin_nt, 0, 0, cos_nt],
        ]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # use closed-form solution
        self.state = np.matmul(self.transition_matrix, self.state.T)

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def dynamics(self, t, x, u):
        """Dynamics for the system."""
        x1, x2, x3, x4, x5, x6 = x
        u1, u2, u3 = u

        dx1 = x1
        dx2 = x2
        dx3 = x3
        dx4 = (
            3 * (self.angular_velocity ** 2) * x1
            + 2 * self.angular_velocity * x5
            + (u1 / self.chief_mass)
        )
        dx5 = -2 * self.angular_velocity * x4 + (u2 / self.chief_mass)
        dx6 = -(self.angular_velocity ** 2) * x3 + (u3 / self.chief_mass)

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=np.float32)

    def reset(self):
        self.state = self.np_random.uniform(
            low=[-1, -1, 0, 0, 0, 0],
            high=[1, 0, 0, 0, 0, 0],
            size=self.observation_space.shape,
        )
        return np.array(self.state)


class StochasticCWH4DEnv(StochasticMixin, CWH4DEnv):
    """
    Stochastic 4D Clohessy-Wiltshire-Hill system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=4, action_dim=2, disturbance_dim=4, *args, **kwargs)

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        x1, x2, x3, x4 = x
        u1, u2 = u
        w1, w2, w3, w4 = w

        dx1 = x1 + w1
        dx2 = x2 + w2
        dx3 = (
            3 * (self.angular_velocity ** 2) * x1
            + 2 * self.angular_velocity * x4
            + (u1 / self.chief_mass)
            + w3
        )
        dx4 = -2 * self.angular_velocity * x3 + (u2 / self.chief_mass) + w4

        return np.array([dx1, dx2, dx3, dx4], dtype=np.float32)

    def sample_disturbance(self):
        w = self.np_random.standard_normal(size=self.disturbance_space.shape)
        w = np.multiply([1e-4, 1e-4, 5e-8, 5e-8], w)
        return np.array(w)


class StochasticCWH6DEnv(StochasticMixin, CWH6DEnv):
    """
    Stochastic 6D Clohessy-Wiltshire-Hill system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the system."""
        super().__init__(state_dim=6, action_dim=3, disturbance_dim=6, *args, **kwargs)

    def dynamics(self, t, x, u, w):
        """Dynamics for the system."""
        x1, x2, x3, x4, x5, x6 = x
        u1, u2, u3 = u
        w1, w2, w3, w4, w5, w6 = w

        dx1 = x1 + w1
        dx2 = x2 + w2
        dx3 = x3 + w3
        dx4 = (
            3 * (self.angular_velocity ** 2) * x1
            + 2 * self.angular_velocity * x5
            + (u1 / self.chief_mass)
            + w4
        )
        dx5 = -2 * self.angular_velocity * x4 + (u2 / self.chief_mass) + w5
        dx6 = -(self.angular_velocity ** 2) * x3 + (u3 / self.chief_mass) + w6

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=np.float32)

    def sample_disturbance(self):
        w = self.np_random.standard_normal(size=self.disturbance_space.shape)
        w = np.multiply([1e-4, 1e-4, 1e-4, 5e-8, 5e-8, 5e-8], w)
        return np.array(w)
