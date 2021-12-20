from abc import abstractmethod

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

from scipy.constants import gravitational_constant


class BaseCWH(object):
    """CWH base class.

    This class holds the shared parameters for the CWH systems, which include:

    * orbital radius
    * gravitational constant
    * celestial body mass
    * chief mass

    And provides methods to compute:

    * graviational parameter (:math:`\mu`)
    * angular velocity (n)

    """

    time_horizon = 600  # [s]
    sampling_time = 20  # [s]

    # system parameters
    _orbital_radius = 850 + 6378.1  # [m]
    _gravitational_constant = gravitational_constant
    _celestial_body_mass = 5.9472e24  # [kg]
    _chief_mass = 300  # [kg]

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

        self.state_matrix = self.compute_state_matrix(sampling_time=self.sampling_time)
        self.input_matrix = self.compute_input_matrix(sampling_time=self.sampling_time)


class CWH4DEnv(BaseCWH, DynamicalSystem):
    """4D Clohessy-Wiltshire-Hill (CWH) system.

    Bases: :py:class:`gym_socks.envs.cwh.BaseCWH`, :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    The 4D CWH system is a simplification of the 6D dynamics to operate within a plane.
    Essentially, it ignores the :math:`z` component of the dynamics.

    """

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.01, high=0.01, shape=(2,), dtype=np.float32
        )

        self.state = None

        self.seed(seed=seed)

        self.compute_mu()
        self.compute_angular_velocity()

        self.state_matrix = self.compute_state_matrix(sampling_time=self.sampling_time)
        self.input_matrix = self.compute_input_matrix(sampling_time=self.sampling_time)

    def compute_state_matrix(self, sampling_time):
        n = self.angular_velocity
        nt = n * sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        return [
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

    def compute_input_matrix(self, sampling_time):
        n = self.angular_velocity
        Ts = sampling_time
        nt = n * Ts
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        sin_nt2 = np.sin((nt) / 2)

        M = self.chief_mass

        eAt = [
            [
                4 * Ts - (3 * sin_nt) / n,
                0,
                (2 * (sin_nt2 ** 2)) / (n ** 2),
                -(2 * (sin_nt - nt)) / (n ** 2),
            ],
            [
                (12 * (sin_nt2 ** 2)) / n - 3 * (Ts ** 2) * n,
                Ts,
                (2 * (sin_nt - nt)) / (n ** 2),
                (8 * (sin_nt2 ** 2)) / (n ** 2) - (3 * (Ts ** 2)) / 2,
            ],
            [3 - 3 * cos_nt, 0, sin_nt / n, (4 * (sin_nt2 ** 2)) / n],
            [
                6 * sin_nt - 6 * nt,
                0,
                -(4 * (sin_nt2 ** 2)) / n,
                (4 * sin_nt) / n - 3 * Ts,
            ],
        ]

        B = [
            [0, 0],
            [0, 0],
            [(1 / M), 0],
            [0, (1 / M)],
        ]

        return np.matmul(eAt, B)

    def step(self, action, time=0):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        disturbance = self.generate_disturbance(time, self.state, action)
        self.state = self.dynamics(time, self.state, action, disturbance)
        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = {}

        return observation, cost, done, info

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        w = np.multiply([1e-4, 1e-4, 5e-8, 5e-8], w)
        return np.array(w)

    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system.

        Attention:
            For this system, the :py:obj:`scipy.solve_ivp` function does not return the
            correct result for the dynamical equations, and will quickly run into
            numerical issues where the states explode. However, the CWH system has a
            closed-form solution for the equations of motion, so we use this instead to
            calculate the evolution of the system. See the :py:meth:`step` function for
            additional details regarding how the next state is calculated.

        """

        return (
            np.matmul(self.state_matrix, state)
            + np.matmul(self.input_matrix, action)
            + disturbance
        )

        # x1, x2, x3, x4 = state
        # u1, u2 = action
        # w1, w2, w3, w4 = disturbance

        # dx1 = x1 + w1
        # dx2 = x2 + w2
        # dx3 = (
        #     3 * (self.angular_velocity ** 2) * x1
        #     + 2 * self.angular_velocity * x4
        #     + (u1 / self.chief_mass)
        #     + w3
        # )
        # dx4 = -2 * self.angular_velocity * x3 + (u2 / self.chief_mass) + w4

        # return np.array([dx1, dx2, dx3, dx4], dtype=np.float32)


class CWH6DEnv(BaseCWH, DynamicalSystem):
    """6D Clohessy-Wiltshire-Hill (CWH) system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    """

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.01, high=0.01, shape=(3,), dtype=np.float32
        )

        self.state = None

        self.seed(seed=seed)

        self.compute_mu()
        self.compute_angular_velocity()

        self.state_matrix = self.compute_state_matrix(sampling_time=self.sampling_time)
        self.input_matrix = self.compute_input_matrix(sampling_time=self.sampling_time)

    def compute_state_matrix(self, sampling_time):
        n = self.angular_velocity
        nt = n * sampling_time
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        return [
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

    def compute_input_matrix(self, sampling_time):
        n = self.angular_velocity
        Ts = sampling_time
        nt = n * Ts
        sin_nt = np.sin(nt)
        cos_nt = np.cos(nt)

        sin_nt2 = np.sin((nt) / 2)

        M = self.chief_mass

        eAt = [
            [
                4 * Ts - (3 * sin_nt) / n,
                0,
                0,
                (2 * (sin_nt2 ** 2)) / (n ** 2),
                -(2 * (sin_nt - nt)) / (n ** 2),
                0,
            ],
            [
                (12 * (sin_nt2 ** 2)) / n - 3 * (Ts ** 2) * n,
                Ts,
                0,
                (2 * (sin_nt - nt)) / (n ** 2),
                (8 * (sin_nt2 ** 2)) / (n ** 2) - (3 * (Ts ** 2)) / 2,
                0,
            ],
            [0, 0, sin_nt / n, 0, 0, (2 * (sin_nt2 ** 2)) / (n ** 2)],
            [
                3 - 3 * cos_nt,
                0,
                0,
                sin_nt / n,
                (4 * (sin_nt2 ** 2)) / n,
                0,
            ],
            [
                6 * sin_nt - 6 * nt,
                0,
                0,
                -(4 * (sin_nt2 ** 2)) / n,
                (4 * sin_nt) / n - 3 * Ts,
                0,
            ],
            [0, 0, cos_nt - 1, 0, 0, sin_nt / n],
        ]

        B = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [(1 / M), 0, 0],
            [0, (1 / M), 0],
            [0, 0, (1 / M)],
        ]

        return np.matmul(eAt, B)

    def step(self, action, time=0):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        disturbance = self.generate_disturbance(time, self.state, action)
        self.state = self.dynamics(time, self.state, action, disturbance)
        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = {}

        return observation, cost, done, info

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        w = np.multiply([1e-4, 1e-4, 1e-4, 5e-8, 5e-8, 5e-8], w)
        return np.array(w)

    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system.

        Attention:
            For this system, the :py:obj:`scipy.solve_ivp` function does not return the
            correct result for the dynamical equations, and will quickly run into
            numerical issues where the states explode. However, the CWH system has a
            closed-form solution for the equations of motion, so we use this instead to
            calculate the evolution of the system. See the :py:meth:`step` function for
            additional details regarding how the next state is calculated.

        """

        return (
            np.matmul(self.state_matrix, self.state)
            + np.matmul(self.input_matrix, action)
            + disturbance
        )

        # x1, x2, x3, x4, x5, x6 = state
        # u1, u2, u3 = action
        # w1, w2, w3, w4, w5, w6 = disturbance

        # dx1 = x1 + w1
        # dx2 = x2 + w2
        # dx3 = x3 + w3
        # dx4 = (
        #     3 * (self.angular_velocity ** 2) * x1
        #     + 2 * self.angular_velocity * x5
        #     + (u1 / self.chief_mass)
        #     + w4
        # )
        # dx5 = -2 * self.angular_velocity * x4 + (u2 / self.chief_mass) + w5
        # dx6 = -(self.angular_velocity ** 2) * x3 + (u3 / self.chief_mass) + w6

        # return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=np.float32)
