"""Non-Markovian Integrator system."""

import gym
from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class NonMarkovIntegratorEnv(DynamicalSystem):
    """Non-Markovian Integrator System

    Example:

        >>> from gym_socks.envs import NDIntegratorEnv
        >>> env = NDIntegratorEnv(dim=2)
        >>> env.reset()
        >>> num_steps = env.num_time_steps
        >>> for i in range(num_steps):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, _ = env.step(action)
        >>> env.close()

    """

    _sampling_time = 1.0

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.state = None

        # self.mass = 1
        # self.alpha = 1

        self.seed(seed=seed)

        self.state_matrix = self.compute_state_matrix(sampling_time=self.sampling_time)
        self.input_matrix = self.compute_input_matrix(sampling_time=self.sampling_time)

    def compute_state_matrix(self, sampling_time):
        return np.array(
            [
                [1, sampling_time, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, sampling_time],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def compute_input_matrix(self, sampling_time):
        return np.array(
            [
                [(sampling_time ** 2) / 2, 0],
                [sampling_time, 0],
                [0, (sampling_time ** 2) / 2],
                [0, sampling_time],
            ],
            dtype=np.float32,
        )

    def step(self, action, time=0):
        # action = np.asarray(action)

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        disturbance = self.generate_disturbance(time, self.state, action)

        drag_vector = -self.alpha * np.array(
            [
                (self.sampling_time ** 2) * np.abs(self.state[1]) * self.state[1] / 2,
                self.sampling_time * np.abs(self.state[1]) * self.state[1],
                (self.sampling_time ** 2) * np.abs(self.state[3]) * self.state[3] / 2,
                self.sampling_time * np.abs(self.state[3]) * self.state[3],
            ],
            dtype=np.float32,
        )

        # use closed-form solution
        self.state = (
            np.matmul(self.state_matrix, self.state)
            + np.matmul((1 / self.mass) * self.input_matrix, action)
            + drag_vector
            + disturbance
        )

        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = {}

        return observation, cost, done, info

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-2 * np.array(w)

    def dynamics(self, time, state, action, disturbance):
        raise NotImplementedError

    def reset(self):
        # Mass is at least 0.1, with an additive squared exponential term.
        # self.mass = 0.1 + self.np_random.exponential(scale=0.1, size=(1,))
        # self.mass = 0.75 + (self.np_random.beta(a=2, b=2) / 2)
        # self.mass = 0.5 + self.np_random.beta(a=2, b=2)
        self.mass = 0.9 + (self.np_random.beta(a=2, b=2) / 5)
        # self.mass = self.np_random.beta(a=2, b=2)
        self.alpha = self.np_random.beta(a=2, b=5) * 0.02
        # self.mass = 0.5
        # self.alpha = 0

        # self.state = self.state_space.sample()
        # return np.array(self.state, dtype=np.float32)
