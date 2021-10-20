from abc import ABC, abstractmethod
import warnings

import gym
from gym.utils import seeding

import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystem(gym.Env, ABC):
    """
    Base class for dynamical system models.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from DynamicalSystem, and define a custom
    'dynamics' function.

    Example:

    class CustomDynamicalSystem(DynamicalSystem): def __init__(self):
        super().__init__(state_dim=2, action_dim=1)

        def dynamics(self, t, x, u, w):
            return u

        def reset(self):
            self.state = self.np_random.uniform(
                low=-1, high=1, size=self.observation_space.shape
            )
            return np.array(self.state)


    * The state space and input space are assumed to be R^n and R^m, where n and m are
      set by state_dim and action_dim above (though these can be altered, see gym.spaces
      for more info).
    * The dynamics function (defined by you) returns dx/dt, and the system is integrated
      using scipy.integrate.solve_ivp to determine the state at the next time instant
      and discretize the system.
    * The reset function sets a new initial condition for the system.

    The system can then be simulated using the standard gym environment.

    Example:

    import gym import numpy as np

    env = CustomDynamicalSystem()
    env.reset()

    num_steps = env.num_time_steps

    for i in range(num_steps):

        # get random action
        action = env.action_space.sample()

        # advance the system one time step
        obs, reward, done, _ = env.step(action)

    env.close()
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        observation_space=None,
        state_space=None,
        action_space=None,
        seed=None,
        euler=False,
        *args,
        **kwargs,
    ):
        """Initialize the dynamical system."""

        if observation_space is None:
            raise ValueError("Must supply an observation_space.")

        self.observation_space = observation_space

        if state_space is None:
            state_space = observation_space

        self.state_space = state_space

        if action_space is None:
            raise ValueError("Must supply an action_space.")

        self.action_space = action_space

        # Time parameters.
        self._time_horizon = 1
        self._sampling_time = 0.1

        self.state = None

        self._seed = None
        self._np_random = None
        if seed is not None:
            self._seed = self.seed(seed)

        self._euler = euler

    @property
    def np_random(self):
        if self._np_random is None:
            self._seed = self.seed()

        return self._np_random

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def sampling_time(self):
        return self._sampling_time

    @sampling_time.setter
    def sampling_time(self, value):
        msg = f"Sampling time {value} is less than time horizon {self._time_horizon}."
        if value > self._time_horizon:
            warnings.warn(msg)
        self._sampling_time = value

    @property
    def time_horizon(self):
        return self._time_horizon

    @time_horizon.setter
    def time_horizon(self, value):
        msg = f"Sampling time {value} is less than time horizon {self._time_horizon}."
        if value < self._sampling_time:
            warnings.warn(msg)
        self._time_horizon = value

    @property
    def num_time_steps(self):
        return int(self._time_horizon // self._sampling_time)

    @property
    def observation_dim(self):
        """Dimensionality of the observation space."""
        return self.observation_space.shape

    @property
    def state_dim(self):
        """Dimensionality of the state space."""
        return self.state_space.shape

    @property
    def action_dim(self):
        """Dimensionality of the action space."""
        return self.action_space.shape

    def step(self, action, time=0):
        """
        Step function defined by OpenAI Gym.

        Advances the system forward one time step.

        Returns
        -------

        obs : ndarray
            The observation vector. If the system is fully observable, this is the state
            of the system at the next time step.

        cost : float32
            The cost (reward) obtained by the system for taking action u in state x and
            transitioning to state y.

        done : bool
            Flag to indicate the simulation has terminated. Usually toggled by guard
            conditions, which terminates the simulation if the system violates certain
            operating constraints.

        info : {}
            Extra information.
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # Generate disturbance.
        disturbance = self.generate_disturbance(time, self.state, action)

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

        # Generate observation.
        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = {}

        return observation, cost, done, info

    def reset(self):
        """Reset the system to a random initial condition."""
        self.state = self.state_space.sample()

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def generate_disturbance(self, time, state, action):
        """Generate disturbance."""
        return self._np_random.standard_normal(size=self.state_space.shape)

    @abstractmethod
    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system.

        y = f(t, x, u, w)
              ┬  ┬  ┬  ┬
              │  │  │  └┤ w : Disturbance
              │  │  └───┤ u : Control action
              │  └──────┤ x : System state
              └─────────┤ t : Time variable
        """
        raise NotImplementedError

    def generate_observation(self, time, state, action):
        """Generate observation."""
        return state

    def cost(self, time, state, action):
        """Cost function for the system."""
        return 0.0
