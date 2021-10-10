from abc import ABC, abstractmethod
import warnings

import gym
from gym.utils import seeding

import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystem(gym.Env, ABC):
    """
    Base class for dynamical system models.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly. Instead, define a new class that inherits from DynamicalSystem, and define a custom 'dynamics' function.

    Example:

    class CustomDynamicalSystem(DynamicalSystem):
        def __init__(self):
            super().__init__(state_dim=2, action_dim=1)

        def dynamics(self, t, x, u, w):
            return u

        def reset(self):
            self.state = self.np_random.uniform(
                low=-1, high=1, size=self.observation_space.shape
            )
            return np.array(self.state)


    * The state space and input space are assumed to be R^n and R^m, where n and
      m are set by state_dim and action_dim above (though these can be altered,
      see gym.spaces for more info).
    * The dynamics function (defined by you) returns dx/dt, and the system is
      integrated using scipy.integrate.solve_ivp to determine the state at the
      next time instant and discretize the system.
    * The reset function sets a new initial condition for the system.

    The system can then be simulated using the standard gym environment.

    Example:

    import gym
    import numpy as np

    env = CustomDynamicalSystem()
    obs = env.reset()

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
        action_space=None,
        seed=None,
        euler=False,
        *args,
        **kwargs,
    ):
        """
        Initialize the dynamical system.

        Example usage:
        env = systems.envs.integrator.DoubleIntegratorEnv
        env.time_horizon = 5
        env.sampling_time = 0.25
        """

        if observation_space is not None:
            self.observation_space = observation_space
        else:
            raise ValueError("Must supply an observation_space.")

        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("Must supply an action_space.")

        # time parameters
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
        if value > self._time_horizon:
            warnings.warn(
                f"Sampling time {value} is less than time horizon {self._time_horizon}."
            )
        self._sampling_time = value

    @property
    def time_horizon(self):
        return self._time_horizon

    @time_horizon.setter
    def time_horizon(self, value):
        if value < self._sampling_time:
            warnings.warn(
                f"Sampling time {value} is less than time horizon {self._time_horizon}."
            )
        self._time_horizon = value

    @property
    def num_time_steps(self):
        return int(self._time_horizon // self._sampling_time)

    @property
    def state_dim(self):
        return self.observation_space.shape

    @property
    def action_dim(self):
        return self.action_space.shape

    def step(self, action):
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

        if self._euler is True:
            next_state = self.state + self.sampling_time * self.dynamics(
                0, self.state, action
            )
            self.state = next_state
        else:
            # solve the initial value problem
            sol = solve_ivp(
                self.dynamics, [0, self.sampling_time], self.state, args=(action,)
            )
            *_, self.state = sol.y.T

        cost = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), cost, done, info

    def reset(self):
        """Resets the system to a random initial condition."""

        self.state = self.np_random.uniform(
            low=0, high=1, size=self.observation_space.shape
        )

        return np.array(self.state)

    def render(self, mode="human"):
        """Render function for displaying the system graphically."""
        raise NotImplementedError

    def close(self):
        """Deconstructor method."""
        raise NotImplementedError

    @abstractmethod
    def dynamics(
        self, t: "Time variable.", x: "State vector.", u: "Input vector."
    ) -> "dx/dt":
        """
        Dynamics for the system.

        Required to be overloaded by subclasses.
        """
        ...

    def cost(self, u: "Input vector.") -> "R":
        """
        Cost function for the system.
        """
        return 0.0


class StochasticMixin(DynamicalSystem):
    """
    Base class for stochastic dynamical system models.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly. Instead, define a new class that inherits from DynamicalSystem, and define a custom 'dynamics' function.

    Example:

    class CustomDynamicalSystem(StochasticMixin, DynamicalSystem):
        def __init__(self):
            super().__init__(state_dim=2, action_dim=1, disturbance_dim=2)

        def dynamics(self, t, x, u, w):
            return u + w

        def reset(self):
            self.state = self.np_random.uniform(
                low=-1, high=1, size=self.observation_space.shape
            )
            return np.array(self.state)

        def sample_disturbance(self):
            w = self.np_random.standard_normal(size=self.disturbance_space.shape)
            return 1e-2 * np.array(w)


    * The state space and input space are assumed to be R^n and R^m, where n and
      m are set by state_dim and action_dim above. The disturbance dimension is assumed
      to be R^p, and is generally p = n.
    * The sample_disturbance function is a custom function which you define, and
      returns a disturbance vector, which is added to the state at each each time step.
      Thus, the output of this function should be of the same dimension as the state.

    """

    def __init__(self, disturbance_space=None, *args, **kwargs):
        """
        Initialize the dynamical system.

        Example usage:
        env = systems.envs.integrator.DoubleIntegratorEnv
        env.time_horizon = 5
        env.sampling_time = 0.25
        """

        super().__init__(*args, **kwargs)

        if disturbance_space is not None:
            self.disturbance_space = disturbance_space
            self.disturbance_dim = self.disturbance_space.shape
        else:
            raise ValueError("Must supply a disturbance_space.")

    def step(self, action):
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

        # generate a disturbance
        disturbance = self.sample_disturbance()

        if self._euler is True:
            next_state = self.state + self.sampling_time * self.dynamics(
                0, self.state, action, disturbance
            )
            self.state = next_state
        else:
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

        cost = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), cost, done, info

    @abstractmethod
    def dynamics(
        self,
        t: "Time variable.",
        x: "State vector.",
        u: "Input vector.",
        w: "Disturbance vector." = None,
    ) -> "dx/dt":
        """
        Dynamics for the system.

        Required to be overloaded by subclasses.
        """
        ...

    def sample_disturbance(self):
        """
        Sample the disturbance.

        By default, returns a Gaussian sample from the disturbance space.
        """
        w = self.np_random.standard_normal(size=self.disturbance_space.shape)
        return np.array(w)
