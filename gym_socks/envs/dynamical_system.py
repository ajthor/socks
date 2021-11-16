from abc import ABC, abstractmethod

import gym
from gym.utils import seeding

import numpy as np
from scipy.integrate import solve_ivp

from gym_socks.envs.core import BaseDynamicalObject


class DynamicalSystem(BaseDynamicalObject, ABC):
    """Base class for dynamical system models.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from `DynamicalSystem`, and define a
    custom `dynamics` function.

    Example:

        >>> from gym_socks.envs.dynamical_system import DynamicalSystem
        >>> class CustomDynamicalSystem(DynamicalSystem):
        ...     def __init__(self):
        ...         super().__init__(state_dim=2, action_dim=1)
        ...
        ...     def dynamics(self, t, x, u, w):
        ...         return u

    Note:
        * The state space and input space are assumed to be R^n and R^m, where n and m
        are set by state_dim and action_dim above (though these can be altered, see
        gym.spaces for more info).
        * The dynamics function (defined by you) returns dx/dt, and the system is
        integrated using scipy.integrate.solve_ivp to determine the state at the next
        time instant and discretize the system.
        * The reset function sets a new initial condition for the system.

    The system can then be simulated using the standard gym environment.

    Example:

        >>> import gym
        >>> import numpy as np
        >>> env = CustomDynamicalSystem()
        >>> env.reset()
        >>> num_steps = env.num_time_steps
        >>> for i in range(num_steps):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, _ = env.step(action)
        >>> env.close()

    Args:
        observation_space: The space of system observations.
        state_space: The state space of the system.
        action_space: The action (input) space of the system.
        seed: Random seed.
        euler: Whether to use the Euler approximation for simulation.

    """

    observation_space = None
    action_space = None
    state_space = None

    _euler = False

    _sampling_time = 0.1

    @property
    def sampling_time(self):
        """Sampling time."""
        return self._sampling_time

    @sampling_time.setter
    def sampling_time(self, value):
        self._sampling_time = value

    def generate_disturbance(self, time, state, action):
        """Generate disturbance."""
        return self._np_random.standard_normal(size=self.state_space.shape)

    @abstractmethod
    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system.

        The dynamics are typically specified by a function::
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
        return np.array(state, dtype=np.float32)

    def cost(self, time, state, action):
        """Cost function for the system."""
        return 0.0

    def step(self, time=0, action=None):
        """Step function defined by OpenAI Gym.

        Advances the system forward one time step.

        Args:
            time: Time of the simulation. Used primarily for time-varying systems.
            action: Action (input) applied to the system at the current time step.

        Returns:
            obs: The observation vector. Generally, it is the state of the system
                corrupted by some measurement noise. If the system is fully observable,
                this is the state of the system at the next time step.
            cost: The cost (reward) obtained by the system for taking action u in state
                x and transitioning to state y. In general, this is not typically used with `DynamicalSystem` models.
            done: Flag to indicate the simulation has terminated. Usually toggled by
                guard conditions, which terminates the simulation if the system
                violates certain operating constraints.
            info: Extra information.

        """

        action = np.asarray(action, dtype=np.float32)

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
            # Solve the initial value problem.
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
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def np_random(self):
        """Random number generator."""
        if self._np_random is None:
            self.seed()

        return self._np_random

    def seed(self, seed=None):
        """Sets the seed of the random number generator.

        Args:
            seed: Integer value representing the random seed.

        Returns:
            The seed of the RNG.

        """
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
