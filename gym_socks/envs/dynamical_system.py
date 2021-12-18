from abc import ABC, abstractmethod

import gym
from gym.utils import seeding

import numpy as np
from scipy.integrate import solve_ivp

from gym_socks.envs.core import BaseDynamicalObject


class DynamicalSystem(BaseDynamicalObject, ABC):
    r"""Base class for dynamical system models.

    Bases: :py:class:`gym_socks.envs.core.BaseDynamicalObject`, :py:obj:`abc.ABC`

    This class is **abstract**, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from :py:class:`DynamicalSystem`, and
    define a custom :py:meth:`dynamics` function.

    Example:
        ::

            import gym
            import numpy as np
            from gym_socks.envs.dynamical_system import DynamicalSystem
            class CustomDynamicalSystem(DynamicalSystem):

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                    state_dim = 2  # 2-D state and observation space
                    action_dim = 1  # 1-D action space

                    self.observation_space = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                    )
                    self.state_space = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                    )
                    self.action_space = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
                    )

                    self.state = None

                    self.seed(seed=seed)

                def dynamics(self, time, state, action, disturbance):
                    ...

    Important:
        The state space and input space are assumed to be :math:`\mathbb{R}^{n}` and
        :math:`\mathbb{R}^{m}`, respectively, where :math:`n` and :math:`m` are
        specified by ``state_dim`` and ``action_dim`` above.

        In addition, the default :py:meth:`step` function solves an initial value
        problem via :py:obj:`scipy.integrate.solve_ivp` to compute the state of the
        system at the next time step. Thus, in order to use discrete spaces, you will
        need to override the :py:meth:`step` function.

        See :py:obj:`gym.spaces` for more information on the different available spaces.

    The system can then be simulated using the standard gym environment.

    Example:

        >>> env = CustomDynamicalSystem()
        >>> env.reset()
        >>> for t in range(10):
        ...     action = env.action_space.sample()
        ...     obs, *_ = env.step(action)
        ...
        >>> env.close()

    """

    observation_space = None
    """The space of system observations.

    Caution:
        The observation space typically only differs from the state space if the system
        is partially observable. If this is the case, :py:meth:`generate_observation`
        should be defined to return an element of the observation space.

    """
    action_space = None
    """The action (input) space of the system."""
    state_space = None
    """The state space of the system.

    Note:
        By convention, in controls theory the state space of the system is the set of
        all possible states that the system can have. OpenAI Gym's convention is to
        ignore the underlying state space, opting to use only the
        :py:attr:`observation_space`.

    """

    _euler = False

    _sampling_time = 0.1

    @property
    def sampling_time(self):
        """Sampling time, in seconds."""
        return self._sampling_time

    @sampling_time.setter
    def sampling_time(self, value):
        self._sampling_time = value

    def generate_disturbance(self, time, state, action):
        """Generate a disturbance.

        Note:
            Override :py:meth:`generate_disturbance` in subclasses to modify the
            disturbance properties, such as the scale or distribution.

        """

        return self._np_random.standard_normal(size=self.state_space.shape)

    @abstractmethod
    def dynamics(self, time, state, action, disturbance):
        """Dynamics for the system.

        .. math::

            \dot{x} = f(x, u, w)

        Args:
            time: The time variable.
            state: The state of the system at the current time step.
            action: The control action applied at the current time step.
            disturbance: A realization of a random variable representing process noise.

        Returns:
            The state of the system at the next time step.

        Important:
            The dynamics function (defined by you) returns :math:`\dot{x}`, and the
            system is integrated using :py:obj:`scipy.integrate.solve_ivp` in the
            :py:meth:`step` function to determine the state at the next time instant
            and discretize the system in time.

            To specify discrete time dynamics explicitly (for instance with linear
            dynamics such as :math:`x_{t+1} = A x_{t} + B x_{t} + w_{t}` where :math:`A`
            and :math:`B` are known, but the ODEs are difficult to write or are
            time-varying), override the :py:meth:`step` function.

            For example::

                import gym
                import numpy as np
                from gym_socks.envs.dynamical_system import DynamicalSystem
                class CustomDynamicalSystem(DynamicalSystem):

                    def __init__(self, state_dim, action_dim, *args, **kwargs):
                        super().__init__(*args, **kwargs)

                        self.observation_space = gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(state_dim,),
                            dtype=np.float32
                        )
                        self.state_space = gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(state_dim,),
                            dtype=np.float32
                        )
                        self.action_space = gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(action_dim,),
                            dtype=np.float32
                        )

                        self.state = None

                        self.seed(seed=seed)

                        def step(self, time, action):

                            disturbance = self.generate_disturbance(
                                time, self.state, action
                            )
                            self.state = self.dynamics(
                                time, self.state, action, disturbance
                            )
                            obs = self.generate_observation(time, self.state, action)

                            return obs, 0, False, {}

                        def dynamics(self, time, state, action, disturbance):

                            A = compute_state_matrix(time)
                            B = compute_input_matrix(time)

                            return A @ state + B @ action + disturbance

        """
        raise NotImplementedError

    def generate_observation(self, time, state, action):
        """Generate an observation from the system.

        .. math::

            y = h(x, u, v)

        Args:
            time: The time variable.
            state: The state of the system at the current time step.
            action: The control action applied at the current time step.

        Returns:
            An observation of the system at the current time step.

        Note:
            Override :py:meth:`generate_observation` in subclasses if the system is
            partially observable. By default, the function returns the system state
            directly, meaning it is fully observable.

        """

        return np.array(state, dtype=np.float32)

    def cost(self, time, state, action):
        """Cost function for the system.

        Warning:
            This function is typically not used in SOCKS, but is included here for
            compatibility with OpenAI gym, which returns the cost from the
            :py:meth:`step` function.

        """

        return 0.0

    def step(self, time=0, action=None) -> tuple:
        """Advances the system forward one time step.

        Args:
            time: Time of the simulation. Used primarily for time-varying systems.
            action: Action (input) applied to the system at the current time step.

        Returns:
            A tuple ``(obs, cost, done, info)``, where ``obs`` is the observation
            vector. Generally, it is the state of the system corrupted by some
            measurement noise. If the system is fully observable, this is the actual
            state of the system at the next time step. ``cost`` is the cost
            (reward) obtained by the system for taking action u in state x and
            transitioning to state y. In general, this is not typically used with
            :py:class:`DynamicalSystem` models. ``done`` is a flag to indicate the
            simulation has terminated. Usually toggled by guard conditions, which
            terminates the simulation if the system violates certain operating
            constraints. ``info`` is a dictionary containing extra information.

        """

        action = np.asarray(action)

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
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
