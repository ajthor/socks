from abc import ABC, abstractmethod

import gym
from gym.utils import seeding

import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystemEnv(gym.Env, ABC):
    """
    Base class for dynamical system models.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly. Instead, define a new class that inherits from DynamicalSystemEnv, and define a custom 'dynamics' function.

    Example:

    class CustomDynamicalSystemEnv(DynamicalSystemEnv):
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

    env = CustomDynamicalSystemEnv()
    obs = env.reset()

    num_steps = int(np.floor(env.time_horizon/env.sampling_time))

    for i in range(num_steps):

        # get random action
        action = env.action_space.sample()

        # advance the system one time step
        obs, reward, done, _ = env.step(action)

    env.close()
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, state_dim, action_dim):
        """
        Initialize the dynamical system.

        Example usage:
        env = gym_basic.envs.integrator.DoubleIntegratorEnv
        env.time_horizon = 5
        env.sampling_time = 0.25
        """

        # time parameters
        self.time_horizon = 1
        self.sampling_time = 0.1

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )

        self.seed()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Step function defined by OpenAI Gym.

        Advances the system forward one time step.

        Returns
        -------

        obs : ndarray
            The observation vector. If the system is fully observable, this is the state of the system at the next time step.

        reward : float32
            The reward (cost) obtained by the system for taking action u in state x and transitioning to state y.

        done : bool
            Flag to indicate the simulation has terminated. Usually toggled by guard conditions, which terminates the simulation if the system violates certain operating constraints.

        info : {}
            Extra information.
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # solve the initial value problem
        sol = solve_ivp(
            self.dynamics, [0, self.sampling_time], self.state, args=(action,)
        )
        *_, self.state = sol.y.T

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

    def reset(self):
        """
        Resets the system to a random initial condition.
        """

        self.state = self.np_random.uniform(
            low=0, high=1, size=self.observation_space.shape
        )

        return np.array(self.state)

    def render(self, mode="human"):
        """
        Render function for displaying the system graphically.
        """
        ...

    def close(self):
        """
        Deconstructor method.
        """
        ...

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


class StochasticDynamicalSystemEnv(DynamicalSystemEnv):
    """
    Base class for stochastic dynamical system models.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly. Instead, define a new class that inherits from DynamicalSystemEnv, and define a custom 'dynamics' function.

    Example:

    class CustomDynamicalSystemEnv(DynamicalSystemEnv):
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

    env = CustomDynamicalSystemEnv()
    obs = env.reset()

    num_steps = int(np.floor(env.time_horizon/env.sampling_time))

    for i in range(num_steps):

        # get random action
        action = env.action_space.sample()

        # advance the system one time step
        obs, reward, done, _ = env.step(action)

    env.close()
    """

    def __init__(self, state_dim, action_dim, disturbance_dim):
        """
        Initialize the dynamical system.

        Example usage:
        env = gym_basic.envs.integrator.DoubleIntegratorEnv
        env.time_horizon = 5
        env.sampling_time = 0.25
        """

        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.disturbance_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(disturbance_dim,), dtype=np.float32
        )

        self.disturbance_dim = disturbance_dim

    def step(self, action):
        """
        Step function defined by OpenAI Gym.

        Advances the system forward one time step.

        Returns
        -------

        obs : ndarray
            The observation vector. If the system is fully observable, this is the state of the system at the next time step.

        reward : float32
            The reward (cost) obtained by the system for taking action u in state x and transitioning to state y.

        done : bool
            Flag to indicate the simulation has terminated. Usually toggled by guard conditions, which terminates the simulation if the system violates certain operating constraints.

        info : {}
            Extra information.
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # generate a disturbance
        disturbance = self.sample_disturbance()

        # print("state")
        # print(type(self.state))
        # print("action")
        # print(type(action))
        # print("disturbance")
        # print(type(disturbance))

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

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info

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
        w = 0*self.np_random.standard_normal(size=self.disturbance_space.shape)
        return np.array(w)
