
from gym_basic.envs.dynamical_system_env import DynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp

class NDIntegratorEnv(DynamicalSystemEnv):
    """
    ND integrator system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, dim):
        """Initialize the system."""
        super().__init__(state_dim=dim, action_dim=1)



    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        def dynamics(t, y):
            """Dynamics for the system."""
            _, *x = y
            return (*x, action)

        # solve the initial value problem
        sol = solve_ivp(dynamics, [0, self.sampling_time], self.state)
        *y, self.state = sol.y.T

        reward = self.cost(action)

        done = False
        info = {}

        return np.array(self.state), reward, done, info



    def reset(self):
        self.state = self.np_random.uniform(low=-1, high=1, size=self.observation_space.shape)
        return np.array(self.state)



class DoubleIntegratorEnv(NDIntegratorEnv):
    """
    2D integrator system.
    """

    def __init__(self):
        """Initialize the system."""
        super().__init__(dim=2)



    def reset(self):
        self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
        return np.array(self.state)
