from abc import ABC, abstractmethod

import math
import gym

from gym_basic.envs.dynamical_system import DynamicalSystemEnv

import numpy as np
from scipy.integrate import solve_ivp


def generate_initial_conditions(
    sample_space: gym.spaces, system: DynamicalSystemEnv, n: "Number of samples." = 10
):
    """Generate a collection of initial conditions."""
    return np.array([sample_space.sample() for i in range(n)])


def generate_sample(
    sample_space: gym.spaces, system: DynamicalSystemEnv, n: "Number of samples." = 10
):
    """
    Generate a sample from a dynamical system.

    The sample consists of n state pairs (x_i, y_i) where x_i is a state vector sampled randomly from the 'sample_space', and y_i is the state vector at the next time step. In a deterministic system, this has the form:

        y = f(x, u)

    where u is chosen randomly from the system's action space.
    """

    # assert sample space and observation space are the same dimensionality
    err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
    assert system.observation_space.shape == sample_space.shape, err_msg

    # generate initial conditions
    x = generate_initial_conditions(sample_space, system, n)

    def generate_next_state(x0):
        sol = solve_ivp(
            system.dynamics,
            [0, system.sampling_time],
            x0,
            args=(system.action_space.sample(),),
        )
        *_, next_state = sol.y.T
        return next_state

    S = np.array([(x0, generate_next_state(x0)) for x0 in x])

    return S


def generate_sample_trajectories(
    sample_space: gym.spaces, system: DynamicalSystemEnv, n: "Number of samples." = 10
):
    """
    Generate a sample of trajectories from a dynamical system.

    The sample consists of n state trajectories (x_0, x_1, ..., x_N)_i, where x_0 are state vectors sample randomly from the 'sample_space', and x_1, ..., x_N are state vectors at subsequent time steps, determined by the system.sampling_time, and N is the system.time_horizon. The control actions u_0, ..., u_N-1 applied at each time step are chosen randomly from the system's action space.

    Parameters
    ----------


    Returns
    -------

    ndarray
        The array has the shape (n, t, d), where n is the number of samples, t is the number of time steps in [0, N], and d is the dimensionality of the sample space.
    """

    # assert sample space and observation space are the same dimensionality
    err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
    assert system.observation_space.shape == sample_space.shape, err_msg

    # generate initial conditions
    x = generate_initial_conditions(sample_space, system, n)

    N = system.time_horizon
    Ts = system.sampling_time

    # create list of time indices to integrate to
    def t_range(start, stop, step):
        h = 1 / step
        i = 0
        r = start
        while r < stop:
            i += 1
            r = start + i / h
            yield r

    times = list(t_range(0, N, Ts))

    # NOTE: Python has problems with floating-point precision with simple
    # caluclations such as 3 * 0.1 = 0.30000000000000004, which is harder to
    # integrate to and may cause issues with the number of elements in the time
    # vector we want to integrate over. Dividing by the reciprocal seems to
    # solve this, but is simply rounding again.
    #
    # Some other possible methods to obtain time steps:

    # times = np.linspace(0, N, int(np.floor(N/Ts)), endpoint=False)

    # Hz = 1/Ts
    # num = int(np.floor(N/Ts))
    # times = [i/Hz for i in range(num)]

    def generate_state_trajectory(x0):
        sol = solve_ivp(
            system.dynamics,
            [0, N],
            x0,
            args=(system.action_space.sample(),),
            t_eval=times,
        )
        return sol.y.T

    S = [(x0, *generate_state_trajectory(x0)) for x0 in x]

    return np.array(S)
