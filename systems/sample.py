from abc import ABC, abstractmethod

import gym

from systems.envs.dynamical_system import DynamicalSystem

import numpy as np
from scipy.integrate import solve_ivp


def generate_initial_conditions(
    sample_space: gym.spaces, system: DynamicalSystem, n: "Number of samples." = 10
):
    """Generate a collection of initial conditions."""
    return np.array([sample_space.sample() for i in range(n)])


def generate_sample(
    sample_space: gym.spaces, system: DynamicalSystem, n: "Number of samples." = 10
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
        system.state = x0

        action = system.action_space.sample()
        next_state, reward, done, _ = system.step(action)

        return next_state

    S = np.array([(x0, generate_next_state(x0)) for x0 in x])

    return S


def generate_sample_trajectories(
    sample_space: gym.spaces, system: DynamicalSystem, n: "Number of samples." = 10
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

    def generate_state_trajectory(x0):
        system.state = x0

        def generate_next_state():
            action = system.action_space.sample()
            next_state, reward, done, _ = system.step(action)
            return next_state

        return np.array([generate_next_state() for i in range(int(np.floor(N / Ts)))])

    S = [(x0, *generate_state_trajectory(x0)) for x0 in x]

    return np.array(S)
