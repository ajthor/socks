"""
Sampling Methods
"""
from abc import ABC, abstractmethod

import itertools

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import RandomizedPolicy
from gym_socks.envs.policy import ConstantPolicy

import numpy as np
from scipy.integrate import solve_ivp


def random_initial_conditions(
    system: DynamicalSystem, sample_space: gym.spaces, n: "Sample size." = None
):
    """
    Generate a collection of random initial conditions.

    As per the gym documentation, a different distribution is used depending on the
    boundedness of the sampling space.

    * [a, b] : uniform distribution
    * [a, oo) : shifted exponential distribution
    * (-oo, b] : negative shifted exponential distribution
    * (-oo, oo) : normal distribution

    See the numpy documentation for more information about the sampling distribution.
    """

    # assert sample space and observation space are the same dimensionality
    err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
    assert system.observation_space.shape == sample_space.shape, err_msg

    if n is None:
        n = 1

    return np.array([sample_space.sample() for i in range(n)])


def uniform_grid(sample_space: gym.spaces, n: "Sample size." = None):
    """
    Generate a grid of points.

    The bounds of the uniform grid are dermined by the sample_space. The space should be
    a bounded gym.spaces.Box, but if the space is unbounded (either above or below) in
    any dimension, the grid will be constrained to 1 in that dimension if unbounded
    above and -1 if unbounded below. For example, Given a 2D system with the first
    dimension in [a, oo) and the second dimension (-oo, b], then the resulting grid will
    be in the ranges [a, 1] in the first dimension and [-1, b] in the second.
    """

    if n is None:
        n = [1] * sample_space.shape[0]

    num_dims = sample_space.shape[0]

    low = sample_space.low
    high = sample_space.high

    xn = []
    for i in range(num_dims):
        bounded_below = -np.inf < low[i]
        bounded_above = np.inf > high[i]

        if bounded_above and bounded_below:
            xn.append(np.round(np.linspace(low[i], high[i], n[i]), 3))
        elif bounded_below:
            xn.append(np.round(np.linspace(low[i], 1, n[i]), 3))
        elif bounded_above:
            xn.append(np.round(np.linspace(-1, high[i], n[i]), 3))
        else:
            xn.append(np.round(np.linspace(-1, 1, n[i]), 3))

    x = list(itertools.product(*xn))

    return np.array(x), xn


def uniform_initial_conditions(
    system: DynamicalSystem, sample_space: gym.spaces, n: "Sample size." = None
):
    """Generate a collection of uniformly spaced initial conditions."""

    # assert sample space and observation space are the same dimensionality
    err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
    assert system.observation_space.shape == sample_space.shape, err_msg

    x, _ = uniform_grid(sample_space=sample_space, n=n)

    return np.array(x)


def sample(
    system: DynamicalSystem,
    initial_conditions: "Set of initial conditions." = None,
    policy: "Policy." = None,
) -> "S, U":
    """
    Generate a sample from a dynamical system.

    The sample consists of n state pairs (x_i, y_i) where x_i is a state vector sampled randomly from the 'sample_space', and y_i is the state vector at the next time step. In a deterministic system, this has the form:

        y = f(x, u)

    If no policy is specified, u is chosen randomly from the system's action space.
    """

    if system is None:
        return None, None

    if initial_conditions is None:
        initial_conditions = random_initial_conditions(
            system=system, sample_space=system.observation_space, n=1
        )

    elif not isinstance(initial_conditions, (list, np.ndarray)):
        return None, None

    if policy is None:
        policy = RandomizedPolicy(system)

    def generate_next_state(x0):
        system.state = x0

        action = policy(time=0, state=system.state)
        next_state, reward, done, _ = system.step(action)

        return (next_state, action)

    S, U = zip(*[generate_next_state(x0) for x0 in initial_conditions])
    S = [[x0, S[i]] for i, x0 in enumerate(initial_conditions)]
    U = np.expand_dims(U, axis=1)

    return np.array(S), np.array(U)


def sample_action(
    system: DynamicalSystem,
    initial_conditions: "Set of initial conditions." = None,
    action_set: "Action set." = None,
) -> "S, U":
    """
    Generate sample using each action in the action_set.
    """

    if system is None:
        return None, None

    if action_set is None:
        action_set = [system.action_space.sample() for i in range(3)]

    S = []
    U = []

    for action in action_set:
        _S, _U = sample(
            system=system,
            initial_conditions=initial_conditions,
            policy=ConstantPolicy(system, action),
        )

        if S == []:
            S = _S
            U = _U
        else:
            S = np.concatenate((S, _S), axis=0)
            U = np.concatenate((U, _U), axis=0)

    return S, U


def sample_trajectories(
    system: DynamicalSystem,
    initial_conditions: "Set of initial conditions." = None,
    policy: "Policy." = None,
) -> "S, U":
    """
    Generate a sample of trajectories from a dynamical system.

    The sample consists of n state trajectories (x_0, x_1, ..., x_N)_i, where x_0 are
    state vectors sampled randomly from the 'sample_space', x_1, ..., x_N are state
    vectors at subsequent time steps, determined by the system.sampling_time, and N is
    the system.time_horizon. If no policy is specified, the control actions u_0,
    ..., u_N-1 applied at each time step are chosen randomly from the system's action
    space.

    Parameters
    ----------


    Returns
    -------

    S : ndarray
        The array has the shape (n, t, d), where n is the number of samples, t is the number of time steps in [0, N], and d is the dimensionality of the sample space.

    U : ndarray

    """

    if system is None:
        return None, None

    if initial_conditions is None:
        initial_conditions = random_initial_conditions(
            system=system, sample_space=system.observation_space, n=1
        )
    elif not isinstance(initial_conditions, (list, np.ndarray)):
        return None, None

    if policy is None:
        policy = RandomizedPolicy(system)

    def generate_state_trajectory(x0):
        system.state = x0

        Xt = []
        Ut = []

        time = 0
        for t in range(system.num_time_steps):
            action = policy(time=time, state=system.state)
            next_state, reward, done, _ = system.step(action)

            Xt.append(next_state)
            Ut.append(action)

            time += 1

        # def generate_next_state():
        #     action = policy(time=0, state=system.state)
        #     next_state, reward, done, _ = system.step(action)
        #     return (next_state, action)
        #
        # Xt, Ut = zip(*[generate_next_state() for i in range(system.num_time_steps)])

        return (Xt, Ut)

    S, U = zip(*[generate_state_trajectory(x0) for x0 in initial_conditions])
    S = [[x0, *S[i]] for i, x0 in enumerate(initial_conditions)]

    return np.array(S), np.array(U)
