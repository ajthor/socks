"""
Sampling Methods
"""
from abc import ABC, abstractmethod

import itertools

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem

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


def uniform_initial_conditions(
    system: DynamicalSystem, sample_space: gym.spaces, n: "Sample size." = None
):
    """Generate a collection of uniformly spaced initial conditions."""

    # assert sample space and observation space are the same dimensionality
    err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
    assert system.observation_space.shape == sample_space.shape, err_msg

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

    # xn = [
    #     np.round(np.linspace(low[i], high[i], n[i]), 3)
    #     for i in range(num_dims)
    # ]

    x = list(itertools.product(*xn))

    return np.array(x)


def generate_sample(
    system: DynamicalSystem,
    initial_conditions: "Set of initial conditions." = None,
    controller: "Controller." = None,
) -> "S, U":
    """
    Generate a sample from a dynamical system.

    The sample consists of n state pairs (x_i, y_i) where x_i is a state vector sampled randomly from the 'sample_space', and y_i is the state vector at the next time step. In a deterministic system, this has the form:

        y = f(x, u)

    If no controller is specified, u is chosen randomly from the system's action space.
    """

    if system is None:
        return None, None

    if initial_conditions is None:
        initial_conditions = random_initial_conditions(
            system=system, sample_space=system.observation_space, n=1
        )

    elif not isinstance(initial_conditions, (list, np.ndarray)):
        return None, None

    if controller is None:

        def random_controller(state):
            return system.action_space.sample()

        controller = random_controller

    def generate_next_state(x0):
        system.state = x0

        action = controller(state=system.state)
        next_state, reward, done, _ = system.step(action)

        return (next_state, action)

    S, U = zip(*[generate_next_state(x0) for x0 in initial_conditions])
    S = [(x0, S[i]) for i, x0 in enumerate(initial_conditions)]
    U = np.expand_dims(U, axis=1)

    # def generate_next_state(x0):
    #     system.state = x0
    #
    #     action = controller(state=system.state)
    #     next_state, reward, done, _ = system.step(action)
    #
    #     return next_state, action
    #
    # result = []
    # for x0 in initial_conditions:
    #     x1, u0 = generate_next_state(x0)
    #     result.append(((x0, x1), u0))
    #
    # S, U = zip(*result)

    return np.array(S), np.array(U)


# def generate_uniform_sample(
#     sample_space: gym.spaces,
#     system: DynamicalSystem,
#     controller: "Controller." = None,
#     n: "Number of samples." = 10,
# ):
#     """
#     Generate a uniform sample from a dynamical system.
#     """
#
#     # assert sample space and observation space are the same dimensionality
#     err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
#     assert system.observation_space.shape == sample_space.shape, err_msg
#
#     if controller is None:
#         def random_controller(state):
#             return system.action_space.sample()
#
#         controller = random_controller
#
#
#     # generate initial conditions
#     x = generate_uniform_initial_conditions(sample_space, system, n)
#
#     def generate_next_state(x0):
#         system.state = x0
#
#         action = controller(state=system.state)
#         next_state, reward, done, _ = system.step(action)
#
#         return next_state
#
#     S = np.array([(x0, generate_next_state(x0)) for x0 in x])
#
#     return S, np.array(x)


def generate_sample_trajectories(
    system: DynamicalSystem,
    initial_conditions: "Set of initial conditions." = None,
    controller: "Controller." = None,
) -> "S, U":
    """
    Generate a sample of trajectories from a dynamical system.

    The sample consists of n state trajectories (x_0, x_1, ..., x_N)_i, where x_0 are
    state vectors sampled randomly from the 'sample_space', x_1, ..., x_N are state
    vectors at subsequent time steps, determined by the system.sampling_time, and N is
    the system.time_horizon. If no controller is specified, the control actions u_0,
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

    if controller is None:

        def random_controller(state):
            return system.action_space.sample()

        controller = random_controller

    def generate_state_trajectory(x0):
        system.state = x0

        def generate_next_state():
            action = controller(state=system.state)
            next_state, reward, done, _ = system.step(action)
            return (next_state, action)

        Xt, Ut = zip(*[generate_next_state() for i in range(system.num_time_steps)])

        return (Xt, Ut)

    S, U = zip(*[generate_state_trajectory(x0) for x0 in initial_conditions])
    S = [[x0, *S[i]] for i, x0 in enumerate(initial_conditions)]

    return np.array(S), np.array(U)

    # def generate_state_trajectory(x0):
    #     system.state = x0
    #
    #     def generate_next_state():
    #         action = controller(state=system.state)
    #         next_state, reward, done, _ = system.step(action)
    #         return next_state, action
    #
    #     Xt = []
    #     Ut = []
    #     for i in range(system.num_time_steps):
    #         Xtemp, Utemp = generate_next_state()
    #         Xt.append(Xtemp)
    #         Ut.append(Utemp)
    #
    #     return Xt, Ut
    #
    # S = []
    # U = []
    # for x0 in initial_conditions:
    #     Tx, Tu = generate_state_trajectory(x0)
    #     S.append(Tx)
    #     U.append(Tu)
    #
    # return np.array(S), np.array(U)

    # def generate_state_trajectory(x0):
    #     system.state = x0
    #
    #     def generate_next_state():
    #         action = controller(state=system.state)
    #         next_state, reward, done, _ = system.step(action)
    #         return next_state, action
    #
    #     return np.array([generate_next_state() for i in range(system.num_time_steps)])

    # print(generate_state_trajectory(initial_conditions[0]))
    # T = [zip(*generate_state_trajectory(x0)) for x0 in initial_conditions]
    # Xt, Ut = *T
    # S = [(x0, *generate_state_trajectory(x0)) for x0 in initial_conditions]

    # return np.array(S)
