"""
Sampling Methods
"""
from abc import ABC, abstractmethod
import warnings

from inspect import isgenerator, isgeneratorfunction
from functools import partial, wraps
from itertools import islice, product

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import RandomizedPolicy
from gym_socks.envs.policy import ConstantPolicy

import numpy as np
from scipy.integrate import solve_ivp


def sample_generator(fun):
    @wraps(fun)
    def _wrapper(*args, **kwargs):
        while True:
            if isgeneratorfunction(fun):
                yield from fun(*args, **kwargs)
            else:
                yield fun(*args, **kwargs)

    return _wrapper


def step_sampler(system=None, policy=None, sample_space=None):
    """Default sampler (one step)."""

    @sample_generator
    def _sample_generator():
        state = sample_space.sample()
        action = policy(state=state)

        system.state = state
        next_state, cost, done, _ = system.step(action)

        yield (state, action, next_state)

    return _sample_generator


def uniform_grid(xi):
    """Create a uniform grid from a list of arrays."""
    return list(product(*xi))


def uniform_grid_step_sampler(xi, system=None, policy=None, sample_space=None):
    """Uniform sampler (one step)."""

    xc = uniform_grid(xi)

    @sample_generator
    def _sample_generator():
        for point in xc:
            state = point
            action = policy(state=state)

            system.state = state
            next_state, cost, done, _ = system.step(action)

            yield (state, action, next_state)

    return _sample_generator


def trajectory_sampler(system=None, policy=None, sample_space=None):
    """Default trajectory sampler."""

    @sample_generator
    def _sample_generator():
        state = sample_space.sample()

        state_sequence = []
        action_sequence = []

        system.state = state

        time = 0
        for t in range(system.num_time_steps):
            action = policy(time=time, state=system.state)
            next_state, cost, done, _ = system.step(action)

            state_sequence.append(next_state)
            action_sequence.append(action)

            time += 1

        yield (state, action_sequence, state_sequence)

    return _sample_generator


def sample(sampler=None, sample_size: int = None, *args, **kwargs):
    """Generate a sample using the sample generator."""

    if sampler is None:
        raise ValueError("Must supply a sample function.")

    if sample_size is None:
        raise ValueError("Must supply a sample size.")

    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("Sample size must be a positive integer.")

    return list(islice(sampler(*args, **kwargs), sample_size))


def transpose_sample(sample):
    """
    Transpose the sample.

    By default, a sample should be a list of tuples of the form:

        S = [(x_1, y_1), ..., (x_n, y_n)]

    For most algorithms, we need to isolate the sample components (e.g. all x's).
    This function converts a sample from a list of tuples to a tuple of lists:

        S_T = ([x_1, ..., x_n], [y_1, ..., y_n])

    This can then be unpacked as: X, Y = S_T

    Parameters
    ----------

    sample : list of tuples

    Returns
    -------

    S : tuple of lists
    """
    return tuple(map(list, zip(*sample)))


def reshape_trajectory_sample(sample):

    sample_size = len(sample)
    _S = transpose_sample(sample)

    return zip(*[np.reshape(item, (sample_size, -1)) for item in _S])


def validate_sample(sample):
    """
    Validate the sample.

    By default, a sample should be a list of tuples.

    We verify the following criteria:

    * Sample is a list of tuples.
    * All tuples should have the same length.
    * All tuple elements should be the same length (i.e. len(x_1) = ... = len(x_n))
    """
    err_msg = "Sample must be list of tuples."
    assert isinstance(sample, list), err_msg
    assert all(isinstance(item, tuple) for item in sample), err_msg

    # All tuples in sample must be the same length.
    assert all(len(item) == len(sample[0]) for item in sample), "Must be same len."

    # # All elements of tuples in sample must be the same length.
    # _tr = transpose_sample(sample)
    #
    # assert all(len(item) = len(sample) for item in sample)
    #
    # # for item in sample:
    # #     print(type(item))
    # #     print(len(item))
    # #     for element in item:
    # #         print(type(element))
    # #         print(len(element))
    # #         # break
    # #     break


# def sample(sample_space: gym.spaces.Space = None, n: int = None):
#
#     if sample_space is None:
#         raise ValueError("Must supply a sample_space.")
#
#     if n is None:
#         n = 1
#
#     if sample_space is gym.spaces.Box:
#         return _sample_box(sample_space=sample_space, n=n)
#     else:
#         return _sample_space(sample_space=sample_space, n=n)


# def _sample_space(sample_space: gym.spaces.Space = None, n: int = None):
#     """Sample from sample_space."""
#     return np.array([sample_space.sample() for i in range(n)])
#
#
# def _sample_box(sample_space: gym.spaces.Box = None, n: int = None):
#     """
#     Generate a random sample from a gym.spaces.Box
#
#     From the gym documentation, a different distribution is used depending on the
#     boundedness of the sampling space.
#
#     * [a, b] : uniform distribution
#     * [a, oo) : shifted exponential distribution
#     * (-oo, b] : negative shifted exponential distribution
#     * (-oo, oo) : normal distribution
#
#     See the numpy documentation for more information about the sampling distribution.
#     """
#
#     bounded_above = np.inf > sample_space.high
#     bounded_below = -np.inf < sample_space.low
#
#     unbounded = ~bounded_below & ~bounded_above
#     upper_bounded = ~bounded_below & bounded_above
#     lower_bounded = bounded_below & ~bounded_above
#     bounded = bounded_below & bounded_above
#
#     sample = np.empty((n, *sample_space.shape))
#
#     # unbounded
#     sample[:, unbounded] = sample_space.np_random.normal(
#         size=(n, *unbounded[unbounded].shape)
#     )
#
#     # upper bounded
#     sample[:, upper_bounded] = (
#         -sample_space.np_random.exponential(
#             size=(n, *upper_bounded[upper_bounded].shape)
#         )
#         + sample_space.high[upper_bounded]
#     )
#
#     # lower bounded
#     sample[:, lower_bounded] = (
#         sample_space.np_random.exponential(
#             size=(n, *lower_bounded[lower_bounded].shape)
#         )
#         + sample_space.low[lower_bounded]
#     )
#
#     # bounded
#     sample[:, bounded] = sample_space.np_random.uniform(
#         low=sample_space.low, high=sample_space.high, size=(n, *bounded[bounded].shape)
#     )
#
#     return sample.astype(sample_space.dtype)
#
#
# def sample_policy(policy=None, n=None):
#     """Sample from a provided policy."""
#
#
# def random_initial_conditions(
#     system: DynamicalSystem, sample_space: gym.spaces, n: "Sample size." = None
# ):
#     """
#     Generate a collection of random initial conditions.
#
#     As per the gym documentation, a different distribution is used depending on the
#     boundedness of the sampling space.
#
#     * [a, b] : uniform distribution
#     * [a, oo) : shifted exponential distribution
#     * (-oo, b] : negative shifted exponential distribution
#     * (-oo, oo) : normal distribution
#
#     See the numpy documentation for more information about the sampling distribution.
#     """
#
#     # assert sample space and observation space are the same dimensionality
#     err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
#     assert system.observation_space.shape == sample_space.shape, err_msg
#
#     if n is None:
#         n = 1
#
#     return np.array([sample_space.sample() for i in range(n)])


# def uniform_grid(sample_space: gym.spaces, n: "Sample size." = None):
#     """
#     Generate a grid of points.
#
#     The bounds of the uniform grid are dermined by the sample_space. The space should be
#     a bounded gym.spaces.Box, but if the space is unbounded (either above or below) in
#     any dimension, the grid will be constrained to 1 in that dimension if unbounded
#     above and -1 if unbounded below. For example, Given a 2D system with the first
#     dimension in [a, oo) and the second dimension (-oo, b], then the resulting grid will
#     be in the ranges [a, 1] in the first dimension and [-1, b] in the second.
#     """
#
#     if n is None:
#         n = [1] * sample_space.shape[0]
#
#     num_dims = sample_space.shape[0]
#
#     low = sample_space.low
#     high = sample_space.high
#
#     xn = []
#     for i in range(num_dims):
#         bounded_below = -np.inf < low[i]
#         bounded_above = np.inf > high[i]
#
#         if bounded_above and bounded_below:
#             xn.append(np.round(np.linspace(low[i], high[i], n[i]), 3))
#         elif bounded_below:
#             xn.append(np.round(np.linspace(low[i], 1, n[i]), 3))
#         elif bounded_above:
#             xn.append(np.round(np.linspace(-1, high[i], n[i]), 3))
#         else:
#             xn.append(np.round(np.linspace(-1, 1, n[i]), 3))
#
#     x = list(product(*xn))
#
#     return np.array(x), xn


# def uniform_initial_conditions(
#     system: DynamicalSystem, sample_space: gym.spaces, n: "Sample size." = None
# ):
#     """Generate a collection of uniformly spaced initial conditions."""
#
#     # assert sample space and observation space are the same dimensionality
#     err_msg = "%r (%s) invalid shape" % (sample_space, type(sample_space))
#     assert system.observation_space.shape == sample_space.shape, err_msg
#
#     x, _ = uniform_grid(sample_space=sample_space, n=n)
#
#     return np.array(x)
#
#
# def sample(
#     system: DynamicalSystem,
#     initial_conditions: "Set of initial conditions." = None,
#     policy: "Policy." = None,
# ) -> "S, U":
#     """
#     Generate a sample from a dynamical system.
#
#     The sample consists of n state pairs (x_i, y_i) where x_i is a state vector sampled randomly from the 'sample_space', and y_i is the state vector at the next time step. In a deterministic system, this has the form:
#
#         y = f(x, u)
#
#     If no policy is specified, u is chosen randomly from the system's action space.
#     """
#
#     if system is None:
#         return None, None
#
#     if initial_conditions is None:
#         initial_conditions = random_initial_conditions(
#             system=system, sample_space=system.observation_space, n=1
#         )
#
#     elif not isinstance(initial_conditions, (list, np.ndarray)):
#         return None, None
#
#     if policy is None:
#         policy = RandomizedPolicy(system)
#
#     def generate_next_state(x0):
#         system.state = x0
#
#         action = policy(time=0, state=system.state)
#         next_state, reward, done, _ = system.step(action)
#
#         return (next_state, action)
#
#     S, U = zip(*[generate_next_state(x0) for x0 in initial_conditions])
#     S = [[x0, S[i]] for i, x0 in enumerate(initial_conditions)]
#     U = np.expand_dims(U, axis=1)
#
#     return np.array(S), np.array(U)
#
#
# def sample_action(
#     system: DynamicalSystem,
#     initial_conditions: "Set of initial conditions." = None,
#     action_set: "Action set." = None,
# ) -> "S, U":
#     """
#     Generate sample using each action in the action_set.
#     """
#
#     if system is None:
#         return None, None
#
#     if action_set is None:
#         action_set = [system.action_space.sample() for i in range(3)]
#
#     S = []
#     U = []
#
#     for action in action_set:
#         _S, _U = sample(
#             system=system,
#             initial_conditions=initial_conditions,
#             policy=ConstantPolicy(system, action),
#         )
#
#         if S == []:
#             S = _S
#             U = _U
#         else:
#             S = np.concatenate((S, _S), axis=0)
#             U = np.concatenate((U, _U), axis=0)
#
#     return S, U
#
#
# def sample_trajectories(
#     system: DynamicalSystem,
#     initial_conditions: "Set of initial conditions." = None,
#     policy: "Policy." = None,
# ) -> "S, U":
#     """
#     Generate a sample of trajectories from a dynamical system.
#
#     The sample consists of n state trajectories (x_0, x_1, ..., x_N)_i, where x_0 are
#     state vectors sampled randomly from the 'sample_space', x_1, ..., x_N are state
#     vectors at subsequent time steps, determined by the system.sampling_time, and N is
#     the system.time_horizon. If no policy is specified, the control actions u_0,
#     ..., u_N-1 applied at each time step are chosen randomly from the system's action
#     space.
#
#     Parameters
#     ----------
#
#
#     Returns
#     -------
#
#     S : ndarray
#         The array has the shape (n, t, d), where n is the number of samples, t is the number of time steps in [0, N], and d is the dimensionality of the sample space.
#
#     U : ndarray
#
#     """
#
#     if system is None:
#         return None, None
#
#     if initial_conditions is None:
#         initial_conditions = random_initial_conditions(
#             system=system, sample_space=system.observation_space, n=1
#         )
#     elif not isinstance(initial_conditions, (list, np.ndarray)):
#         return None, None
#
#     if policy is None:
#         policy = RandomizedPolicy(system)
#
#     def generate_state_trajectory(x0):
#         system.state = x0
#
#         Xt = []
#         Ut = []
#
#         time = 0
#         for t in range(system.num_time_steps):
#             action = policy(time=time, state=system.state)
#             next_state, reward, done, _ = system.step(action)
#
#             Xt.append(next_state)
#             Ut.append(action)
#
#             time += 1
#
#         return (Xt, Ut)
#
#     S, U = zip(*[generate_state_trajectory(x0) for x0 in initial_conditions])
#     S = [[x0, *S[i]] for i, x0 in enumerate(initial_conditions)]
#
#     return np.array(S), np.array(U)
