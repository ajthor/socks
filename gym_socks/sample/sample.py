"""Sampling methods.

This file contains a collection of sampling methods. The core principle is to define a
function that returns a single observation (either via return or yield) from a
probability measure. Then, the `sample_generator` is a decorator, which converts a
function that returns a single observation into a generator, that can be sampled using
`islice`.

Example:
    Sample the stochastic kernel of a dynamical system (i.e. the state transition
    probability kernel).

        >>> env = NdIntegrator(2)
        >>> sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        >>> sampler = step_sampler(
        ...     system=env, policy=RandomizedPolicy(env), sample_space=sample_space
        ... )
        >>> S = sample(sampler=sampler, sample_size=100)

The main reason for this setup is to allow for 'observation functions' which have
different structures, e.g.
* a function which `return`s an observation,
* an infinite generator which `yield`s observations, and
* a finite generator that `yield`s observations.

"""

from inspect import isgeneratorfunction
from functools import partial, wraps
from itertools import islice

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import BasePolicy
from gym_socks.envs.policy import RandomizedPolicy
from gym_socks.envs.policy import ConstantPolicy

import numpy as np
from scipy.integrate import solve_ivp


def sample_generator(fun):
    """Sample generator decorator.

    Converts a sample function into a generator function. Any function that returns a
    single observation (as a tuple) can be converted into a sample generator.

    Args:
        fun: Sample function that returns or yields an observation.

    Returns:
        A function that can be used to `islice` a sample from the sample generator.

    Example:

        >>> from itertools import islice
        >>> from gym_socks.envs.sample import sample_generator
        >>> @sample_generator
        ... def custom_sampler(env, policy, sample_space):
        ...     env.state = sample_space.sample()
        ...     action = policy(state=state)
        ...     next_state, *_ = env.step(action)
        ...     yield (env.state, action, next_state)
        >>> S = list(islice(custom_sampler(), 100))

    """

    @wraps(fun)
    def _wrapper(*args, **kwargs):
        while True:
            if isgeneratorfunction(fun):
                yield from fun(*args, **kwargs)
            else:
                yield fun(*args, **kwargs)

    return _wrapper


@sample_generator
def random_sampler(sample_space: gym.Space):
    """Random sampler.

    Returns a random sample taken from the space. See `gym.spaces.Box` for more
    information on the distributions used for sampling.

    Args:
        sample_space: The space to sample from.

    Yields:
        A sample from the `sample_space`.

    """

    yield sample_space.sample()


@sample_generator
def grid_sampler(grid: list):
    """Grid sampler.

    Returns a sample arranged on a uniformly-spaced grid. Use `make_grid_from_ranges` or
    `make_grid_from_space` from `gym_socks.utils.grid` to generate a grid of points.

    Args:
        grid: The grid of points.

    Yields:
        A sample from the `sample_space`.

    """

    for item in grid:
        yield item


@sample_generator
def repeat(sampler, num: int):
    """Repeat sampler.

    Repeats the output of a sample generator `num` times.

    Args:
        sampler: The sample generator function.
        num: The number of times to repeat a sample.

    Yields:
        A repeated sample.

    """

    for item in sampler:
        for i in range(num):
            yield item


def default_sampler(
    state_sampler=None,
    action_sampler=None,
    env: DynamicalSystem = None,
):
    """Default trajectory sampler.

    Args:
        state_sampler: The state space sampler.
        action_sampler: The action space sampler.
        env: The system to sample from.

    Returns:
        A generator function that yields system observations as tuples.

    """

    @sample_generator
    def _sample_generator():
        state = next(state_sampler)
        action = next(action_sampler)

        env.state = state
        next_state, *_ = env.step(action=action)

        yield (state, action, next_state)

    return _sample_generator


def default_trajectory_sampler(
    state_sampler=None,
    action_sampler=None,
    env: DynamicalSystem = None,
    time_horizon: int = 1,
):
    """Default trajectory sampler.

    Args:
        state_sampler: The state space sampler.
        action_sampler: The action space sampler.
        env: The system to sample from.
        time_horizon: The time horizon to simulate over.

    Returns:
        A generator function that yields system observations as tuples.

    """

    @sample_generator
    def _sample_generator():
        state = next(state_sampler)

        state_sequence = []
        action_sequence = []

        env.state = state

        time = 0
        for t in range(time_horizon):
            action = next(action_sampler)
            next_state, *_ = env.step(time=t, action=action)

            state_sequence.append(next_state)
            action_sequence.append(action)

            time += 1

        yield (state, action_sequence, state_sequence)

    return _sample_generator


def sample(sampler=None, sample_size: int = None, *args, **kwargs):
    """Generate a sample using the sample generator.

    Args:
        sampler: Sample generator function.
        sample_size: Size of the sample.

    Returns:
        list of tuples

    """

    if sampler is None:
        raise ValueError("Must supply a sample function.")

    if sample_size is None:
        raise ValueError("Must supply a sample size.")

    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("Sample size must be a positive integer.")

    return list(islice(sampler(*args, **kwargs), sample_size))
