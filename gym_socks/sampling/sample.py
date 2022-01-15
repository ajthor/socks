"""Sampling methods."""

from inspect import isgeneratorfunction
from functools import partial, wraps
from itertools import islice

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem


def sample_generator(fun):
    """Sample generator decorator.

    Converts a sample function into a generator function. Any function that returns a
    single observation (as a tuple) can be converted into a sample generator.

    Args:
        fun: Sample function that returns or yields an observation.

    Returns:
        A function that can be used to ``islice`` a sample from the sample generator.

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

    Returns a random sample taken from the space. See :py:obj:`gym.spaces.Box` for more
    information on the distributions used for sampling.

    Args:
        sample_space: The space to sample from.

    Yields:
        A sample from the ``sample_space``.

    """

    yield sample_space.sample()


@sample_generator
def grid_sampler(grid: list):
    """Grid sampler.

    Returns a sample arranged on a uniformly-spaced grid. Use
    :py:func:`make_grid_from_ranges` or :py:func:`make_grid_from_space` from
    :py:mod:`gym_socks.utils.grid` to generate a grid of points.

    Args:
        grid: The grid of points.

    Yields:
        A sample from the ``sample_space``.

    """

    for item in grid:
        yield item


@sample_generator
def repeat(sampler, num: int):
    """Repeat sampler.

    Repeats the output of a sample generator ``num`` times.

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
