"""Sampling methods."""

from inspect import isgeneratorfunction
from functools import wraps
from itertools import islice
from collections.abc import Generator

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


def _repeat(generator, num: int = 1):
    """Repeat sampler.

    Repeats the output of a sample generator ``num`` times.

    Args:
        generator: The sample generator function.
        num: The number of times to repeat a sample.

    Yields:
        A repeated sample.

    """

    while True:
        for item in generator:
            for _ in range(num):
                yield item


class _SampleGenerator(Generator):
    """Sample generator object.

    This class wraps a generator created via :py:func:`sample_fn` and adds
    useful helper functions for manipulating the sampling function.

    Args:
        generator: The generator function to wrap.

    """

    def __init__(self, generator):
        self._generator = generator

    def send(self, value):
        return self._generator.send(value)

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        return False

    def repeat(self, num: int = 1):
        """Modify the generator to repeat the output ``num`` times.

        See: :py:func:`_repeat` for more information.

        Args:
            num: The number of times to repeat the output before continuing.

        Returns:
            An instance of the :py:class:`_SampleGenerator` object.

        """

        self._generator = _repeat(self._generator, num)
        return self

    def sample(self, size: int = 1):
        """Generate a sample from the generator of a certain size."""

        return list(islice(self._generator, size))


def sample_fn(fn):
    """Sample function decorator.

    Converts a sample function into a sample generator.

    Args:
        fn: Sample function that returns or yields an observation.

    Returns:
        A function that can be used to ``islice`` a sample from the sample generator.

    Example:

        >>> from gym_socks.envs.sample import sample_fn
        >>> @sample_fn
        ... def custom_sampler(env, policy):
        ...     state = env.reset()
        ...     action = policy(state=state)
        ...     next_state, *_ = env.step(action)
        ...     yield state, action, next_state
        >>> sampler = custom_sampler(env, policy)
        >>> S = sampler.sample(size=100)

    """

    # The inner wrapper protects in case the wrapped function is a generator.
    if isgeneratorfunction(fn):

        @wraps(fn)
        def _inner_wrapper(*args, **kwargs):
            while True:
                yield from fn(*args, **kwargs)

    else:

        @wraps(fn)
        def _inner_wrapper(*args, **kwargs):
            while True:
                yield fn(*args, **kwargs)

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        return _SampleGenerator(_inner_wrapper(*args, **kwargs))

    return _wrapper


@sample_fn
def space_sampler(space: gym.spaces.Space):
    """Randomly sample from a space.

    Args:
        space: A :py:obj:`gym.space.Space` that implements a :py:func:`sample` function.

    Yields:
        A random sample from the space.

    """

    yield space.sample()


@sample_fn
def grid_sampler(grid_points: np.ndarray):
    """Sample from a set of pre-defined grid points.

    Hint:
        Use :py:func:`boxgrid` or :py:func:`cartesian` from
        :py:mod:`gym_socks.utils.grid` to generate a grid of points.

    Args:
        grid_points: A collection of grid points in a list or array.

    Yields:
        A point in the grid. Note that points are yielded in the order they are given.

    """

    for item in grid_points:
        yield item


@sample_fn
def transition_sampler(
    env: gym.Env,
    state_sampler,
    action_sampler,
):
    """Transition sampler."""

    initial_condition = next(state_sampler)
    env.reset(initial_condition)

    action = next(action_sampler)
    state, *_ = env.step(action=action)

    yield initial_condition, action, state


@sample_fn
def trajectory_sampler(
    env: gym.Env,
    state_sampler,
    action_sampler,
    time_horizon: int = 1,
):
    """Trajectory sampler."""

    initial_condition = next(state_sampler)
    env.reset(initial_condition)

    state_sequence = []
    action_sequence = []

    for t in range(time_horizon):
        action = next(action_sampler)
        state, *_ = env.step(time=t, action=action)

        state_sequence.append(state)
        action_sequence.append(action)

    yield initial_condition, np.array(action_sequence), np.array(state_sequence)
