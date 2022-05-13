"""Sampling methods."""

from inspect import isgeneratorfunction
from functools import wraps
from functools import update_wrapper
from itertools import islice
from collections.abc import Generator

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

# class SampleGenerator:
#     def __init__(self, fun):

#         self._fun = fun
#         self._is_generator = isgeneratorfunction(fun)

#         if self._is_generator is True:

#             def _wrapper(*args, **kwargs):
#                 while True:
#                     yield from fun(*args, **kwargs)

#         else:

#             def _wrapper(*args, **kwargs):
#                 while True:
#                     yield fun(*args, **kwargs)

#         self._wrapper = _wrapper

#     def __call__(self, *args, **kwargs):
#         self._generator = self._wrapper(*args, **kwargs)
#         return _SampleGenerator(self._generator)


# class _SampleGenerator:
#     def __init__(self, gen):
#         self._generator = gen

#     def __iter__(self):
#         return self._generator

#     def __next__(self):
#         return next(self._generator)

#     def sample(self, size: int = 1):
#         return list(islice(self._generator, size))


# class _SampleGeneratorWrapper(Generator):
#     def __init__(self, generator):
#         self._generator = generator

#     def send(self, value):
#         return self._generator.send(value)

#     def throw(self, type=None, value=None, traceback=None):
#         raise StopIteration

#     def sample(self, size=1):
#         return list(islice(self._generator, size))

# class _SampleGenerator:
#     def __init__(self, generator):
#         self._generator = generator
#         update_wrapper(self, generator)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return next(self._generator)

#     def repeat(self, num: int = 1):
#         self._generator = _repeat(self._generator, num)
#         return self

#     def sample(self, size: int = 1):
#         return list(islice(self._generator, size))


# class SampleGenerator:
#     def __init__(self, fun):

#         self._is_generator = isgeneratorfunction(fun)
#         if self._is_generator is True:

#             @wraps(fun)
#             def _wrapper(*args, **kwargs):
#                 while True:
#                     yield from fun(*args, **kwargs)

#         else:

#             @wraps(fun)
#             def _wrapper(*args, **kwargs):
#                 while True:
#                     yield fun(*args, **kwargs)

#         self._fun = _wrapper

#         update_wrapper(self, fun)
#         self.__call__ = wraps(fun)

#     def __call__(self, *args, **kwargs):
#         generator = self._fun(*args, **kwargs)
#         return _SampleGenerator(generator)


# # class SampleGenerator:
# #     def __init__(self):
# #         pass

# #     def __call__(self, fun):
# #         self._is_generator = isgeneratorfunction(fun)
# #         if self._is_generator is True:

# #             @wraps(fun)
# #             def _wrapper(*args, **kwargs):
# #                 while True:
# #                     yield from fun(*args, **kwargs)

# #         else:

# #             @wraps(fun)
# #             def _wrapper(*args, **kwargs):
# #                 while True:
# #                     yield fun(*args, **kwargs)

# #         return _SampleGenerator(_wrapper)


# def _generator_function_wrapper(fun):
#     while True:
#         yield from fun()


# def _function_wrapper(fun):
#     while True:
#         yield fun()


# class SampleGenerator:
#     def __init__(self, fun):

#         self._fun = fun

#         if isgeneratorfunction(fun):

#             def _wrapper(*args, **kwargs):
#                 while True:
#                     yield from fun(*args, **kwargs)

#             self._generator = _wrapper

#             def _fun_wrapper(*args, **kwargs):
#                 return next(fun(*args, **kwargs))

#             # self._fun = _fun_wrapper

#         else:

#             def _wrapper(*args, **kwargs):
#                 while True:
#                     yield fun(*args, **kwargs)

#             self._generator = _wrapper
#             # self._fun = fun

#         # self._generator.send(None)

#     def __next__(self):
#         return next(self._generator)

#     def __iter__(self):
#         return self._generator

#     def __call__(self, *args, **kwargs):
#         return self._fun(*args, **kwargs)

#     def sample(self, size):
#         return list(islice(self._generator, size))


# def sample_generator(fun):
#     """Sample generator decorator.

#     Converts a sample function into a generator function. Any function that returns a
#     single observation (as a tuple) can be converted into a sample generator.

#     Args:
#         fun: Sample function that returns or yields an observation.

#     Returns:
#         A function that can be used to ``islice`` a sample from the sample generator.

#     Example:

#         >>> from itertools import islice
#         >>> from gym_socks.envs.sample import sample_generator
#         >>> @sample_generator
#         ... def custom_sampler(env, policy, sample_space):
#         ...     env.reset()
#         ...     action = policy(state=state)
#         ...     next_state, *_ = env.step(action)
#         ...     yield (env.state, action, next_state)
#         >>> S = list(islice(custom_sampler(), 100))

#     """

#     if isgeneratorfunction(fun):

#         @wraps(fun)
#         def _wrapper(*args, **kwargs):
#             while True:
#                 yield from fun(*args, **kwargs)

#     else:

#         @wraps(fun)
#         def _wrapper(*args, **kwargs):
#             while True:
#                 yield fun(*args, **kwargs)

#     return _wrapper


# @sample_generator
# def random_sampler(sample_space: gym.Space):
#     """Random sampler.

#     Returns a random sample taken from the space. See :py:obj:`gym.spaces.Box` for more
#     information on the distributions used for sampling.

#     Args:
#         sample_space: The space to sample from.

#     Yields:
#         A sample from the ``sample_space``.

#     """

#     yield sample_space.sample()


# @sample_generator
# def grid_sampler(grid: list):
#     """Grid sampler.

#     Returns a sample arranged on a uniformly-spaced grid.
#     Use :py:func:`boxgrid` or :py:func:`cartesian` from
#     :py:mod:`gym_socks.utils.grid` to generate a grid of points.

#     Args:
#         grid: The grid of points.

#     Yields:
#         A sample from the ``sample_space``.

#     """

#     for item in grid:
#         yield item


# @sample_generator
# def repeat(sampler, num: int):
#     """Repeat sampler.

#     Repeats the output of a sample generator ``num`` times.

#     Args:
#         sampler: The sample generator function.
#         num: The number of times to repeat a sample.

#     Yields:
#         A repeated sample.

#     """

#     for item in sampler:
#         for i in range(num):
#             yield item


# def default_sampler(
#     state_sampler=None,
#     action_sampler=None,
#     env: DynamicalSystem = None,
# ):
#     """Default trajectory sampler.

#     Args:
#         state_sampler: The state space sampler.
#         action_sampler: The action space sampler.
#         env: The system to sample from.

#     Returns:
#         A generator function that yields system observations as tuples.

#     """

#     @sample_generator
#     def _sample_generator():
#         state = next(state_sampler)
#         action = next(action_sampler)

#         env.reset(state)
#         next_state, *_ = env.step(action=action)

#         yield (state, action, next_state)

#     return _sample_generator


# def default_trajectory_sampler(
#     state_sampler=None,
#     action_sampler=None,
#     env: DynamicalSystem = None,
#     time_horizon: int = 1,
# ):
#     """Default trajectory sampler.

#     Args:
#         state_sampler: The state space sampler.
#         action_sampler: The action space sampler.
#         env: The system to sample from.
#         time_horizon: The time horizon to simulate over.

#     Returns:
#         A generator function that yields system observations as tuples.

#     """

#     @sample_generator
#     def _sample_generator():
#         state = next(state_sampler)

#         state_sequence = []
#         action_sequence = []

#         env.reset(state)

#         time = 0
#         for t in range(time_horizon):
#             action = next(action_sampler)
#             next_state, *_ = env.step(time=t, action=action)

#             state_sequence.append(next_state)
#             action_sequence.append(action)

#             time += 1

#         yield (state, action_sequence, state_sequence)

#     return _sample_generator


# def sample(sampler=None, sample_size: int = None) -> list:
#     """Generate a sample using the sample generator.

#     Args:
#         sampler: Sample generator function.
#         sample_size: Size of the sample.

#     Returns:
#         list of tuples

#     """

#     if sampler is None:
#         raise ValueError("Must supply a sample function.")

#     if sample_size is None:
#         raise ValueError("Must supply a sample size.")

#     if not isinstance(sample_size, int) or sample_size < 1:
#         raise ValueError("Sample size must be a positive integer.")

#     return list(islice(sampler, sample_size))


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

    This class wraps a generator created via :py:func:`SampleGenerator` and adds
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
