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
from itertools import islice, product

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
        ... def custom_sampler(system, policy, sample_space):
        ...     system.state = sample_space.sample()
        ...     action = policy(state=state)
        ...     next_state, *_ = system.step(action)
        ...     yield (system.state, action, next_state)
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


def step_sampler(
    system: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
):
    """Default sampler (one step).

    Args:
        system: The system to sample from.
        policy: The policy applied to the system during sampling.
        sample_space: The space where initial conditions are drawn from.

    Returns:
        A generator function that yields system observations as tuples.

    """

    @sample_generator
    def _sample_generator():
        state = sample_space.sample()
        action = policy(state=state)

        system.state = state
        next_state, cost, done, _ = system.step(action)

        yield (state, action, next_state)

    return _sample_generator


def uniform_grid(xi):
    """Create a uniform grid from a list of ranges.

    Args:
        xi: List of ranges.

    Returns:
        Grid of points (the product of all points in ranges).

    Example:

        >>> import numpy as np
        >>> from gym_socks.envs.sample import uniform_grid
        >>> grid = uniform_grid([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)])

    """

    return list(product(*xi))


def uniform_grid_step_sampler(
    xi: list,
    system: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
):
    """Uniform sampler (one step).

    Args:
        xi: List of ranges.
        system: The system to sample from.
        policy: The policy applied to the system during sampling.
        sample_space: The space where initial conditions are drawn from.

    Returns:
        A generator function that yields system observations as tuples.

    """

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


def sequential_action_sampler(
    ui: list,
    env: DynamicalSystem,
    sample_space: gym.Space,
    sampler,
):

    uc = uniform_grid(ui)

    @sample_generator
    def _wrapper(*args, **kwargs):
        for action in uc:
            _policy = ConstantPolicy(env, constant=action)

            sampler_instance = sampler(
                system=env,
                policy=_policy,
                sample_space=sample_space,
            )

            yield sampler_instance(*args, **kwargs)

    return _wrapper


def trajectory_sampler(
    system: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
):
    """Default trajectory sampler.

    Args:
        system: The system to sample from.
        policy: The policy applied to the system during sampling.
        sample_space: The space where initial conditions are drawn from.

    Returns:
        A generator function that yields system observations as tuples.

    """

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


def transpose_sample(sample):
    """Transpose the sample.

    By default, a sample should be a list of tuples of the form::

        S = [(x_1, y_1), ..., (x_n, y_n)]

    For most algorithms, we need to isolate the sample components (e.g. all x's).
    This function converts a sample from a list of tuples to a tuple of lists::

        S_T = ([x_1, ..., x_n], [y_1, ..., y_n])

    This can then be unpacked as: ``X, Y = S_T``

    Args:
        sample: list of tuples

    Returns:
        tuple of lists

    """

    return tuple(map(list, zip(*sample)))


def reshape_trajectory_sample(sample):
    """Reshapes trajectory samples.

    Often, trajectory samples are organized such that the "trajectory" components are a
    2D array of points indexed by time. However, for kernel methods, we typically
    require that the trajectories be concatenated into a single vector (1D array)::

        [[x1], [x2], ..., [xn]] -> [x1, x2, ..., xn]

    This function converts the sample so that the trajectories are 1D arrays.

    Args:
        sample: list of tuples

    Returns:
        List of tuples, where the components of the tuples are flattened.

    """
    sample_size = len(sample)
    _S = transpose_sample(sample)

    return zip(*[np.reshape(item, (sample_size, -1)) for item in _S])
