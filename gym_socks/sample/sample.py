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
    env: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
):
    """Default sampler (one step).

    Args:
        env: The system to sample from.
        policy: The policy applied to the system during sampling.
        sample_space: The space where initial conditions are drawn from.

    Returns:
        A generator function that yields system observations as tuples.

    """

    @sample_generator
    def _sample_generator():
        state = sample_space.sample()
        action = policy(state=state)

        env.state = state
        next_state, cost, done, _ = env.step(action=action)

        yield (state, action, next_state)

    return _sample_generator


def uniform_grid_step_sampler(
    xi: list,
    env: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
):
    """Uniform sampler (one step).

    Args:
        xi: List of ranges.
        env: The system to sample from.
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

            env.state = state
            next_state, cost, done, _ = env.step(action=action)

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
                env=env,
                policy=_policy,
                sample_space=sample_space,
            )

            yield sampler_instance(*args, **kwargs)

    return _wrapper


def trajectory_sampler(
    time_horizon: int,
    env: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
):
    """Default trajectory sampler.

    Args:
        env: The system to sample from.
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

        env.state = state

        time = 0
        for t in range(time_horizon):
            action = policy(time=time, state=env.state)
            next_state, cost, done, _ = env.step(time=t, action=action)

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
