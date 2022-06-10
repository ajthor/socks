"""Sampling methods.

This module contains a collection of sampling methods. The core principle is to define a
function that returns a single observation (either via return or yield) from a
probability measure. Then, using the decorator ``sample_generator``, a function that
returns a single observation can be converted into a generator, that can then be sampled
using ``islice``.

Example:

    Sample the stochastic kernel of a dynamical system (i.e. the state transition
    probability kernel).

        >>> from gym_socks.envs.integrator import NDIntegratorEnv
        >>> from gym_socks.policies import RandomizedPolicy
        >>> env = NDIntegratorEnv(2)
        >>> sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
        >>> sampler = default_sampler(
        ...     system=env, policy=RandomizedPolicy(env), sample_space=sample_space
        ... )
        >>> S = sample(sampler=sampler, sample_size=100)

The main reason for this setup is to allow for "observation functions" which have
different structures, e.g.

* functions that ``return`` an observation,
* infinite generators that ``yield`` observations, and
* finite generators that ``yield`` observations.

"""

from gym_socks.sampling.sample import sample_fn

from gym_socks.sampling.sample import space_sampler
from gym_socks.sampling.sample import grid_sampler

from gym_socks.sampling.sample import transition_sampler
from gym_socks.sampling.sample import trajectory_sampler


__all__ = [
    "sample_fn",
    "space_sampler",
    "grid_sampler",
    "transition_sampler",
    "trajectory_sampler",
]
