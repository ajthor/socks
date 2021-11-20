from abc import ABC, abstractmethod

import gym
from gym.utils import seeding

import numpy as np

from functools import wraps


class BaseDynamicalObject(gym.Env, ABC):
    """Base dynamical object class.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from BaseDynamicalObject.

    This class serves as the base interface for dynamical objects, represented in most
    cases by either a `DynamicalSystem` or an obstacle.

    """

    @abstractmethod
    def step(self):
        """Advance the object forward in time.

        Advances the object in the simulation. By default, an object is uncontrolled,
        meaning it accepts no parameters and the system evolves forward in time
        according to its own, internal dynamics. Controlled systems should accept an
        `action` parameter, which represents the user-selected control input.

        Additionally, time-varying systems should also accept a `time` parameter.

        """

        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the object to an initial state.

        Resets the object to an initial state (which may be random).

        Returns:
            The new initial state of the object.

        """

        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        """Sets the seed of the random number generator.

        This is primarily useful for objects which incorporate some sort of
        stochasticity to ensure repeatability.

        Args:
            seed: Integer value representing the random seed.

        Returns:
            The seed of the RNG.

        """

        return


def pre_hook_wrapper(pre_hook):
    def _outer_wrapper(fun):
        @wraps(fun)
        def _wrapper(*args, **kwargs):
            pre_hook()
            result = fun(*args, **kwargs)
            return result

        return _wrapper

    return _outer_wrapper


def post_hook_wrapper(post_hook):
    def _outer_wrapper(fun):
        @wraps(fun)
        def _wrapper(*args, **kwargs):
            result = fun(*args, **kwargs)
            post_hook()
            return result

        return _wrapper

    return _outer_wrapper


class BaseWrapper(object):
    """Wrapper for BaseDynamicalObjects."""

    def __init__(self, env):
        self.__class__ = type(
            env.__class__.__name__, (self.__class__, env.__class__), {}
        )

        self.__dict__ = env.__dict__

        self.env = env


# class HookWrapper(BaseWrapper):
#     def __getattr__(self, attr):
#         """Wrapped __getattr__."""
#         _attr = self.env.__getattribute__(attr)

#         if callable(_attr):

#             @wraps(callable)
#             def _wrapper(*args, **kwargs):
#                 self.pre_hook()
#                 result = _attr(*args, **kwargs)
#                 self.post_hook()

#                 # prevent unwrapping
#                 if result is self.env:
#                     return self

#                 return result

#             return _wrapper

#         else:
#             return _attr

#     # @abstractmethod
#     def pre_hook(self):
#         raise NotImplementedError

#     # @abstractmethod
#     def post_hook(self):
#         raise NotImplementedError

#     # def reset(self):
#     #     self.pre_hook()
#     #     result = self.env.reset()
#     #     self.post_hook()
#     #     return result
