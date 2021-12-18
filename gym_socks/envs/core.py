from abc import ABC, abstractmethod

import gym
from gym.utils import seeding

import numpy as np


class BaseDynamicalObject(gym.Env, ABC):
    """Base dynamical object class.

    Bases: :py:obj:`gym.Env`, :py:obj:`abc.ABC`

    This class is **abstract**, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from :py:class:`BaseDynamicalObject`.

    This class serves as the base interface for dynamical objects, represented in most
    cases by either a :py:class:`DynamicalSystem` or an obstacle.

    """

    @abstractmethod
    def step(self):
        """Advance the object forward in time.

        Advances the object in the simulation. By default, an object is uncontrolled,
        meaning it accepts no parameters and the system evolves forward in time
        according to its own, internal dynamics. Controlled systems should accept an
        ``action`` parameter, which represents the user-selected control input.

        Additionally, time-varying systems should also accept a ``time`` parameter.

        Returns:
            A tuple ``(obs, cost, done, info)``, where ``obs`` is the observation
            vector. Generally, it is the state of the system corrupted by some
            measurement noise. If the system is fully observable, this is the actual
            state of the system at the next time step. ``cost`` is the cost
            (reward) obtained by the system for taking action u in state x and
            transitioning to state y. In general, this is not typically used with
            :py:class:`DynamicalSystem` models. ``done`` is a flag to indicate the
            simulation has terminated. Usually toggled by guard conditions, which
            terminates the simulation if the system violates certain operating
            constraints. ``info`` is a dictionary containing extra information.

        See also:
            :py:meth:`gym_socks.envs.dynamical_system.DynamicalSystem.step`

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
        """Renders the environment.

        This method must be overridden in subclasses in order to enable rendering.
        Not all environments support rendering.

        Args:
            mode: the mode to render with

        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically :py:func:`close` themselves when garbage collected or when the program exits.

        """
        pass

    def seed(self, seed=None):
        """Sets the seed of the random number generator.

        Args:
            seed: Integer value representing the random seed.

        Returns:
            The seed of the RNG.

        """

        return
