from abc import ABC, abstractmethod
from collections.abc import MutableSequence

from gym_socks.envs.core import BaseDynamicalObject

import numpy as np
from scipy.integrate import solve_ivp


class _WorldObjectMeta(type):
    """_WorldObject meta class.

    The meta class defines a virtual interface for world objects, which must implement
    the following methods: ``step``, ``reset``, ``render``, ``close``, and ``seed``.

    """

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (
            hasattr(subclass, "step")
            and callable(subclass.step)
            and hasattr(subclass, "reset")
            and callable(subclass.reset)
            and hasattr(subclass, "render")
            and callable(subclass.render)
            and hasattr(subclass, "close")
            and callable(subclass.close)
            and hasattr(subclass, "seed")
            and callable(subclass.seed)
        )


class _WorldObject(metaclass=_WorldObjectMeta):
    """Virtual interface for world objects.

    Example:

        >>> class DummyObject(object):
        ...     def step(self):
        ...         pass
        ...
        ...     def reset(self):
        ...         pass
        ...
        ...     def render(self):
        ...         pass
        ...
        ...     def close(self):
        ...         pass
        ...
        ...     def seed(self):
        ...         pass
        ...
        >>> dummy = DummyObject()
        >>> isinstance(DummyObject, _WorldObject)

    """

    pass


class World(MutableSequence):
    """World.

    The ``World`` class is essentially a ``MutableSequence`` (such as a list), of
    objects contained within the world. The objects contained within the world must be
    of type ``_WorldObject``, meaning the implement the ``step``, ``reset``, ``render``,
    ``close``, and ``seed`` methods. If an object that does not implement these methods
    is added, an assertion error will be thrown.

    It can be used in much the same was as a list, for example::

        >>> world = World()
        >>> world.append(item)
        >>> world[1] = item
        >>> world += [item]

    The ``World`` is primarily used for keeping track of multiple objects, such as
    obstacles, which are contained within the world environment and simulated together.
    While it can track ``DynamicalSystem``, it is moreso intended to track uncontrolled
    or fully autonomous systems or objects, and to synchronize the simulation time.

    In addition, it implements the following methods, which are applied to all objects
    in the world environment: ``step``, ``reset``, ``render``, ``close``, and ``seed``.

    E.g., the ``step`` method calls the ``step`` method for each item in the world. The
    order in which the items are iterated over is the same as the order of the list. If
    the functions return a value, the results are given in a list.

    """

    _time_horizon = 1

    def __init__(self) -> None:
        self._objects = []

    @property
    def time_horizon(self):
        """Time horizon for the simulation."""
        return self._time_horizon

    @time_horizon.setter
    def time_horizon(self, value: int):
        msg = "Horizon must be an integer."
        assert value >= 0 and isinstance(value, (int, np.integer)), msg

        self._time_horizon = int(value)

    @staticmethod
    def _check_item(item: object):
        """Checks whether an item is a ``_WorldObject``.

        Raises:
            ValueError: If item is not an instance of ``_WorldObject``.

        """

        if not isinstance(item, _WorldObject):
            raise ValueError("invalid world object")

    def __getitem__(self, _index: int):
        return self._objects[_index]

    def __setitem__(self, _index: int, _object: _WorldObject):
        self._check_item(_object)
        self._objects[_index] = _object

    def __delitem__(self, _index: int):
        del self._objects[_index]

    def __len__(self):
        return len(self._objects)

    def insert(self, _index: int, _object: _WorldObject):
        self._check_item(_object)
        self._objects.insert(_index, _object)

    def step(self, time: int = None) -> list:
        """Advances the simulation forward one time step.

        Args:
            time: The simulation time.

        Returns:
            The result of each step function in a list.

        """

        result = []
        for item in self._objects:
            result.append(item.step(time=time))

        return result

    def reset(self) -> list:
        """Reset the world to a random initial condition.

        Returns:
            The result of each reset function in a list.

        """

        result = []
        for item in self._objects:
            result.append(item.reset())

        return result

    def render(self, mode: str = "human"):
        for item in self._objects:
            item.render(mode)

    def close(self):
        for item in self._objects:
            item.close()

    def seed(self, seed: int = None) -> list:
        """Sets the seed of the random number generator.

        This is primarily useful for objects which incorporate some sort of
        stochasticity to ensure repeatability.

        Args:
            seed: Integer value representing the random seed.

        Returns:
            The seed of the RNG for each object in a list.

        """

        result = []
        for item in self._objects:
            result.append(item.seed(seed=seed))

        return result
