"""Control policies.

Note:
    Policies can be either time-invariant or time-varying, and can be either open- or
    closed-loop. Thus, the arguments to the :py:meth:`__call__` method should allow for
    ``time`` and ``state`` to be specified (if needed), and should be optional kwargs::

        >>> def __call__(self, time=None, state=None):
        ...     ...

"""

from abc import ABC, abstractmethod

from gym import Space

import numpy as np


class BasePolicy(ABC):
    """Base policy class.

    This class is **abstract**, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from :py:class:`BasePolicy`.

    The :py:meth:`__call__` method is the main point of entry for the policy classes.
    All subclasses must implement a :py:meth:`__call__` method. This makes the class
    callable, so that policies can be evaluated as::

        action = policy(state)

    """

    action_space = None
    state_space = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Evaluate the policy.

        Returns:
            An action in the action space.

        """
        raise NotImplementedError


class RandomizedPolicy(BasePolicy):
    """Randomized policy.

    A policy which returns a random control action.

    Args:
        action_space: The action space of the system.

    """

    def __init__(self, action_space: Space = None):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

    def __call__(self, *args, **kwargs):
        return self.action_space.sample()


class ConstantPolicy(BasePolicy):
    """Constant policy.

    A policy which returns a constant control action.

    Args:
        action_space: The action space of the system.
        constant: The constant value returned by the policy.

    """

    def __init__(self, action_space: Space = None, constant=0):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

        self._constant = constant

    def __call__(self, *args, **kwargs):
        return np.full(
            self.action_space.shape,
            self._constant,
            dtype=self.action_space.dtype,
        )


class ZeroPolicy(ConstantPolicy):
    """Zero policy.

    A policy which returns a constant (zero) control action.

    Args:
        action_space: The action space of the system.

    """

    def __init__(self, action_space: Space = None):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

        self._constant = 0
