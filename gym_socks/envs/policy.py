from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Base policy class.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from BasePolicy.

    The `__call__` method is the main point of entry for the policy classes. All
    subclasses must implement a `__call__` method. This makes the class callable, so
    that policies can be evaluated as::

        u = policy(x)

    Note:
        Policies come in four main varieties:

        * Time-invariant open-loop policies.
        * Time-invariant closed-loop policies.
        * Time-varying open-loop policies.
        * Time-varying closed-loop policies.

        Thus, the arguments to the `__call__` method should allow for `time` and
        `state` to be specified (if needed), and should be optional kwargs, meaning
        they should have a `None` default value, like so:

            >>> def __call__(self, time=None, state=None):
            ...     ...

    """

    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Evaluate the policy.

        Returns:
            action: An action returned by the policy.

        """
        raise NotImplementedError


class RandomizedPolicy(BasePolicy):
    """Randomized policy.

    A policy which returns a random control action.

    Args:
        system: The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.

    """

    def __init__(self, system, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system

    def __call__(self, *args, **kwargs):
        return self.system.action_space.sample()


class ConstantPolicy(BasePolicy):
    """Constant policy.

    A policy which returns a constant control action.

    Args:
        system: The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.
        constant: The constant value returned by the policy.

    """

    def __init__(self, system, constant=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
        self.constant = constant

    def __call__(self, *args, **kwargs):
        return [self.constant] * self.system.action_space.shape[0]


class ZeroPolicy(ConstantPolicy):
    """Zero policy.

    A policy which returns a constant (zero) control action.

    Args:
        system : The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.

    """

    def __init__(self, system, *args, **kwargs):
        super().__init__(system, *args, **kwargs)
