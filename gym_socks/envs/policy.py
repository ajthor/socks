from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Base policy class."""

    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RandomizedPolicy(BasePolicy):
    """Randomized policy."""

    def __init__(self, system, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system

    def __call__(self, *args, **kwargs):
        return self.system.action_space.sample()


class ConstantPolicy(BasePolicy):
    """Constant policy."""

    def __init__(self, system, constant=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
        self.constant = constant

    def __call__(self, *args, **kwargs):
        return [self.constant] * self.system.action_space.shape[0]


class ZeroPolicy(ConstantPolicy):
    """Zero policy."""

    def __init__(self, system, *args, **kwargs):
        super().__init__(system, *args, **kwargs)
