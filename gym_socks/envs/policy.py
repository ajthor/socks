from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Base policy class."""

    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


class RandomizedPolicy(BasePolicy):
    """Randomized policy."""

    def __init__(self, system, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.system = system

    def __call__(self, *args, **kwargs):
        return self.system.action_space.sample()


class ZeroPolicy(BasePolicy):
    """Zero policy."""

    def __init__(self, system, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.system = system

    def __call__(self, *args, **kwargs):
        return [0] * self.system.action_space.shape[0]
