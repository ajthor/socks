from abc import ABC, abstractmethod

import numpy as np


class AlgorithmInterface(ABC):
    """
    Base class for algorithms.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from AlgorithmInterface.

    """

    @abstractmethod
    def fit(cls, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(cls, *args, **kwargs):
        raise NotImplementedError
