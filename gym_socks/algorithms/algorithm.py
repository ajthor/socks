from abc import ABC, abstractmethod

import numpy as np


class AlgorithmInterface(ABC):
    """
    Base class for algorithms.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from AlgorithmInterface.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the algorithm.
        """

        # Algorithm parameters go here.

    @classmethod
    @abstractmethod
    def run(cls, *args, **kwargs):
        ...
