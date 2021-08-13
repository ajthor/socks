import numpy as np

from sklearn.metrics.pairwise import rbf_kernel

class Kernel():

    @abstractmethod
    def __call__(self, X, Y=None):
        """Evaluate the kernel."""

class RBF(Kernel):

    def __init__(self, bandwidth_parameter=0.1):
        self.bandwidth_parameter = bandwidth_parameter

    def __call__(self, X, Y=None):
        """Evaluate the kernel."""

        if Y is None:


        else:
