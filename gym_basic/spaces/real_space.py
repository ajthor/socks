import numpy as np

from gym.spaces.space import Space

class RealSpace(Space):
    """Real space."""

    def __init__(self, dim, **kwargs):
        self.dim = dim
        # super().__init__(self.dim, np.float32)

    def sample(self):
        return np.random.rand(self.dim)

    def contains(self, x):
        return x in self.values
