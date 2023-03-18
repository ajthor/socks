import numpy as np


class Space:
    def __init__(self, shape=None, dtype=None, seed=None) -> None:
        """Initialize the class."""
        self._shape = shape
        self._dtype = dtype
        self._seed = seed

        # assert dtype is not None
        if self.dtype is None:
            raise ValueError("dtype must be specified.")

        # assert dtype is valid numpy dtype
        if not isinstance(self.dtype, np.dtype):
            try:
                self._dtype = np.dtype(self.dtype)
            except TypeError:
                raise ValueError("dtype must be a valid numpy dtype.")

        # assert shape is not None
        if self.shape is None:
            raise ValueError("shape must be specified.")

        self._np_random = None

        if seed is not None:
            if isinstance(seed, int):
                self._np_random = np.random.RandomState(seed)
            else:
                self.seed(seed)

    @property
    def np_random(self):
        """Returns the random number generator and initializes it if None."""
        if self._np_random is None:
            self._np_random = np.random.RandomState(self._seed)

        return self._np_random

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def seed(self, seed=None):
        """Sets the seed of the random number generator."""

        # assert seed is an integer or None
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer or None.")

        self._seed = seed

        # initialize the random number generator
        self._np_random = np.random.RandomState(self._seed)

        return self._seed

    def contains(self, x) -> bool:
        """Check if x is in the space."""
        raise NotImplementedError

    def sample(self):
        """Sample a random element of the space."""
        raise NotImplementedError


class Box(Space):
    def __init__(
        self,
        low=None,
        high=None,
        shape=None,
        dtype=float,
        seed=None,
    ) -> None:
        """Initialize the class."""

        self._shape = shape
        self._dtype = dtype
        self._seed = seed

        # assert dtype is not none
        if dtype is None:
            raise ValueError("dtype must be specified.")

        # assert dtype is valid numpy dtype
        if not isinstance(self.dtype, np.dtype):
            try:
                self._dtype = np.dtype(self.dtype)
            except TypeError:
                raise ValueError("dtype must be a valid numpy dtype.")

        # if shape is not none, then it must be a tuple of integers
        if self.shape is not None:
            if not isinstance(self.shape, tuple):
                raise ValueError("shape must be a tuple of integers.")

            if not all(isinstance(x, int) for x in self.shape):
                raise ValueError("shape must be a tuple of integers.")
        # else, if low is a numpy array, then shape is the shape of low
        elif isinstance(low, np.ndarray):
            self._shape = low.shape
        # else, if low is a list, then shape is the shape of low
        elif isinstance(low, list):
            self._shape = np.asarray(low).shape
        # else, if low is a tuple, then shape is the shape of low
        elif isinstance(low, tuple):
            self._shape = np.asarray(low).shape
        # else, if low is a scalar, then shape is a tuple of length 1
        elif isinstance(low, (int, float)):
            self._shape = (1,)
        # else, if high is a numpy array, then shape is the shape of high
        elif isinstance(high, np.ndarray):
            self._shape = high.shape
        # else, if high is a list, then shape is the shape of high
        elif isinstance(high, list):
            self._shape = np.asarray(high).shape
        # else, if high is a tuple, then shape is the shape of high
        elif isinstance(high, tuple):
            self._shape = np.asarray(high).shape
        # else, if high is a scalar, then shape is a tuple of length 1
        elif isinstance(high, (int, float)):
            self._shape = (1,)
        # else, raise an error
        else:
            raise ValueError("low or high must be specified if shape is unspecified.")

        # if low is none, then low is -inf
        if low is None:
            low = -np.inf

        # if high is none, then high is inf
        if high is None:
            high = np.inf

        # if low is a scalar, then low is a numpy array of shape self.shape
        if isinstance(low, (int, float)):
            low = np.full(self.shape, low, dtype=self.dtype)

        # if high is a scalar, then high is a numpy array of shape self.shape
        if isinstance(high, (int, float)):
            high = np.full(self.shape, high, dtype=self.dtype)

        # ensure low is a numpy array and it has the same shape as self._shape
        if not isinstance(low, np.ndarray):
            low = np.asarray(low, dtype=self.dtype)

        if low.shape != self._shape:
            raise ValueError("low must have the same shape as shape.")

        # if low has a different dtype than self.dtype, then low is cast to self.dtype
        if low.dtype != self.dtype:
            low = low.astype(self.dtype)

        # ensure high is a numpy array and it has the same shape as self._shape
        if not isinstance(high, np.ndarray):
            high = np.asarray(high, dtype=self.dtype)

        if high.shape != self._shape:
            raise ValueError("high must have the same shape as shape.")

        # if high has a different dtype than self.dtype, then high is cast to self.dtype
        if high.dtype != self.dtype:
            high = high.astype(self.dtype)

        # assert low is less than or equal to high
        if not np.all(low <= high):
            raise ValueError("low must be less than or equal to high.")

        self._low = low
        self._high = high

        super().__init__(shape=self.shape, dtype=self.dtype, seed=self._seed)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def contains(self, x) -> bool:
        """Check if x is in the box."""

        # if x is a scalar, then x is a numpy array of shape self.shape
        if isinstance(x, (int, float)):
            x = np.full(self.shape, x, dtype=self.dtype)
        # else, if x is not a numpy array, then try converting it to a numpy array
        elif not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=self.dtype)

        return bool(
            np.can_cast(x, self.dtype, casting="safe")
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def is_bounded_above(self) -> bool:
        """Check if the box is bounded above."""
        return bool(np.all(self.high < np.inf))

    def is_bounded_below(self) -> bool:
        """Check if the box is bounded below."""
        return bool(np.all(self.low > -np.inf))

    def is_bounded(self) -> bool:
        """Check if the box is bounded."""
        return bool(self.is_bounded_above() and self.is_bounded_below())

    def sample(self):
        """Generate a random sample from the box."""

        has_lower_bound = self.low > -np.inf
        has_upper_bound = self.high < np.inf

        # bounded below is true if low has lower bound and not bounded above
        bounded_below = has_lower_bound & ~has_upper_bound
        # bounded above is true if high has upper bound and not bounded below
        bounded_above = has_upper_bound & ~has_lower_bound

        bounded = bounded_below & bounded_above
        unbounded = ~bounded

        sample = np.zeros(self.shape, dtype=self.dtype)

        # bounded samples are generated from a uniform distribution
        sample[bounded] = self._np_random.uniform(
            size=bounded[bounded].shape, low=bounded[bounded], high=self.high[bounded]
        )

        # unbounded samples are generated from a normal distribution
        sample[unbounded] = self._np_random.normal(size=unbounded[unbounded].shape)

        # unbounded below samples are generated from an exponential distribution
        sample[bounded_below] = (
            self._np_random.exponential(size=bounded_below[bounded_below].shape)
            + self.low[bounded_below]
        )

        # unbounded above samples are generated from an exponential distribution
        sample[bounded_above] = (
            self._np_random.exponential(size=bounded_above[bounded_above].shape)
            + self.high[bounded_above]
        )

        # if dtype is int, then sample is rounded to the nearest integer
        if self.dtype.kind == "i":
            sample = np.round(sample).astype(self.dtype)

        return sample.astype(self.dtype)

    def __eq__(self, other: object) -> bool:
        """Check if two boxes are equal."""
        if not isinstance(other, Box):
            return False
        return bool(
            self.shape == other.shape
            and np.all(self.low == other.low)
            and np.all(self.high == other.high)
        )

    def __repr__(self) -> str:
        """Return a string representation of the box."""
        return f"Box(low={self.low}, high={self.high})"
