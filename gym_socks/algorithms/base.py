from abc import ABC, abstractmethod

import numpy as np


class ClassifierMixin(ABC):
    """Base class for algorithms.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from ClassifierMixin.

    The ClassifierMixin is meant to mimic the sklearn estimator base class in order
    to promote a standard interface among machine learning algorithms. It requires that
    all subclasses implement a ``fit``, ``predict``, and ``score`` method.

    """

    _estimator_type = "classifier"

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def score(self):
        raise NotImplementedError


class ClusterMixin(ABC):
    """Base class for algorithms.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from ClusterMixin.

    The ClusterMixin is meant to mimic the sklearn estimator base class in order
    to promote a standard interface among machine learning algorithms. It requires that
    all subclasses implement a ``fit`` and ``fit_predict`` method.

    """

    _estimator_type = "clusterer"

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self):
        raise NotImplementedError


class RegressorMixin(ABC):
    """Base class for algorithms.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from RegressorMixin.

    The RegressorMixin is meant to mimic the sklearn estimator base class in order
    to promote a standard interface among machine learning algorithms. It requires that
    all subclasses implement a ``fit``, ``predict``, and ``score`` method.

    """

    _estimator_type = "regressor"

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def score(self):
        raise NotImplementedError
