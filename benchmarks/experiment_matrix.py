import logging
import numpy as np

from collections.abc import Generator

from itertools import product
from functools import reduce
from operator import mul
from operator import itemgetter

from scipy import stats

from time import perf_counter

from sacred import Experiment


def _result_accumulator(
    config,
    min_iter=None,
    max_iter=None,
    warmup=False,
    callback=None,
):
    """Result accumulator.

    Maintains a list of results from the current experiment.

    Args:
        config: The configuration dictionary.
        min_iter: The minimum number of iterations per config.
        max_iter: The maximum number of iterations per config.
        warmup: Whether to run the experiment once as a warmup.
        callback: Boolean function to determine whether the experiment should continue.

    Yields:
        The current configuration for the experiment.

    Returns:
        The accumulated results for the experiment.

    """

    results = []
    if warmup is True:
        yield config, {"warmup": True}

    while True:
        result = yield config, {"warmup": False}

        if result is None:
            return results
        results.append(result)

        # Stopping logic.
        if len(results) >= min_iter:
            if callback is not None:
                callback_result = callback(results)
                assert isinstance(callback_result, bool), "Callback must return a bool."
            else:
                callback_result = False

            if callback_result is False or len(results) >= max_iter:
                return results


def _config_generator(
    local_config,
    min_iter=None,
    max_iter=None,
    warmup=False,
    callback=None,
):
    """Configuration generator.

    Turns the iterable config into a generator of config dictionaries.

    Args:
        local_config: The configuration dictionary. All values must be iterable.
        min_iter: The minimum number of iterations per config.
        max_iter: The maximum number of iterations per config.
        warmup: Whether to run the experiment once as a warmup.
        callback: Boolean function to determine whether the experiment should continue.

    Raises:
        StopIteration

    Yields:
        A configuration dictionary.

    Returns:
        Accumulated results for all experiments. This is returned via the ``value`` of
        the StopIteration exception.

    """

    matrix = zip(*local_config.items())
    keys, values = matrix
    configs = product(*values)

    results = []

    for i, config in enumerate(configs):
        a = _result_accumulator(
            dict(zip(keys, config)),
            min_iter=min_iter,
            max_iter=max_iter,
            warmup=warmup,
            callback=callback,
        )
        result = yield from a

        results.append(result)

    return results


def experiment_matrix(
    ex,
    min_iter: int = 3,
    max_iter: int = 256,
    warmup: bool = False,
    callback=None,
    same_seed: bool = False,
):
    """Experiment matrix factory.

    Creates an experiment matrix, which is a typical Sacred :py:class:`Experiment`.

    Args:
        ex: The Sacred :py:class:`Experiment` to iterate over.
        min_iter: The minimum number of iterations per config.
        max_iter: The maximum number of iterations per config.
        warmup: Whether to run the experiment once as a warmup.
        callback: Boolean function to determine whether the experiment should continue.
        same_seed: Whether all iterations should use the same seed.

    Returns:
        A Sacred :py:class:`Experiment` object.

    Example:
    ::

        ex = Experiment(ingredients=[data_ingredient])

        @ex.config
        def ex_config():
            value1 = 1
            value2 = 2

        @ex.main
        def main(value1, value2, _run):
            print(f"value1: {value1}, value2: {value2}")
            print(_run.info)
            return 1

        mx = experiment_matrix(ex, warmup=True)

        @mx.config
        def mx_config():
            value1 = [1, 2]
            value2 = [3, 4]

        if __name__ == "__main__":
            r = mx.run_commandline()

    """

    mx = Experiment("matrix")

    assert (
        isinstance(min_iter, int) and min_iter >= 0
    ), "min_iter must be non-negative integer."

    assert (
        isinstance(max_iter, int) and max_iter > 0 and max_iter > min_iter
    ), "max_iter must be strictly positive and greater than min_iter."

    @mx.main
    def main_wrapper(_config, _log, _run):
        local_config = _config.copy()

        if not same_seed:
            local_config.pop("seed", None)
        else:
            local_config["seed"] = [local_config["seed"]]

        # Ensure config contains only iterable items.
        for k, v in local_config.items():
            if isinstance(v, str):
                local_config[k] = [v]
            else:
                try:
                    _ = iter(v)
                except TypeError:
                    _log.warning("Config variable not iterable. Converting to list.")
                    local_config[k] = [v]

        gen = _config_generator(
            local_config,
            min_iter=min_iter,
            max_iter=max_iter,
            warmup=warmup,
            callback=callback,
        )

        result = None
        try:
            while True:
                config, meta_info = gen.send(result)
                run = ex.run(config_updates=config, meta_info=meta_info)
                result = run.result

        except StopIteration as e:
            gen_result = e.value

        if any(gen_result):
            return gen_result

    return mx


# def _result_accumulator(config):
#     """Result accumulator.

#     Args:
#         config: A dictionary containing the current config values.

#     Yields:
#         The current config values.

#     Returns:
#         The list of accumulated results.

#     """

#     results = []
#     while True:
#         result = yield config
#         if result is None:
#             return results
#         results.append(result)

#         # Stopping logic.
#         if len(results) >= 3:
#             return results


# def experiment_matrix(config, results):
#     """Lightweight experiment matrix function.

#     Args:
#         config: A dictionary of configuration values, where each value is an iterable.
#         results: A list to hold results.

#     Yields:
#         The parameters for the current experiment.

#     Example:

#         >>> config = {"value1": [1, 2], "value2": [3, 4]}
#         >>> results = []
#         >>> M = experiment_matrix(config, results)
#         >>> for experiment_config in M:
#         ...     # Run experiment.
#         ...     result = 1
#         ...     M.send(result)
#         >>> print(results)

#     """

#     experiment_matrix = zip(*config.items())
#     keys, values = experiment_matrix

#     for v in values:
#         try:
#             _ = iter(v)
#         except TypeError:
#             raise TypeError("Config values must be iterables.")

#     for experiment_config in product(*values):
#         a = _result_accumulator(dict(zip(keys, experiment_config)))
#         result = yield from a

#         results.append(result)


class ExperimentMatrix(Generator):
    """Experiment matrix.

    Generator class that returns the config for each eperiment.

    Can be iterated over using a for loop, meaning the results will be handled manually,
    or via a while loop within a try block catching a StopIteration exception, meaning
    the results will be handled by the class.

    Args:
        config: A dictionary of configuration values, where each value is an iterable.
        min_iter: The minimum number of iterations to generate for each config.
            Default 3.
        warmup: Whether to run a dummy execution for each config.
            Default True.

    Example:

        >>> config = {"value1": [1, 2], "value2": [3, 4]}
        >>> mx = ExperimentMatrix(config, min_iter=3, warmup=True)
        >>> @mx.main
        ... def main(value1, value2):
        ...     print({"value1": value1, "value2", value2})
        ...     # Run experiment here.
        ...     result = 1
        ...     return result
        >>> if __name__ == "__main__":
        ...     main()
        ...     print(mx.results)
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 4}
        {'value1': 1, 'value2': 4}
        {'value1': 1, 'value2': 4}
        {'value1': 1, 'value2': 4}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 4}
        {'value1': 2, 'value2': 4}
        {'value1': 2, 'value2': 4}
        {'value1': 2, 'value2': 4}
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

    Alternatively, the config values can be collected by the kwargs. The following is
    the same as the above.

    Example:

        >>> @mx.main
        ... def main(**kwargs):
        ...     print(kwargs)

    If you wish to loop through the matrix manually, the recommended method is to wrap a
    while loop in a try block, accounting for the StopIteration exception.

    Example:

        >>> config = {"value1": [1, 2], "value2": [3, 4]}
        >>> mx = ExperimentMatrix(config, min_iter=3, warmup=True)
        >>> result = None
        >>> try:
        ...     while True:
        ...         experiment_config = mx.send(result)
        ...         print(experiment_config)
        ...         # Run experiment using experiment_config.
        ...         result = 1
        ... except StopIteration:
        ...     pass
        >>> print(M.results)
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 4}
        {'value1': 1, 'value2': 4}
        {'value1': 1, 'value2': 4}
        {'value1': 1, 'value2': 4}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 4}
        {'value1': 2, 'value2': 4}
        {'value1': 2, 'value2': 4}
        {'value1': 2, 'value2': 4}
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

    The default behavior of the ExperimentMatrix is to use a while loop wrapped in a try
    block to detect the StopIteration exception. The results of each experiment are
    stored by the ExperimentMatrix, and are accumulated by sending the result back to
    the generator. The generator then uses an optional callback to determine whether to
    end the current iteration. The callback should return False if it should continue
    running, or True if it should move on to the next experiment configuration.

    Alternatively, using a for loop, the ExperimentMatrix simply iterates through all
    experiment configurations. It is then up to the user to send the results back to the
    ExperimentMatrix for accumulation, either within a nested for loop or some other
    mechanism. As a last resort, the results can also be stored outside the
    ExperimentMatrix class, in case there is a need to track results separately from the
    matrix, and the user only wishes to iterate through the experiment configurations.

    Example:

        >>> config = {"value1": [1, 2], "value2": [3, 4]}
        >>> results = []
        >>> mx = ExperimentMatrix(config)
        >>> for experiment_config in mx:
        ...     print(experiment_config)
        ...     # Run experiment using experiment_config.
        ...     result = 1
        ...     results.append(result)
        >>> print(results)
        {'value1': 1, 'value2': 3}
        {'value1': 1, 'value2': 4}
        {'value1': 2, 'value2': 3}
        {'value1': 2, 'value2': 4}
        [1, 1, 1, 1]

    """

    def __init__(
        self,
        config,
        min_iter: int = 3,
        max_iter: int = 256,
        callback=None,
        warmup: bool = True,
    ):
        matrix = zip(*config.items())
        self._keys, values = matrix

        for v in values:
            try:
                _ = iter(v)
            except TypeError:
                raise TypeError("Config values must be iterables.")

        self._configs = product(*values)
        self._values = values

        # Use None value to indicate experiments are not running.
        self._index = None
        # Initialize the result list.
        self.results = [[] for _ in range(len(self))]
        self.computation_time = [[] for _ in range(len(self))]

        self._min_iter = min_iter
        self._max_iter = max_iter

        self._warmup = warmup
        self._warmup_flag = False

        self._callback = callback

        self._pre_run = []
        self._post_run = []
        self._final = []

    @property
    def min_iter(self):
        """Minimum number of iterations."""
        return self._min_iter

    @min_iter.setter
    def min_iter(self, value):
        assert value > 0, "Min iterations must be strictly positive."
        self._min_iter = value

    @property
    def max_iter(self):
        """Maximum number of iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        assert value > 0, "Max iterations must be strictly positive."
        self._max_iter = value

    @property
    def warmup(self):
        """Boolean flag indicating whether to use a warmup run."""
        return self._warmup

    @warmup.setter
    def warmup(self, value):
        # assert boolean
        self._warmup = value

    @property
    def configs(self):
        """Generator expression of config values."""
        return (dict(zip(self._keys, values)) for values in product(*self._values))

    def __next__(self):
        try:
            config = next(self._configs)

        except StopIteration:
            self.throw(StopIteration)

        else:
            if self._index is None:
                self._index = 0
            else:
                self._index += 1

            self._current = dict(zip(self._keys, config))

        if self._warmup is True:
            self._warmup_flag = True

        return self._current

    def send(self, value):
        # Advance the iterator manually.
        if value is None:
            return self.__next__()

        # If the warmup flag is set, run the experiment once without logging the
        # results. This is to discard executions which may not be cached by the CPU.
        if self._warmup_flag is True:
            self._warmup_flag = False
            return self._current

        # Handle the result.
        self.results[self._index].append(value)

        if len(self.results[self._index]) >= self._min_iter:
            if self._callback is not None:
                callback_result = self._callback(self.results[self._index])
            else:
                callback_result = None

            if (
                callback_result is None
                or callback_result is False
                or len(self.results[self._index]) >= self._max_iter
            ):
                return self.__next__()

        return self._current

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __len__(self):
        return reduce(mul, map(len, self._values), 1)

    def main(self, fun):
        """Main function.

        The main function is a decorator that specifies the main function to be called
        for each configuration in the experiment matrix. Each value of the experiment
        dictionary is passed as an argument to the main function, which can be collected
        as kwargs, or explicitly specified as an argument.

        Args:
            fun: The main function of the experiment.

        Returns:
            The decorated function.

        Example:

            >>> config = {"value1": [1, 2], "value2": [3, 4]}
            >>> mx = ExperimentMatrix(config)
            >>> @mx.main
            ... def main(value1, value2):
            ...     # Run experiment here.
            >>> if __name__ == "__main__":
            ...     main()

        """

        logger = logging.getLogger(fun.__name__)

        def _wrapper(*args, **kwargs):
            result = None
            try:
                while True:
                    config = self.send(result)
                    new_kwargs = {**config, **kwargs}

                    if self._pre_run:
                        for pre_run_fn in self._pre_run:
                            pre_run_fn()

                    start_time = perf_counter()
                    result = fun(*args, **new_kwargs)
                    elapsed_time = perf_counter() - start_time
                    if self._warmup_flag is False:
                        self.computation_time[self._index].append(elapsed_time)

                    if self._post_run:
                        for post_run_fn in self._post_run:
                            post_run_fn()

            except StopIteration:
                pass

            finally:
                if self._final:
                    for final_fn in self._final:
                        final_fn()

        return _wrapper

    def pre_run(self, fun):
        def _wrapper(*args, **kwargs):
            try:
                fun(*args, **kwargs)
            except Exception as e:
                raise e

        self._pre_run.append(_wrapper)
        return _wrapper

    def post_run(self, fun):
        def _wrapper(*args, **kwargs):
            try:
                fun(*args, **kwargs)
            except Exception as e:
                raise e

        self._post_run.append(_wrapper)
        return _wrapper

    def final(self, fun):
        def _wrapper(*args, **kwargs):
            try:
                fun(*args, **kwargs)
            except Exception as e:
                raise e

        self._final.append(_wrapper)
        return _wrapper


def sample_mean(results: list, alpha: float = 0.95) -> bool:
    """Callback to determine when the results are within a confidence interval.

    The sample mean function computes the relative error of the results and returns True
    if the last entry of the results list is within the interval. Returns False
    otherwise.

    Args:
        results: list of values returned from a sequence of experiments.
        alpha: The probability between 0 and 1 that a value is within the interval.

    Returns:
        True if the result is not within the interval. False otherwise.

    Example:

        >>> config = {"value1": [1, 2], "value2": [3, 4]}
        >>> mx = ExperimentMatrix(config, callback=sample_mean)

    """

    results = np.asarray(results)
    last = results[-1]

    mu, sigma = results.mean(), results.std(ddof=1)
    confidence_interval = stats.norm.interval(alpha=alpha, loc=mu, scale=sigma)

    if last >= confidence_interval[0] and last < confidence_interval[1]:
        return True

    return False
