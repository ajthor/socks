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
        StopIteration: Raised when the list of configs is exhausted. The value of the
            StopIteration exception contains the accumulated results.

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
    warmup: bool = True,
    callback=None,
    same_seed: bool = True,
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

            ex = Experiment()

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
    def main(_config, _log, _run):
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


def sample_mean(results: list, alpha: float = 0.95) -> bool:
    """Callback to determine when the results are within a confidence interval.

    The sample mean function computes the relative error of the results and returns True
    if the last entry of the results list is within the interval. Returns False
    otherwise.

    Args:
        results: list of values returned from a sequence of experiments.
        alpha: The probability between 0 and 1 that a value is within the interval.

    Returns:
        True if the last result is within the interval. False otherwise.

    Example:

        >>> config = {"value1": [1, 2], "value2": [3, 4]}
        >>> mx = ExperimentMatrix(config, callback=sample_mean)

    """

    results = np.asarray(results)
    last = results[-1]

    mu = results.mean()
    sigma = results.std(ddof=1)
    confidence_interval = stats.norm.interval(alpha=alpha, loc=mu, scale=sigma)

    if last >= confidence_interval[0] and last < confidence_interval[1]:
        return True

    return False
