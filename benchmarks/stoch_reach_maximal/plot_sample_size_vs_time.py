import numpy as np
from scipy import stats

from functools import partial

from benchmarks.experiment_matrix import experiment_matrix
from benchmarks.experiment_matrix import sample_mean

import matplotlib
import matplotlib.pyplot as plt

from benchmarks.stoch_reach_maximal.baseline import ex

mx = experiment_matrix(ex, callback=partial(sample_mean, alpha=0.95))


@mx.config
def mx_config():
    sample_size = range(100, 1600, 100)


@mx.post_run_hook
def _save(_run, _log):
    """Save the results to disk, for use externally."""

    if _run.main_function.__name__ == "main" and _run.result is not None:
        _log.info("Saving the results to disk.")
        result = np.empty((len(_run.result),), dtype=object)
        result[:] = _run.result
        np.savez("results/data.npz", *result)


@mx.post_run_hook
def _plot(sample_size, _run, _log):
    """Plot the results."""

    if _run.main_function.__name__ == "main" and _run.result is not None:
        _log.info("Plotting the results.")
        data = np.load("results/data.npz")

        fig = plt.figure()
        ax = plt.axes()
        plt.grid(zorder=0)

        x = np.array(list(sample_size))
        plt.xlabel("Sample Size")
        plt.ylabel("Computation Time [s]")

        mu = []
        low = []
        high = []

        for k, v in data.items():
            mu.append(np.mean(v))
            sigma = np.std(v, ddof=1)
            ci = stats.norm.interval(alpha=0.95, loc=mu[-1], scale=sigma)
            _low, _high = ci
            low.append(_low)
            high.append(_high)

        plt.errorbar(x, mu, yerr=[low, high], fmt="-", capsize=5, zorder=3)

        for i, item in enumerate(data.items()):
            _, v = item
            plt.scatter([x[i]] * len(v), v, marker=".", c="black", zorder=4)

        plt.savefig("results/stoch_reach_maximal_sample_size_vs_time.png")


if __name__ == "__main__":
    r = mx.run_commandline()
