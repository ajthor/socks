"""Stochastic reachability benchmark.

Compares stochastic reachability implementations.

"""
import numpy as np

from collections.abc import Generator

import logging
from gym_socks.utils.logging import ms_tqdm, _progress_fmt
from tqdm.contrib.logging import logging_redirect_tqdm

from itertools import product

from functools import reduce
from operator import mul
from operator import itemgetter

from sacred import Experiment, config, experiment

ex = Experiment()


@ex.config
def _config():
    pass


@ex.capture
def benchmark_kernel_sr():
    ex = Experiment()


@ex.capture
def benchmark_monte_carlo():
    pass


def _result_accumulator(config):
    results = []
    while True:
        result = yield config
        if result is None:
            return results
        results.append(result)


def experiment_matrix(config, results):
    experiment_matrix = zip(*config.items())
    keys, values = experiment_matrix

    for v in values:
        try:
            _ = iter(v)
        except TypeError:
            raise TypeError("Config values must be iterables.")

    for experiment_config in product(*values):
        a = _result_accumulator(dict(zip(keys, experiment_config)))
        result = yield from a

        results.append(result)


class ExperimentMatrix(Generator):
    def __init__(self, config):
        matrix = zip(*config.items())
        self._keys, values = matrix

        for v in values:
            try:
                _ = iter(v)
            except TypeError:
                raise TypeError("Config values must be iterables.")

        self._configs = product(*values)
        self._values = values

        self.results = [[] for _ in range(len(self))]

        self._index = -1

    def send(self, value):
        if value is not None:
            self.log_result(value)

        try:
            config = next(self._configs)

            self._index += 1

            return dict(zip(self._keys, config))

        except GeneratorExit:
            raise GeneratorExit

        except StopIteration:
            raise StopIteration

        else:
            raise RuntimeError("generator ignored GeneratorExit")

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __len__(self):
        return reduce(mul, map(len, self._values), 1)

    def log_result(self, result):
        self.results[self._index].append(result)


@ex.main
def main(seed, _log):

    sigmas = [1, 2, 3]
    sample_sizes = [1, 2, 3]

    config = {
        "sigma": [1, 2, 3],
        "sample_size": [[1, 1], [2, 2], [3, 3]],
        "delta": [1],
        "types": ["linear", "rbf"],
    }

    results = []
    M = experiment_matrix(config, results)

    for params in M:
        print(params)
        M.send(1)
        M.send(2)
        # M.send(None)

    print(results)

    # ---

    M = ExperimentMatrix(config)
    for params in M:
        print(params)
        M.log_result(1)
        M.log_result(2)

    print(M.results)

    # M = ExperimentMatrix(config)
    # num_iters = 2

    # # keys, values = zip(*config.items())
    # # print(values)
    # # # print(list(map(np.array, values)))
    # # print(product(*np.array(values)))

    # # experiment_matrix = zip(*params.items())
    # # keys, configs = experiment_matrix
    # # experiment_matrix = product(sigmas, sample_sizes)

    # pbar = ms_tqdm(total=len(M), bar_format=_progress_fmt)

    # with logging_redirect_tqdm():

    #     for params in M:
    #         # sigma = params["sigma"]
    #         # sample_size = params["sample_size"]
    #         # delta = params["delta"]
    #         # types = params["types"]
    #         g = itemgetter("sigma", "sample_size", "delta", "types")
    #         sigma, sample_size, delta, types = g(params)

    #         for i in range(num_iters):
    #             M.send(1)

    #         _log.info(
    #             f"Computing for sigma={sigma}, sample_size={sample_size}, delta={delta}, types={types}"
    #         )
    #         pbar.update()
    #         # cc_experiment_pd.run(
    #         #     config_updates={
    #         #         "seed": 0,
    #         #         "sigma": sigma,
    #         #         "sample": {"sample_space": {"sample_size": int(sample_size)}},
    #         #         "delta": 0.01,
    #         #         "plot_cfg": {"plot_filename": filename},
    #         #     }
    #         # )

    # pbar.close()

    # print(M._results)


if __name__ == "__main__":
    ex.run_commandline()
