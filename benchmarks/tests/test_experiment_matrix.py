import unittest
from unittest.mock import patch

import numpy as np
from functools import partial

from benchmarks.experiment_matrix import ExperimentMatrix
from benchmarks.experiment_matrix import sample_mean


class TestExperimentMatrix(unittest.TestCase):
    def test_config_len(cls):
        """Length of the matrix should match the product of config value lengths."""

        value1 = range(3)
        value2 = [1, 2, 3]

        config = {
            "value1": value1,
            "value2": value2,
        }

        mx = ExperimentMatrix(config)

        cls.assertEqual(len(mx), len(value1) * len(value2))

    def test_total_len(cls):
        """Total iterations should be the number of iterations times length."""

        value1 = range(3)
        value2 = [1, 2, 3]

        config = {
            "value1": value1,
            "value2": value2,
        }

        min_iter = 3

        warmup = False
        mx1 = ExperimentMatrix(config, min_iter=min_iter, warmup=warmup)
        counter = 0

        result = None
        try:
            while True:
                _ = mx1.send(result)
                result = 1
                counter += 1

        except StopIteration:
            pass

        cls.assertEqual(counter, (len(value1) * len(value2)) * (min_iter + warmup))

        warmup = True
        mx2 = ExperimentMatrix(config, min_iter=min_iter, warmup=warmup)
        counter = 0

        result = None
        try:
            while True:
                _ = mx2.send(result)
                result = 1
                counter += 1

        except StopIteration:
            pass

        cls.assertEqual(counter, (len(value1) * len(value2)) * (min_iter + warmup))

    def test_fails_with_scalar(cls):
        """Test that scalar values raise an error."""

        config = {
            "value1": range(3),
            "value2": 1,
        }

        with cls.assertRaises(TypeError):
            mx = ExperimentMatrix(config)

    def test_comp_time(cls):
        """Test that the number of iterations is the same for sample_mean."""
        state = np.random.get_state()
        np.random.seed(0)

        try:

            config = {
                "value1": [1, 2, 3],
            }

            mx = ExperimentMatrix(config, callback=partial(sample_mean, alpha=0.99))

            result = None
            try:
                while True:
                    _ = mx.send(result)
                    result = np.random.standard_normal()

            except StopIteration:
                pass

            cls.assertEqual(len(mx.results[0]), 20)
            cls.assertEqual(len(mx.results[1]), 250)
            cls.assertEqual(len(mx.results[2]), 62)

            mx = ExperimentMatrix(config, callback=partial(sample_mean, alpha=0.95))

            result = None
            try:
                while True:
                    _ = mx.send(result)
                    result = np.random.standard_normal()

            except StopIteration:
                pass

            cls.assertEqual(len(mx.results[0]), 47)
            cls.assertEqual(len(mx.results[1]), 39)
            cls.assertEqual(len(mx.results[2]), 32)

            mx = ExperimentMatrix(config, callback=partial(sample_mean, alpha=0.68))

            result = None
            try:
                while True:
                    _ = mx.send(result)
                    result = np.random.standard_normal()

            except StopIteration:
                pass

            cls.assertEqual(len(mx.results[0]), 9)
            cls.assertEqual(len(mx.results[1]), 8)
            cls.assertEqual(len(mx.results[2]), 7)

        finally:
            np.random.set_state(state)

    # config = {
    #     "sigma": np.linspace(1, 2, 11),
    #     "sample_size": [[1, 1], [2, 2], [3, 3]],
    #     "delta": [1],
    #     "types": ["linear", "rbf"],
    # }

    # results = []
    # M = experiment_matrix(config, results)

    # for params in M:
    #     print(params)
    #     M.send(1)
    #     M.send(2)
    #     # M.send(None)

    # print(results)

    # # ---

    # results = []
    # M = experiment_matrix(config, results)
    # result = None
    # try:
    #     while True:
    #         # Get params.
    #         params = M.send(result)
    #         print(params)

    #         # Run experiment.
    #         result = 1

    # except StopIteration:
    #     pass

    # print(results)

    # # ---

    # M = ExperimentMatrix(config)
    # result = None

    # for params in M:
    #     print(params)
    #     M.send(1)
    #     M.send(2)

    # print(M.results)

    # # ---

    # M = ExperimentMatrix(config)
    # result = None

    # try:
    #     while True:
    #         # Get params.
    #         params = M.send(result)
    #         print(params)

    #         # Run experiment.
    #         result = 1

    # except StopIteration:
    #     pass

    # print(M.results)
