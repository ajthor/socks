import unittest
from unittest.mock import patch

import numpy as np
from functools import partial

from sacred import Experiment

from benchmarks.experiment_matrix import experiment_matrix
from benchmarks.experiment_matrix import sample_mean


class TestExperimentMatrix(unittest.TestCase):
    def test_config_len(cls):
        """Length of the matrix should match the product of config value lengths."""

        ex = Experiment()

        @ex.config
        def _ex_config():
            value1 = 1
            value2 = 1

        @ex.main
        def _main(value1, value2):
            return 1

        mx = experiment_matrix(ex)

        @mx.config
        def _mx_config():
            value1 = range(3)
            value2 = [1, 2, 3]

        cls.assertEqual(len(mx.run(options={"--loglevel": "40"}).result), 9)

    def test_total_len(cls):
        """Total iterations should be greater than number of iterations times length."""

        ex = Experiment()

        @ex.config
        def _ex_config():
            value1 = 1
            value2 = 1

        @ex.main
        def _main(value1, value2):
            return 1

        mx = experiment_matrix(ex)

        @mx.config
        def _mx_config():
            value1 = range(3)
            value2 = [1, 2, 3]

        results = mx.run(options={"--loglevel": "40"}).result

        for result in results:
            cls.assertGreaterEqual(len(result), 3)
