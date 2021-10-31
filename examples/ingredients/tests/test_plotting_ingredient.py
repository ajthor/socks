from typing import ForwardRef
import unittest
from unittest import mock
from unittest.mock import patch

import os

import gym

from gym_socks.envs import NDIntegratorEnv

import numpy as np

from examples.ingredients.plotting_ingredient import plotting_ingredient
from examples.ingredients.plotting_ingredient import update_rc_params


class TestForwardReachIngredient(unittest.TestCase):
    def test_plotting_config(cls):
        """Test the ingredient config."""
        ingredient = plotting_ingredient
        cls.assertEqual(len(ingredient.configurations), 1)

    def test_rc_params_update(cls):
        """Test that update_rc_params changes the global parameters for matplotlib."""

        rc_params_filename = os.path.abspath("examples/ingredients/matplotlibrc")
        rc_params = {"lines": {"linestyle": ":"}}

        import matplotlib

        cls.assertEqual(matplotlib.rcParams["lines.linestyle"], "-")
        cls.assertEqual(matplotlib.rcParams["lines.linewidth"], 1.5)

        update_rc_params(
            matplotlib,
            rc_params=rc_params,
            rc_params_filename=rc_params_filename,
        )

        cls.assertEqual(matplotlib.rcParams["lines.linestyle"], ":")
        cls.assertEqual(matplotlib.rcParams["lines.linewidth"], 0.75)
