import unittest
from unittest import mock
from unittest.mock import patch

import logging

import gym

import gym_socks

import numpy as np

from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import make_system
from examples.ingredients.system_ingredient import set_system_seed


class TestSystemIngredient(unittest.TestCase):
    def test_system_config(cls):
        """Test the ingredient config."""
        ingredient = system_ingredient
        cls.assertEqual(len(ingredient.configurations), 1)


class TestMakeSystem(unittest.TestCase):
    def test_make_system(cls):
        """Test that make_system makes a system based on system_id string."""

        dummy_logger = logging.getLogger(__name__)
        env = make_system(
            system_id="2DIntegratorEnv-v0",
            time_horizon=None,
            sampling_time=None,
            _config=dict(),
            _log=dummy_logger,
        )

        cls.assertIsInstance(env, gym_socks.envs.integrator.NDIntegratorEnv)
