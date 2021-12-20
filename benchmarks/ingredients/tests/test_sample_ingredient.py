import unittest
from unittest import mock
from unittest.mock import patch

import gym

from gym_socks.envs import NDIntegratorEnv

import numpy as np

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample
from examples.ingredients.sample_ingredient import generate_admissible_actions


class TestSampleIngredient(unittest.TestCase):
    def test_sample_config(cls):
        """Test the ingredient config."""
        ingredient = sample_ingredient
        cls.assertEqual(len(ingredient.configurations), 1)


class TestGenerateSample(unittest.TestCase):
    def test_generate_sample(cls):
        """Test that generate_sample produces a sample from a system."""

        pass


class TestGenerateAdmissibleActions(unittest.TestCase):
    def test_generate_admissible_actions(cls):
        """Test that generate_admissible_actions produces a sample from a system."""

        pass
