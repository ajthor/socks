import unittest
from unittest import mock
from unittest.mock import patch

import gym

import numpy as np

from examples.ingredients.tracking_ingredient import tracking_ingredient


class TestTrackingIngredient(unittest.TestCase):
    def test_tracking_config(cls):
        """Test the ingredient config."""
        ingredient = tracking_ingredient
        cls.assertEqual(len(ingredient.configurations), 1)
