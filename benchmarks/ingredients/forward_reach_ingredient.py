from sacred import Ingredient

import gym
from sacred.utils import convert_camel_case_to_snake_case

from gym_socks.envs.dynamical_system import DynamicalSystem

from gym_socks.sampling import sample
from gym_socks.sampling import sample_generator
from gym_socks.sampling import step_sampler
from gym_socks.sampling import uniform_grid
from gym_socks.sampling import uniform_grid_step_sampler

import numpy as np

from examples.ingredients.common import parse_array, box_factory
from examples.ingredients.common import grid_ranges
from examples.ingredients.common import grid_sample_size

forward_reach_ingredient = Ingredient("forward_reach")


@forward_reach_ingredient.config
def _config():
    """Forward reachability configuration variables."""

    test_points = {
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 50,
    }


@forward_reach_ingredient.capture
def compute_test_point_ranges(shape, test_points, _log):

    lower_bound = test_points["lower_bound"]
    upper_bound = test_points["upper_bound"]

    grid_resolution = test_points["grid_resolution"]

    test_point_space = box_factory(
        lower_bound,
        upper_bound,
        shape=shape,
        dtype=np.float32,
    )

    xi = grid_ranges(test_point_space, grid_resolution)

    return xi


@forward_reach_ingredient.capture
def generate_test_points(shape, _log):
    """Generate test points."""

    _log.info("Generating test points.")

    xi = compute_test_point_ranges(shape)
    T = uniform_grid(xi)

    return T
