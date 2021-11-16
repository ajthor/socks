from sacred import Ingredient

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem

from gym_socks.envs.sample import sample
from gym_socks.envs.sample import sample_generator
from gym_socks.envs.sample import step_sampler
from gym_socks.envs.sample import uniform_grid
from gym_socks.envs.sample import uniform_grid_step_sampler

import numpy as np

from examples.ingredients.common import parse_array, box_factory
from examples.ingredients.common import grid_ranges
from examples.ingredients.common import grid_sample_size

backward_reach_ingredient = Ingredient("backward_reach")


@backward_reach_ingredient.config
def _config():
    """Backward reachability configuration variables."""

    problem = "THT"

    constraint_tube_bounds = {"lower_bound": -1, "upper_bound": 1}
    target_tube_bounds = {"lower_bound": -0.5, "upper_bound": 0.5}

    test_points = {
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 25,
    }


def generate_tube(time_horizon: int, shape: tuple, bounds: dict):
    """Generate a stochastic reachability tube using config.

    This function computes a stochastic reachability tube using the tube configuration.

    Args:
        env: The dynamical system model.
        bounds: The bounds of the tube. Specified as a dictionary.

    Returns:
        A list of spaces indexed by time.

    """

    tube_lb = bounds["lower_bound"]
    tube_ub = bounds["upper_bound"]

    tube_lb = parse_array(tube_lb, shape=(time_horizon,), dtype=np.float32)
    tube_ub = parse_array(tube_ub, shape=(time_horizon,), dtype=np.float32)

    tube = []
    for i in range(time_horizon):
        tube_t = box_factory(
            tube_lb[i],
            tube_ub[i],
            shape=shape,
            dtype=np.float32,
        )
        tube.append(tube_t)

    return tube


@backward_reach_ingredient.capture
def compute_test_point_ranges(env, test_points):

    lower_bound = test_points["lower_bound"]
    upper_bound = test_points["upper_bound"]

    grid_resolution = test_points["grid_resolution"]

    test_point_space = box_factory(
        lower_bound,
        upper_bound,
        shape=env.state_space.shape,
        dtype=np.float32,
    )

    xi = grid_ranges(test_point_space, grid_resolution)

    return xi


@backward_reach_ingredient.capture
def generate_test_points(env, _log):
    """Generate test points to evaluate the safety probabilities."""
    _log.debug("Generating test points.")
    xi = compute_test_point_ranges(env)
    T = uniform_grid(xi)

    return T
