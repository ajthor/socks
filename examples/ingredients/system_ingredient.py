from sacred import Ingredient

import gym
from gym.envs.registration import make

from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np

system_ingredient = Ingredient("system")


@system_ingredient.config
def _config():
    """System configuration variables."""

    system_id = "2DIntegratorEnv-v0"

    time_horizon = None
    sampling_time = None


@system_ingredient.capture
def make_system(
    system_id: str,
    time_horizon: float,
    sampling_time: float,
) -> DynamicalSystem:
    """Construct an instance of the system."""
    env = make(system_id)

    if time_horizon is not None:
        env.time_horizon = time_horizon

    if sampling_time is not None:
        env.sampling_time = sampling_time

    return env


@system_ingredient.capture
def set_system_seed(seed: int, env: DynamicalSystem):
    """Set the random seed."""
    env.seed(seed=seed)
    env.observation_space.seed(seed=seed)
    env.state_space.seed(seed=seed)
    env.action_space.seed(seed=seed)
