from sacred import Ingredient

import gym

import numpy as np

from gym.envs.registration import make

from gym_socks.envs.dynamical_system import DynamicalSystem

from examples.ingredients.common import assert_config_has_key, box_factory

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
    _config,
    _log,
) -> DynamicalSystem:
    """Construct an instance of the system.

    Args:
        system_id: System identifier string.
        time_horizon: The time horizon of the system.
        sampling_time: The sampling time of the system.

    Returns:
        An instance of the dynamical system model.

    """

    _log.debug(f"Making system: {system_id}")
    env = make(system_id)

    if time_horizon is not None:
        env.time_horizon = time_horizon

    if sampling_time is not None:
        env.sampling_time = sampling_time

    if "action_space" in _config:
        assert_config_has_key(_config["action_space"], "lower_bound")
        assert_config_has_key(_config["action_space"], "upper_bound")

        env.action_space = box_factory(
            lower_bound=_config["action_space"]["lower_bound"],
            upper_bound=_config["action_space"]["upper_bound"],
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    return env


@system_ingredient.capture
def set_system_seed(seed: int, env: DynamicalSystem, _log):
    """Set the random seed.

    Args:
        seed: The random seed for the experiment.
        env: The dynamical system model.

    """

    _log.debug(f"Setting seed to: {seed}")
    env.seed(seed=seed)
    env.observation_space.seed(seed=seed)
    env.state_space.seed(seed=seed)
    env.action_space.seed(seed=seed)


@system_ingredient.command(unobserved=True)
def print_info(system_id):
    """Print system info.

    Prints information of the system specified by `system_id` to the screen.

    Args:
        system_id: System identifier string.

    """

    env = make_system()

    from pprint import PrettyPrinter

    printer = PrettyPrinter(indent=4, width=88, compact=False, sort_dicts=False)

    print(env.__doc__)

    info = {
        "system_id": system_id,
        "state_space": {
            "shape": env.state_space.shape,
            "low": env.state_space.low,
            "high": env.state_space.high,
        },
        "action_space": {
            "shape": env.action_space.shape,
            "low": env.action_space.low,
            "high": env.action_space.high,
        },
        "observation_space": {
            "shape": env.observation_space.shape,
            "low": env.observation_space.low,
            "high": env.observation_space.high,
        },
    }

    printer.pprint(info)
