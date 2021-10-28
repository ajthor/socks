"""Sample ingredient.

The sample ingredient is used when an experiment requires a sample from a system to be
generated. It provides configuration variables used to set up the sample scheme, the
shape of the sample space, as well as the sample size. It also provides functions used
to generate the samples used in the experiments.

The main configuration specifications for the sample are given by:

* The need to specify different types of sampling schemes.
* The need to specify the bounds of the sampling region.
* The need to specify the "density" of the samples or, alternatively, the sample size.

The sampling ingredient should then provide functions which:

* Generate the appropriate sampling spaces.
* Provide an interface to the sampling functions that provide a single point of entry.

Notes:
    The main problem is that sacred does not allow for "substituting" different
    configurations depending on other configuration choices. For instance, if a user
    specifies a certain sampling scheme, there is no explicit method to modify the other
    variables required, since the ingredients are not dynamic.

    Ideally, the user would be able to specify a certain sampling scheme, and an
    ingredient could be loaded and added to the experiment "dynamically", injecting new
    configuration variables that need to be specified for the particular sampling
    scheme and providing a consistent "interface" that can be used regardless of the
    chosen sampling scheme.

    However, sacred *does* allow for dynamic configuration variables, which means we
    can use conditional statements to add configuration variables at runtime.
    Additionally, sacred allows for multiple configuration functions, and the
    "precedence" or order of these functions means that the configurations can be
    constructed "in order". Lastly, dictionary configuration variables in sacred are
    updated to include new values rather than overwriting the entire dictionary if the
    same configuration variable is specified multiple times. This means we can simulate
    the dynamic ingredients using this procedure as a workaround.

"""

from sacred import Ingredient

import gym

import gym_socks
from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import BasePolicy, RandomizedPolicy, ZeroPolicy
from gym_socks.envs.sample import sample
from gym_socks.envs.sample import sample_generator
from gym_socks.envs.sample import step_sampler
from gym_socks.envs.sample import uniform_grid
from gym_socks.envs.sample import uniform_grid_step_sampler

import numpy as np

from examples.ingredients.common import grid_sample_size, parse_array
from examples.ingredients.common import box_factory
from examples.ingredients.common import grid_ranges

sample_ingredient = Ingredient("sample")


@sample_ingredient.config
def _config():
    """Sample configuration variables.

    The default configuration sets the sample types for both the `sample_space` and
    `action_space` samplers.

    Config:
        action_space: The configuration settings for the action space. Used for
            generating admissible control actions.
        sample_space: The configuration settings for the sample space. Used for
            generating samples from a system.
        sample_policy: The policy used when generating samples from a system.

    """

    sample_space = {"sample_scheme": "uniform"}
    sample_policy = "random"

    action_space = {"sample_scheme": "random"}


@sample_ingredient.config_hook
def _setup_random_sample_space_config_hook(config, command_name, logger):
    """Random sample configuration.

    If `random` is specified for the `sample_space` or `action_space` configuration
    variables, a `sample_size` is also required.

    """

    sample = config["sample"]
    update = dict()

    if sample["sample_space"]["sample_scheme"] == "random":
        _defaults = {
            "sample_size": 1000,
        }
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_space"] = {**_defaults, **sample["sample_space"]}

    if sample["action_space"]["sample_scheme"] == "random":
        _defaults = {
            "sample_size": 100,
        }
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["action_space"] = {**_defaults, **sample["action_space"]}

    return update


@sample_ingredient.config_hook
def _setup_uniform_sample_space_config_hook(config, command_name, logger):
    """Uniform sample configuration.

    If `uniform` is specified for the `sample_space` or `action_space` configuration
    variables, upper and lower bounds are required.

    Upper and lower bounds can be specified as either a scalar value, or as an array of
    values that has the same shape as the space.

    """

    sample = config["sample"]
    update = dict()

    if sample["sample_space"]["sample_scheme"] == "uniform":
        _defaults = {
            "lower_bound": -1,
            "upper_bound": 1,
            "sample_size": 1000,
        }
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_space"] = {**_defaults, **sample["sample_space"]}

    if sample["action_space"]["sample_scheme"] == "uniform":
        _defaults = {
            "lower_bound": -1,
            "upper_bound": 1,
            "sample_size": 1000,
        }
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["action_space"] = {**_defaults, **sample["action_space"]}

    return update


@sample_ingredient.config_hook
def _setup_grid_sample_space_config_hook(config, command_name, logger):
    """Grid sample configuration.

    If `grid` is specified for the `sample_space` or `action_space` configuration
    variables, upper and lower bounds are required, along with a grid "resolution".

    Upper and lower bounds can be specified as either a scalar value, or as an array of
    values that has the same shape as the space.

    The grid resolution can be specified as a scalar value or as an array of values that
    has the same shape as the space.

    """

    sample = config["sample"]
    update = dict()

    if sample["sample_space"]["sample_scheme"] == "grid":
        _defaults = {
            "lower_bound": -1,
            "upper_bound": 1,
            "grid_resolution": 50,
        }
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_space"] = {**_defaults, **sample["sample_space"]}

    if sample["action_space"]["sample_scheme"] == "grid":
        _defaults = {
            "lower_bound": -1,
            "upper_bound": 1,
            "grid_resolution": 5,
        }
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["action_space"] = {**_defaults, **sample["action_space"]}

    return update


def _sample_space_factory(shape, space_config):

    sample_scheme = space_config["sample_scheme"]

    if sample_scheme == "random":
        lower_bound = -np.inf
        upper_bound = np.inf
    else:
        lower_bound = space_config["lower_bound"]
        upper_bound = space_config["upper_bound"]

    _space = box_factory(lower_bound, upper_bound, shape, dtype=np.float32)
    return _space


@sample_ingredient.capture
def _policy_factory(env: DynamicalSystem, sample_policy: str) -> BasePolicy:
    if sample_policy not in {"random", "zero"}:
        raise ValueError(f"sample_policy config variable must be in {'random', 'zero'}")

    _sample_policy_map = {
        "random": RandomizedPolicy,
        "zero": ZeroPolicy,
    }

    return _sample_policy_map[sample_policy](env)


@sample_ingredient.capture
def _random_sample(
    seed: int,
    env: DynamicalSystem,
    sample_space: dict,
    sample_policy: str,
) -> list:

    sample_size = sample_space["sample_size"]

    _sample_space = _sample_space_factory(env, sample_space)
    _sample_space.seed(seed=seed)

    _sample_policy = _policy_factory(env, sample_policy)

    # Generate the sample.
    S = sample(
        sampler=step_sampler(
            system=env,
            policy=_sample_policy,
            sample_space=_sample_space,
        ),
        sample_size=sample_size,
    )

    return S


@sample_ingredient.capture
def _grid_sample(
    seed: int,
    env: DynamicalSystem,
    sample_space: dict,
    sample_policy: str,
) -> list:

    grid_resolution = sample_space["grid_resolution"]

    _sample_space = _sample_space_factory(env, sample_space)
    _sample_space.seed(seed=seed)

    _sample_policy = _policy_factory(env, sample_policy)

    xi = _compute_grid_ranges(_sample_space, grid_resolution)
    sample_size = _compute_grid_sample_size(_sample_space, grid_resolution)

    # Generate the sample.
    S = sample(
        sampler=uniform_grid_step_sampler(
            xi=xi,
            system=env,
            policy=_sample_policy,
            sample_space=_sample_space,
        ),
        sample_size=sample_size,
    )

    return S


@sample_ingredient.capture
def _uniform_action_sample(
    seed: int,
    env: DynamicalSystem,
    sample_space: dict,
    sample_policy: str,
) -> list:

    grid_resolution = sample_space["grid_resolution"]

    _sample_space = _sample_space_factory(env, sample_space)
    _sample_space.seed(seed=seed)

    @sample_generator
    def uniform_action_sampler() -> tuple:
        """Uniform action sampler.

        Generates a sample using multiple actions at a uniform grid of points taken from
        within the range specified by 'ranges'. Note that this is a simplification to
        make the result appear more uniform, but is not necessary for the correct
        operation of the algorithm. A random iid sample taken from the state space is
        sufficient.

        Yields:
            observation : Observation of input/output from the stochastic kernel.

        """

        xi = _compute_grid_ranges(_sample_space, grid_resolution)
        ui = _compute_grid_ranges(_action_space, action_grid_resolution)

        xc = uniform_grid(xi)

        for action_item in ui:

            for point in xc:
                state = point
                action = [action_item]

                env.state = state
                next_state, cost, done, _ = env.step(action)

                yield (state, action, next_state)

    # Generate the sample.
    S = sample(
        sampler=multi_action_sampler,
        sample_size=sample_size,
    )

    return S


def _uniform_action_decorator(fun):
    @sample_generator
    def _wrapper(ui):
        for action in ui:
            yield from fun()

    return _wrapper


@sample_ingredient.capture
def _get_sample_size(space: gym.spaces.Box, sample_space: dict):
    if sample_space["sample_scheme"] == "grid":
        return grid_sample_size(
            space=space, grid_resolution=sample_space["grid_resolution"]
        )

    return sample_space["sample_size"]


@sample_ingredient.capture
def generate_sample(
    _log, seed: int, env: DynamicalSystem, sample_space: dict, sample_policy: str
) -> list:

    sample_scheme = sample_space["sample_scheme"]
    sample_size = _get_sample_size(space=env.state_space, sample_space=sample_space)

    _sample_space = _sample_space_factory(env.state_space.shape, sample_space)
    _sample_space.seed(seed=seed)

    _sampler_map = {
        "random": step_sampler,
        "uniform": step_sampler,
        "grid": uniform_grid_step_sampler,
    }

    _sampler = _sampler_map[sample_scheme]

    if sample_policy == "grid":
        _sampler = _uniform_action_decorator(_sampler)

    _sample_policy = _policy_factory(env, sample_policy)

    # Generate the sample.
    _log.info("Generating the sample.")
    S = sample(
        sampler=_sampler(
            system=env,
            policy=_sample_policy,
            sample_space=_sample_space,
        ),
        sample_size=sample_size,
    )

    return S


@sample_ingredient.capture
def generate_admissible_actions(
    _log, seed: int, env: DynamicalSystem, action_space: dict
):

    _log.info("Generating admissible control actions.")
    sample_scheme = action_space["sample_scheme"]
    grid_resolution = action_space["grid_resolution"]

    _action_space = _sample_space_factory(env.action_space.shape, action_space)
    _action_space.seed(seed=seed)

    if sample_scheme == "random":
        A = []
        for i in range(action_space["sample_size"]):
            A.append(_action_space.sample())

    ui = grid_ranges(_action_space, grid_resolution)
    A = uniform_grid(ui)

    return A
