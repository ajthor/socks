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

    However, sacred *does* allow for configuration "hooks" which means we can
    dynamically add configuration variables at runtime. Additionally, dictionary
    configuration variables in sacred are updated to include new values rather than
    overwriting the entire dictionary if the same configuration variable is specified
    multiple times. This means we can simulate the dynamic ingredients using this
    procedure as a workaround.

"""


from random import Random
import gym
import gym_socks

import numpy as np

from functools import wraps

from sacred import Ingredient

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import BasePolicy
from gym_socks.envs.policy import ConstantPolicy
from gym_socks.envs.policy import RandomizedPolicy
from gym_socks.envs.policy import ZeroPolicy

from gym_socks.envs.sample import sample, sequential_action_sampler
from gym_socks.envs.sample import sample_generator
from gym_socks.envs.sample import step_sampler
from gym_socks.envs.sample import uniform_grid
from gym_socks.envs.sample import uniform_grid_step_sampler

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
    sample_policy = {"sample_scheme": "random"}

    action_space = {"sample_scheme": "uniform"}


@sample_ingredient.config_hook
def _setup_random_sample_space_config_hook(config, command_name, logger):
    """Random sample configuration.

    If `random` is specified for the `sample_space` or `action_space` configuration
    variables, a `sample_size` is also required.

    """

    sample = config["sample"]
    update = dict()

    _defaults = {
        "sample_size": 1000,
    }

    if sample["sample_space"]["sample_scheme"] == "random":
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_space"] = {**_defaults, **sample["sample_space"]}

    if sample["action_space"]["sample_scheme"] == "random":
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

    _defaults = {
        "lower_bound": -1,
        "upper_bound": 1,
        "sample_size": 1000,
    }

    if sample["sample_space"]["sample_scheme"] == "uniform":
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_space"] = {**_defaults, **sample["sample_space"]}

    if sample["action_space"]["sample_scheme"] == "uniform":
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

    _defaults = {
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 10,
    }

    if sample["sample_space"]["sample_scheme"] == "grid":
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_space"] = {**_defaults, **sample["sample_space"]}

    if sample["sample_policy"]["sample_scheme"] == "grid":
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["sample_policy"] = {**_defaults, **sample["sample_policy"]}

    if sample["action_space"]["sample_scheme"] == "grid":
        # Merge dictionaries, being careful not to overwrite existing entries.
        update["action_space"] = {**_defaults, **sample["action_space"]}

    return update


def _sample_space_factory(shape: tuple, space_config: dict):
    """Sample space factory.

    Creates a sample space based on configuration variables.

    """

    sample_scheme = space_config["sample_scheme"]

    if sample_scheme == "random":
        lower_bound = -np.inf
        upper_bound = np.inf
    else:
        lower_bound = space_config["lower_bound"]
        upper_bound = space_config["upper_bound"]

    _space = box_factory(lower_bound, upper_bound, shape, dtype=np.float32)
    return _space


# @sample_ingredient.capture
# def _policy_factory(env: DynamicalSystem, sample_policy: dict) -> BasePolicy:
#     if sample_policy["sample_scheme"] not in {"random", "zero", "grid"}:
#         raise ValueError(
#             f"sample_policy config variable must be one of {'random', 'zero', 'grid'}."
#         )

#     _sample_policy_map = {
#         "random": RandomizedPolicy,
#         "zero": ZeroPolicy,
#     }

#     return _sample_policy_map[sample_policy["sample_scheme"]](env)


def _get_sample_size(space: gym.spaces.Box, space_config: dict):
    """Gets the sample size from config variables."""

    if space_config["sample_scheme"] == "grid":
        return grid_sample_size(
            space=space, grid_resolution=space_config["grid_resolution"]
        )

    return space_config["sample_size"]


@sample_ingredient.capture
def _default_sampler(seed: int, env: DynamicalSystem, sample_space):
    """Default sampler.

    By default, the sample generator yields a random sample from the stste space. Since
    the space is a `gym.spaces.Box`, it yields a sample using different underlying
    distributions depending on the bounds of the space. For bounded spaces, the
    distribution is uniform.

    Args:
        seed: The random seed.
        env: The dynamical system model.
        sample_space: The sample space configuration variable.

    Returns:
        A sampler used by the `_sample_ingredient_sampler`.

    """

    _space = _sample_space_factory(env.state_space.shape, sample_space)
    _space.seed(seed=seed)

    @sample_generator
    def _sample_generator(*args, **kwargs):
        yield _space.sample()

    return _sample_generator


@sample_ingredient.capture
def _grid_sampler(seed: int, env: DynamicalSystem, sample_space):
    """Grid sampler.

    Returns points on a grid.

    Args:
        seed: Unused.
        env: The dynamical system model.
        sample_space: The sample space configuration variable.

    Returns:
        A sampler used by the `_sample_ingredient_sampler`.

    """

    _space = _sample_space_factory(env.state_space.shape, sample_space)

    @sample_generator
    def _sample_generator(*args, **kwargs):
        xi = grid_ranges(_space, sample_space["grid_resolution"])
        xc = uniform_grid(xi)
        for item in xc:
            yield item

    return _sample_generator


@sample_ingredient.capture
def _random_action_sampler(seed: int, env: DynamicalSystem):
    """Random policy sampler.

    Args:
        seed: Unused.
        env: The dynamical system model.

    Returns:
        A sampler used by the `_sample_ingredient_sampler`.

    """

    _policy = RandomizedPolicy(env)

    @sample_generator
    def _sample_generator(*args, **kwargs):
        yield _policy(*args, **kwargs)

    return _sample_generator


@sample_ingredient.capture
def _zero_action_sampler(seed: int, env: DynamicalSystem):
    """Zero policy sampler.

    Args:
        seed: Unused.
        env: The dynamical system model.

    Returns:
        A sampler used by the `_sample_ingredient_sampler`.

    """
    _policy = ZeroPolicy(env)

    @sample_generator
    def _sample_generator(*args, **kwargs):
        yield _policy(*args, **kwargs)

    return _sample_generator


@sample_ingredient.capture
def _sequential_action_sampler(
    seed: int,
    env: DynamicalSystem,
    sample_space: dict,
    sample_policy: dict,
):
    """Uniform action sampler.

    Generates a sample using multiple actions at a uniform grid of points taken from
    within the range specified by `sample_policy`. Note that this is a simplification to
    make the result appear more uniform, but is not necessary for the correct operation
    of the algorithm. A random iid sample taken from the action space is usually
    sufficient.

    Args:
        seed: Unused.
        env: The dynamical system model.
        sample_space: The sample space configuration variable.
        sample_policy: The sample policy configuration variable.

    Returns:
        A sampler used by the `_sample_ingredient_sampler`.

    """
    _space = _sample_space_factory(env.action_space.shape, sample_policy)

    sample_size = _get_sample_size(env.state_space, sample_space)

    @sample_generator
    def _sample_generator(*args, **kwargs):
        xi = grid_ranges(_space, sample_policy["grid_resolution"])
        xc = uniform_grid(xi)
        for item in xc:
            for i in range(sample_size):
                yield item

    return _sample_generator


@sample_ingredient.capture
def _sample_ingredient_sampler(
    seed: int,
    env: DynamicalSystem,
    sample_space: dict,
    sample_policy: dict,
):
    """Custom sampler for the sample ingredient.

    The sample ingredient requires a sampler which is highly modular, allowing for
    several combinations of config variables. Thus, it uses two generators, one for the
    state space, and another for the action space. These generators are then used within
    a "standard" step-sampler, that generates a (state, action, next_state) tuple.

    Args:
        seed: Unused.
        env: The dynamical system model.
        sample_space: The sample space configuration variable.
        sample_policy: The sample policy configuration variable.

    Returns:
        A sampler that is used by the `generate_sample` function.

    """
    _sampler_map = {
        "random": _default_sampler,
        "uniform": _default_sampler,
        "grid": _grid_sampler,
    }

    state_sampler = _sampler_map[sample_space["sample_scheme"]]
    state_sample_generator = state_sampler(seed=seed, env=env)()

    _action_sampler_map = {
        "random": _random_action_sampler,
        "zero": _zero_action_sampler,
        "grid": _sequential_action_sampler,
    }

    action_sampler = _action_sampler_map[sample_policy["sample_scheme"]]
    action_sample_generator = action_sampler(seed=seed, env=env)()

    @sample_generator
    def _sample_generator(*args, **kwargs):

        state = next(state_sample_generator)
        action = next(action_sample_generator)

        env.state = state
        next_state, cost, done, _ = env.step(action)

        yield (state, action, next_state)

    return _sample_generator


@sample_ingredient.capture
def generate_sample(
    seed: int,
    env: DynamicalSystem,
    sample_space: dict,
    sample_policy: dict,
    action_space: dict,
):
    """Generate a sample based on the ingredient config.

    Generates a sample based on the ingredient configuration variables. For instance, if
    the `sample_space` key `"sample_scheme"` is `"random"`, then the initial conditions
    of the sample are chosen randomly. A similar pattern follows for the `action_space`.
    The `sample_policy` determines the type of policy applied to the system during
    sampling.

    Args:
        seed: Unused.
        env: The dynamical system model.
        sample_space: The sample space configuration variable.
        sample_policy: The sample policy configuration variable.
        action_space: The action_space configuration variable.

    Returns:
        A sample of observations taken from the system evolution.

    """

    _sample_space = _sample_space_factory(env.state_space.shape, sample_space)
    _sample_space.seed(seed=seed)

    _action_space = _sample_space_factory(env.action_space.shape, action_space)
    _action_space.seed(seed=seed)

    _sampler = _sample_ingredient_sampler(seed=seed, env=env)

    sample_size = _get_sample_size(_sample_space, sample_space)

    if sample_policy["sample_scheme"] == "grid":
        action_size = _get_sample_size(_action_space, sample_policy)
        sample_size = sample_size * action_size

    S = sample(
        sampler=_sampler,
        sample_size=sample_size,
    )

    return S


# @sample_ingredient.capture
# def _random_sample(
#     seed: int,
#     env: DynamicalSystem,
#     sample_space: dict,
#     sample_policy: dict,
# ) -> list:

#     _sample_space = _sample_space_factory(env.state_space.shape, sample_space)
#     _sample_space.seed(seed=seed)

#     sample_size = _get_sample_size(_sample_space, sample_space)

#     _sample_policy = _policy_factory(env, sample_policy)

#     # Generate the sample.
#     S = sample(
#         sampler=step_sampler(
#             system=env,
#             policy=_sample_policy,
#             sample_space=_sample_space,
#         ),
#         sample_size=sample_size,
#     )

#     return S


# @sample_ingredient.capture
# def _grid_sample(
#     seed: int,
#     env: DynamicalSystem,
#     sample_space: dict,
#     sample_policy: dict,
# ) -> list:

#     _sample_space = _sample_space_factory(env.state_space.shape, sample_space)
#     _sample_space.seed(seed=seed)

#     _sample_policy = _policy_factory(env, sample_policy)

#     xi = grid_ranges(_sample_space, sample_space["grid_resolution"])
#     sample_size = _get_sample_size(_sample_space, sample_space)

#     # Generate the sample.
#     S = sample(
#         sampler=uniform_grid_step_sampler(
#             xi=xi,
#             system=env,
#             policy=_sample_policy,
#             sample_space=_sample_space,
#         ),
#         sample_size=sample_size,
#     )

#     return S


# @sample_ingredient.capture
# def _grid_action_sample(
#     seed: int,
#     env: DynamicalSystem,
#     sample_space: dict,
#     sample_policy: dict,
#     action_space: dict,
# ) -> list:

#     _sample_space = _sample_space_factory(env.state_space.shape, sample_space)
#     _sample_space.seed(seed=seed)

#     _action_space = _sample_space_factory(env.action_space.shape, action_space)

#     sample_size = _get_sample_size(_sample_space, sample_space)
#     print(type(sample_size))

#     @sample_generator
#     def _grid_action_sampler() -> tuple:
#         """Uniform action sampler.

#         Generates a sample using multiple actions at a uniform grid of points taken from
#         within the range specified by 'ranges'. Note that this is a simplification to
#         make the result appear more uniform, but is not necessary for the correct
#         operation of the algorithm. A random iid sample taken from the state space is
#         sufficient.

#         Yields:
#             observation : Observation of input/output from the stochastic kernel.

#         """

#         xi = grid_ranges(_sample_space, sample_space["grid_resolution"])
#         ui = grid_ranges(_action_space, sample_policy["grid_resolution"])

#         xc = uniform_grid(xi)

#         for action in ui[0]:
#             for point in xc:

#                 state = point

#                 env.state = state
#                 next_state, cost, done, _ = env.step([action])

#                 yield (state, [action], next_state)

#     # Generate the sample.
#     S = sample(
#         sampler=_grid_action_sampler,
#         sample_size=sample_size,
#     )

#     return S


# @sample_ingredient.capture
# def generate_sample(
#     seed: int,
#     env: DynamicalSystem,
#     sample_space: dict,
#     sample_policy: dict,
#     action_space: dict,
# ) -> list:

#     # if sample_space["sample_scheme"] == "random":
#     #     return _random_sample(seed, env)

#     # if sample_space["sample_scheme"] == "uniform":
#     #     return _random_sample(seed, env)

#     # if sample_space["sample_scheme"] == "grid":
#     #     if sample_policy["sample_scheme"] == "grid":
#     #         return _grid_action_sample(seed, env)

#     #     return _grid_sample(seed, env)

#     sample_scheme = sample_space["sample_scheme"]
#     sample_size = _get_sample_size(space=env.state_space, sample_space=sample_space)

#     _sample_space = _sample_space_factory(env.state_space.shape, sample_space)
#     _sample_space.seed(seed=seed)

#     _action_space = _sample_space_factory(env.action_space.shape, action_space)

#     _sampler_map = {
#         "random": step_sampler,
#         "uniform": step_sampler,
#         "grid": uniform_grid_step_sampler,
#     }

#     _sampler = _sampler_map[sample_scheme]

#     if sample_policy["sample_scheme"] == "grid":

#         ui = grid_ranges(_action_space, sample_policy["grid_resolution"])
#         # _sampler = _grid_action_decorator(ui, env, _sample_space, _sampler)

#         # g = _sampler(ui, env=env, sample_space=_sample_space)
#         # print(next(g))
#         # print(next(g))
#         # print(next(g))

#         S = sample(
#             sampler=_grid_action_decorator(
#                 ui,
#                 env=env,
#                 sample_space=_sample_space,
#                 sampler=_sampler,
#             ),
#             sample_size=sample_size,
#         )

#         return S

#     _sample_policy = _policy_factory(env, sample_policy)

#     # Generate the sample.
#     S = sample(
#         sampler=_sampler(
#             system=env,
#             policy=_sample_policy,
#             sample_space=_sample_space,
#         ),
#         sample_size=sample_size,
#     )

#     return S


@sample_ingredient.capture
def generate_admissible_actions(seed: int, env: DynamicalSystem, action_space: dict):
    """Generate a collection of admissible control actions."""

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
