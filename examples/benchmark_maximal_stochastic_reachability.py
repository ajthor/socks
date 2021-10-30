"""Maximal stochastic reachability.

This example shows the maximal stochastic reachability algorithm.

By default, the system is a double integrator (2D stochastic chain of integrators).

Example:
    To run the example, use the following command:

        $ python -m examples.benchmark_maximal_stochastic_reachability

.. [1] `Model-Free Stochastic Reachability
        Using Kernel Distribution Embeddings, 2019
        Adam J. Thorpe, Meeko M. K. Oishi
        IEEE Control Systems Letters,
        <https://arxiv.org/abs/1908.00697>`_

"""

import gym
import gym_socks

import numpy as np

from sacred import Experiment

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

from gym_socks.algorithms.reach.kernel_sr_max import kernel_sr_max

from gym_socks.envs.sample import sample

from examples._computation_timer import ComputationTimer

from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import set_system_seed
from examples.ingredients.system_ingredient import make_system

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample
from examples.ingredients.sample_ingredient import generate_admissible_actions

from examples.ingredients.backward_reach_ingredient import backward_reach_ingredient
from examples.ingredients.backward_reach_ingredient import generate_test_points
from examples.ingredients.backward_reach_ingredient import compute_test_point_ranges
from examples.ingredients.backward_reach_ingredient import generate_tube

from examples.ingredients.plotting_ingredient import plotting_ingredient
from examples.ingredients.plotting_ingredient import update_rc_params


@system_ingredient.config
def system_config():
    system_id = "2DIntegratorEnv-v0"

    time_horizon = 4
    sampling_time = 0.25


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "grid",
        "lower_bound": -1.1,
        "upper_bound": 1.1,
        "grid_resolution": 25,
    }

    sample_policy = {
        "sample_scheme": "grid",
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 5,
    }

    action_space = {"sample_scheme": "grid"}


@backward_reach_ingredient.config
def backward_reach_config():

    # By default, the constraint tube is defined as a box [-1, 1]^d.
    constraint_tube_bounds = {"lower_bound": -1, "upper_bound": 1}
    # By default, the target tube is defined as a box [-0.5, 0.5]^d.
    target_tube_bounds = {"lower_bound": -0.5, "upper_bound": 0.5}

    test_points = {
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 25,
    }


ex = Experiment(
    ingredients=[
        system_ingredient,
        sample_ingredient,
        backward_reach_ingredient,
        plotting_ingredient,
    ]
)


@ex.config
def config():
    """Experiment configuration variables.

    SOCKS uses sacred to run experiments in order to ensure repeatability. Configuration
    variables are parameters that are passed to the experiment, such as the random seed,
    and can be specified at the command-line.

    Example:
        To run the experiment normally, use:

            $ python -m <experiment>

        The full configuration can be viewed using:

            $ python -m <experiment> print_config

        To specify configuration variables, use `with variable=value`, e.g.

            $ python -m <experiment> with seed=123 system.time_horizon=5

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 0.1
    regularization_param = 1

    batch_size = None

    verbose = True

    results_filename = "results/data.npy"
    no_plot = False


@ex.main
def main(
    seed,
    _log,
    sigma,
    regularization_param,
    backward_reach,
    batch_size,
    verbose,
    results_filename,
    no_plot,
):
    """Main experiment."""

    env = make_system()

    # Set the random seed.
    set_system_seed(seed, env)

    # Generate the target and constraint tubes.
    target_tube = generate_tube(env, backward_reach["target_tube_bounds"])
    constraint_tube = generate_tube(env, backward_reach["constraint_tube_bounds"])

    # Generate the sample.
    S = generate_sample(seed=seed, env=env)

    # Generate admissible control actions.
    A = generate_admissible_actions(seed=seed, env=env)

    # Generate the test points.
    T = generate_test_points(env=env)

    with ComputationTimer():

        safety_probabilities = kernel_sr_max(
            S=S,
            A=A,
            T=T,
            num_steps=env.num_time_steps,
            constraint_tube=constraint_tube,
            target_tube=target_tube,
            problem=backward_reach["problem"],
            regularization_param=regularization_param,
            kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
            batch_size=batch_size,
            verbose=verbose,
        )

    # Save the result to NPY file.
    _log.debug(f"Saving the results to file {results_filename}.")
    xi = compute_test_point_ranges(env)
    with open(results_filename, "wb") as f:
        np.save(f, xi)
        np.save(f, safety_probabilities)

    if not no_plot:
        plot_results()


@plotting_ingredient.config_hook
def _plot_config(config, command_name, logger):
    if command_name in {"main", "plot_results"}:
        return {
            "plot_time": 0,
            "plot_system_dims": [0, 1],
            "pcolor_style": {
                "vmin": 0,
                "vmax": 1,
                "shading": "auto",
            },
            "axes": {
                "xlabel": r"$x_1$",
                "ylabel": r"$x_2$",
            },
            "colorbar": True,
        }


@ex.command(unobserved=True)
def plot_results(
    plot_cfg,
    results_filename,
):
    """Plot the results of the experiement."""

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    matplotlib.set_loglevel("notset")
    plt.set_loglevel("notset")

    # Load the result from NPY file.
    with open(results_filename, "rb") as f:
        xi = np.load(f)
        safety_probabilities = np.load(f)

    plot_system_dims = plot_cfg["plot_system_dims"]
    x1 = xi[plot_system_dims[0]]
    x2 = xi[plot_system_dims[1]]
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = safety_probabilities[plot_cfg["plot_time"]].reshape(XX.shape)

    # Plot flat color map.
    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    plt.pcolor(XX, YY, Z, **plot_cfg["pcolor_style"])

    if plot_cfg["colorbar"] is True:
        plt.colorbar()

    plt.savefig(plot_cfg["plot_filename"])


@plotting_ingredient.config_hook
def plot_config_3d(config, command_name, logger):
    if command_name in {"plot_results_3d"}:
        return {
            "plot_time": 0,
            "plot_system_dims": [0, 1],
            "axes": {
                "elev": 30,
                "azim": -45,
                "xlabel": r"$x_1$",
                "ylabel": r"$x_2$",
                "zlabel": r"$\Pr$",
                "zlim": (0, 1),
            },
            "cmap": "viridis",
            "surface_style": {
                "linewidth": 0,
                "antialiased": False,
            },
        }


@ex.command(unobserved=True)
def plot_results_3d(plot_cfg, results_filename):

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    rc_params = matplotlib.rc_params_from_file(
        fname=plot_cfg["rc_params_filename"],
        use_default_template=True,
    )
    matplotlib.rcParams.update(rc_params)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    matplotlib.set_loglevel("notset")
    plt.set_loglevel("notset")

    # Load the result from NPY file.
    with open(results_filename, "rb") as f:
        xi = np.load(f)
        safety_probabilities = np.load(f)

    plot_system_dims = plot_cfg["plot_system_dims"]
    x1 = xi[plot_system_dims[0]]
    x2 = xi[plot_system_dims[1]]
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = safety_probabilities[plot_cfg["plot_time"]].reshape(XX.shape)

    # Plot 3D projection.
    fig = plt.figure()
    ax = plt.axes(projection="3d", **plot_cfg["axes"])

    mappable = cm.ScalarMappable(cmap=plot_cfg["cmap"])
    mappable.set_array(Z)
    mappable.set_clim(vmin=0, vmax=1)

    surf = ax.plot_surface(
        XX,
        YY,
        Z,
        cmap=mappable.cmap,
        norm=mappable.norm,
        **plot_cfg["surface_style"],
    )

    plt.savefig(plot_cfg["plot_filename"])


if __name__ == "__main__":
    ex.run_commandline()
