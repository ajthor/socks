"""Stochastic reachability of a double integrator.

This example shows the stochastic reachability algorithm on a double integrator (2D
stochastic chain of integrators) system.

Example:
    To run the example, use the following command:

        $ python -m examples.benchmark_integrator_sr

.. [1] `Model-Free Stochastic Reachability
        Using Kernel Distribution Embeddings, 2019
        Adam J. Thorpe, Meeko M. K. Oishi
        IEEE Control Systems Letters,
        <https://arxiv.org/abs/1908.00697>`_

"""

from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.reach.kernel_sr import kernel_sr
from gym_socks.envs.policy import RandomizedPolicy, ZeroPolicy
from gym_socks.envs.sample import sample

import numpy as np

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

from examples._computation_timer import ComputationTimer


from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import set_system_seed
from examples.ingredients.system_ingredient import make_system

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample

from examples.ingredients.backward_reach_ingredient import (
    backward_reach_ingredient,
    load_safety_probabilities,
    plot_safety_probabilities,
    save_safety_probabilities,
    generate_test_points,
    compute_test_point_ranges,
)
from examples.ingredients.backward_reach_ingredient import generate_tube


@system_ingredient.config
def system_config():
    system_id = "2DIntegratorEnv-v0"

    time_horizon = 4
    sampling_time = 0.25


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "uniform",
        "lower_bound": -1.1,
        "upper_bound": 1.1,
        "sample_size": 2500,
    }

    sample_policy = "zero"

    action_space = {"sample_scheme": "grid"}


@backward_reach_ingredient.config
def backward_reach_config():
    """Backward reachability configuration.

    We define the constraint tube such that at the final time step, the system is in a
    box [-0.5, 0.5]^d, but that all prior time steps the system is in a box [-1, 1]^d.

    """

    constraint_tube_bounds = {"lower_bound": -1, "upper_bound": 1}
    target_tube_bounds = {"lower_bound": -0.5, "upper_bound": 0.5}

    test_points = {
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 100,
    }


ex = Experiment(
    ingredients=[
        system_ingredient,
        sample_ingredient,
        backward_reach_ingredient,
    ]
)


@ex.config
def config():
    """Experiment configuration variables.

    SOCKS uses sacred to run experiments in order to ensure repeatability. Configuration
    variables are parameters that are passed to the experiment, such as the random seed,
    and can be specified at the command-line.

    Example:
        To run the experiment normally, just run:

            $ python -m <experiment>

        To specify configuration variables, use `with variable=value`, e.g.

            $ python -m <experiment> with seed=123

        To use the default configuration, use:

            $ python -m <experiment> with examples/integrator/config.json

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 0.1
    regularization_param = 1

    batch_size = None

    verbose = True


@ex.main
def main(
    seed,
    _log,
    sigma,
    regularization_param,
    backward_reach,
    batch_size,
    verbose,
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

    # Generate the test points.
    T = generate_test_points(env=env)

    with ComputationTimer():

        safety_probabilities = kernel_sr(
            S=S,
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
    save_safety_probabilities(env=env, safety_probabilities=safety_probabilities)


@ex.config_hook
def plot_config(config, command_name, logger):
    if command_name == "plot_results":
        return {
            "fig_width": 3,
            "fig_height": 3,
            "show_x_axis": True,
            "show_y_axis": True,
            "show_x_label": True,
            "show_y_label": True,
            "plot_filename": "results/plot.png",
            "plot_3d": False,
            "elev": 30,
            "azim": -45,
            "colormap": "viridis",
            "plot_time": 0,
        }

    # fig_width = 3
    # fig_height = 3

    # show_x_axis = True
    # show_y_axis = True

    # show_x_label = True
    # show_y_label = True

    # plot_filename = "results/plot.png"

    # plot_3d = False
    # elev = 30
    # azim = -45


@ex.command(unobserved=True)
def plot_results(
    backward_reach,
    fig_width,
    fig_height,
    colormap,
    show_x_axis,
    show_y_axis,
    show_x_label,
    show_y_label,
    plot_time,
    plot_filename,
    plot_3d,
    elev,
    azim,
):
    """Plot the results of the experiement."""

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 8,
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    import matplotlib.pyplot as plt

    plt.set_loglevel("notset")

    # Load the result from NPY file.
    xi, safety_probabilities = load_safety_probabilities()

    x1, x2, *_ = xi
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = safety_probabilities[plot_time].reshape(XX.shape)

    if plot_3d is False:
        # Plot flat color map.
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)

        plot_safety_probabilities(plt, XX, YY, Z)

    else:
        # Plot 3D projection.
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev, azim)

        ax.tick_params(direction="out", pad=-1)

        ax.plot_surface(XX, YY, Z, cmap=colormap, linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)

        ax.set_zlabel(r"$\Pr$")

    if show_x_axis is False:
        ax.get_xaxis().set_visible(False)
    if show_y_axis is False:
        ax.get_yaxis().set_visible(False)

    if show_x_label is True:
        ax.set_xlabel(r"$x_1$")
    if show_y_label is True:
        ax.set_ylabel(r"$x_2$")

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
