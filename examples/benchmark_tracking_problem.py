"""Stochastic Optimal Control for a nonholonomic vehicle system.

This example demonstrates the optimal controller synthesis algorithm on a nonlinear
dynamical system with nonholonomic vehicle dynamics.

Example:
    To run the example, use the following command:

        $ python -m examples.nonholonomic.kernel_control_fwd

.. [1] `Stochastic Optimal Control via
        Hilbert Space Embeddings of Distributions, 2021
        Adam J. Thorpe, Meeko M. K. Oishi
        IEEE Conference on Decision and Control,
        <https://arxiv.org/abs/2103.12759>`_

"""

from os import system
from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.control import KernelControlFwd
from gym_socks.algorithms.control import KernelControlBwd
from gym_socks.envs.policy import ZeroPolicy, RandomizedPolicy
from gym_socks.envs.sample import sample, transpose_sample

import numpy as np
from numpy.linalg import norm

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

from examples._computation_timer import ComputationTimer

from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import set_system_seed
from examples.ingredients.system_ingredient import make_system

from examples.ingredients.simulation_ingredient import simulation_ingredient
from examples.ingredients.simulation_ingredient import simulate_system
from examples.ingredients.simulation_ingredient import load_simulation
from examples.ingredients.simulation_ingredient import save_simulation
from examples.ingredients.simulation_ingredient import plot_simulation

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample
from examples.ingredients.sample_ingredient import generate_admissible_actions

from examples.ingredients.tracking_ingredient import tracking_ingredient
from examples.ingredients.tracking_ingredient import compute_target_trajectory
from examples.ingredients.tracking_ingredient import make_cost
from examples.ingredients.tracking_ingredient import plot_target_trajectory


@system_ingredient.config
def system_config():

    system_id = "NonholonomicVehicleEnv-v0"

    sampling_time = 0.1
    time_horizon = 2


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "uniform",
        "lower_bound": [-1.2, -1.2, -2 * np.pi],
        "upper_bound": [1.2, 1.2, 2 * np.pi],
        "sample_size": 1500,
    }

    action_space = {
        "sample_scheme": "grid",
        "lower_bound": [0.1, -10.1],
        "upper_bound": [1.1, 10.1],
        "grid_resolution": [10, 20],
    }


@simulation_ingredient.config
def simulation_config():

    initial_condition = [-0.8, 0, 0]


ex = Experiment(
    ingredients=[
        system_ingredient,
        simulation_ingredient,
        sample_ingredient,
        tracking_ingredient,
    ]
)


@ex.config
def config(sample):
    """Experiment configuration variables.

    SOCKS uses sacred to run experiments in order to ensure repeatability.
    Configuration variables are parameters that are passed to the experiment, such as
    the random seed, and can be specified at the command-line.

    Example:
        To run the experiment normally, use::

            $ python -m experiment.<experiment>

        To specify configuration variables, use `with variable=value`, e.g.::

            $ python -m experiment.<experiment> with seed=123

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 3  # Kernel bandwidth parameter.
    # Regularization parameter.
    regularization_param = 1 / (sample["sample_space"]["sample_size"] ** 2)

    # Whether or not to use dynamic programming (backward in time) algorithm.
    dynamic_programming = False
    batch_size = None  # (Optional) batch size for batch computation.
    heuristic = False  # Whether to use the heuristic solution.

    verbose = True


@ex.main
def main(
    seed,
    _log,
    sigma,
    regularization_param,
    dynamic_programming,
    batch_size,
    heuristic,
    verbose,
):
    """Main experiment."""

    # Make the system.
    env = make_system()

    env.action_space = gym.spaces.Box(
        low=np.array([0.1, -10.1]),
        high=np.array([1.1, 10.1]),
        dtype=np.float32,
    )

    # Set the random seed.
    set_system_seed(seed=seed, env=env)

    # # Generate the sample.
    S = generate_sample(seed=seed, env=env)

    # Generate the set of admissible control actions.
    A = generate_admissible_actions(seed=seed, env=env)

    # Compute the target trajectory.
    target_trajectory = compute_target_trajectory(num_steps=env.num_time_steps)
    tracking_cost = make_cost(target_trajectory=target_trajectory)

    if dynamic_programming is True:
        alg_class = KernelControlBwd
    else:
        alg_class = KernelControlFwd

    with ComputationTimer():

        # Compute policy.
        policy = alg_class(
            num_steps=env.num_time_steps,
            cost_fn=tracking_cost,
            heuristic=heuristic,
            verbose=verbose,
            kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
            regularization_param=regularization_param,
            batch_size=batch_size,
        )

        policy.train(S=S, A=A)

    trajectory = simulate_system(env, policy)
    save_simulation(trajectory)


@ex.config_hook
def plot_config(config, command_name, logger):
    if command_name == "plot_results":
        config.update(
            {
                "fig_width": 3,
                "fig_height": 3,
                "show_x_axis": True,
                "show_y_axis": True,
                "show_x_label": True,
                "show_y_label": True,
            }
        )

    return config
    # fig_width = 3
    # fig_height = 3

    # show_x_axis = True
    # show_y_axis = True
    # show_x_label = True
    # show_y_label = True


@ex.command(unobserved=True)
def plot_results(
    fig_width,
    fig_height,
    show_x_axis,
    show_y_axis,
    show_x_label,
    show_y_label,
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

    trajectory = load_simulation()

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    # Plot target trajectory.
    target_trajectory = np.array(compute_target_trajectory(len(trajectory)))
    plot_target_trajectory(plt, target_trajectory)

    # Plot generated trajectory.
    plot_simulation(plt, trajectory, color="C1")

    # Plot the markers as arrows, showing vehicle heading.
    paper_airplane = [(0, -0.25), (0.5, -0.5), (0, 1), (-0.5, -0.5), (0, -0.25)]

    for x in trajectory:
        angle = -np.rad2deg(x[2])

        t = matplotlib.markers.MarkerStyle(marker=paper_airplane)
        t._transform = t.get_transform().rotate_deg(angle)

        plt.plot(x[0], x[1], marker=t, markersize=4, linestyle="None", color="C1")

    if show_x_axis is False:
        ax.get_xaxis().set_visible(False)
    if show_y_axis is False:
        ax.get_yaxis().set_visible(False)

    if show_x_label is True:
        ax.set_xlabel(r"$x_1$")
    if show_y_label is True:
        ax.set_ylabel(r"$x_2$")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
