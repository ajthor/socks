"""Stochastic optimal control.

This example demonstrates the optimal controller synthesis algorithm.

By default, it uses a nonlinear dynamical system with nonholonomic vehicle dynamics.
Other dynamical systems can also be used, by modifying the configuration as needed.

Example:

    To run the example, use the following command:

    .. code-block:: shell

        python -m examples.benchmark_tracking_problem

"""


import gym
import gym_socks

import logging

import numpy as np
from numpy.linalg import norm

from sacred import Experiment

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

from gym_socks.algorithms.control import KernelControlFwd
from gym_socks.algorithms.control import KernelControlBwd

from examples._computation_timer import ComputationTimer

from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import set_system_seed
from examples.ingredients.system_ingredient import make_system

from examples.ingredients.simulation_ingredient import simulation_ingredient
from examples.ingredients.simulation_ingredient import simulate_system

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample
from examples.ingredients.sample_ingredient import generate_admissible_actions

from examples.ingredients.tracking_ingredient import tracking_ingredient
from examples.ingredients.tracking_ingredient import compute_target_trajectory
from examples.ingredients.tracking_ingredient import make_cost

from examples.ingredients.plotting_ingredient import plotting_ingredient
from examples.ingredients.plotting_ingredient import update_rc_params


@system_ingredient.config
def system_config():

    system_id = "NonholonomicVehicleEnv-v0"

    sampling_time = 0.1

    action_space = {
        "lower_bound": [0.1, -10.1],
        "upper_bound": [1.1, 10.1],
    }


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
        "grid_resolution": [10, 21],
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
        plotting_ingredient,
    ]
)


@ex.config
def config(sample):
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

    sigma = 3  # Kernel bandwidth parameter.
    # Regularization parameter.
    regularization_param = 1e-5

    time_horizon = 20

    # Whether or not to use dynamic programming (backward in time) algorithm.
    dynamic_programming = False
    batch_size = None  # (Optional) batch size for batch computation.
    heuristic = False  # Whether to use the heuristic solution.

    verbose = True

    results_filename = "results/data.npy"
    no_plot = False


@ex.main
def main(
    seed,
    sigma,
    regularization_param,
    time_horizon,
    dynamic_programming,
    batch_size,
    heuristic,
    verbose,
    results_filename,
    no_plot,
    _log,
):
    """Main experiment."""

    # Make the system.
    env = make_system()

    # Set the random seed.
    set_system_seed(seed=seed, env=env)

    # # Generate the sample.
    S = generate_sample(seed=seed, env=env)

    # Generate the set of admissible control actions.
    A = generate_admissible_actions(seed=seed, env=env)

    # Compute the target trajectory.
    target_trajectory = compute_target_trajectory(time_horizon=time_horizon)
    tracking_cost = make_cost(target_trajectory=target_trajectory)

    if dynamic_programming is True:
        alg_class = KernelControlBwd
    else:
        alg_class = KernelControlFwd

    with ComputationTimer():

        # Compute policy.
        policy = alg_class(
            time_horizon=time_horizon,
            cost_fn=tracking_cost,
            heuristic=heuristic,
            verbose=verbose,
            kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
            regularization_param=regularization_param,
            batch_size=batch_size,
        )

        policy.train(S=S, A=A)

    trajectory = simulate_system(time_horizon, env, policy)

    with open(results_filename, "wb") as f:
        np.save(f, target_trajectory)
        np.save(f, trajectory)

    if not no_plot:
        plot_results()


@plotting_ingredient.config_hook
def plot_config(config, command_name, logger):
    if command_name in {"main", "plot_results"}:
        return {
            "plot_dims": [0, 1],
            "target_trajectory_style": {
                "lines.marker": "x",
                "lines.linestyle": "--",
                "lines.color": "C0",
            },
            "trajectory_style": {
                "lines.linewidth": 1,
                "lines.linestyle": "-",
                "lines.color": "C1",
            },
            "axes": {
                "xlabel": r"$x_1$",
                "ylabel": r"$x_2$",
                "xlim": (-1.1, 1.1),
                "ylim": (-1.1, 1.1),
            },
        }


@ex.command(unobserved=True)
def plot_results(
    system,
    plot_cfg,
):
    """Plot the results of the experiement."""

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    with open("results/data.npy", "rb") as f:
        target_trajectory = np.load(f)
        trajectory = np.load(f)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    # Plot target trajectory.
    with plt.style.context(plot_cfg["target_trajectory_style"]):
        target_trajectory = np.array(target_trajectory, dtype=np.float32)
        plt.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            label="Target Trajectory",
        )

    # Plot generated trajectory.
    with plt.style.context(plot_cfg["trajectory_style"]):
        trajectory = np.array(trajectory, dtype=np.float32)
        plt.plot(
            trajectory[:, plot_cfg["plot_dims"][0]],
            trajectory[:, plot_cfg["plot_dims"][1]],
            label="System Trajectory",
        )

    plt.legend()

    # # Plot the markers as arrows, showing vehicle heading.
    # paper_airplane = [(0, -0.25), (0.5, -0.5), (0, 1), (-0.5, -0.5), (0, -0.25)]

    # if system["system_id"] == "NonholonomicVehicleEnv-v0":
    #     for x in trajectory:
    #         angle = -np.rad2deg(x[2])

    #         t = matplotlib.markers.MarkerStyle(marker=paper_airplane)
    #         t._transform = t.get_transform().rotate_deg(angle)

    #         plt.plot(x[0], x[1], marker=t, markersize=4, linestyle="None", color="C1")

    plt.savefig(plot_cfg["plot_filename"])


if __name__ == "__main__":
    ex.run_commandline()
