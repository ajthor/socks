"""Satellite rendezvous and docking.

This example demonstrates the optimal controller synthesis algorithm on a satellite rendezvous and docking problem with CWH dynamics.

Example:

    To run the example, use the following command:

    .. code-block:: shell

        python -m examples.benchmark_cwh_problem

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

from gym_socks.envs.sample import transpose_sample

from examples._computation_timer import ComputationTimer

from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import set_system_seed
from examples.ingredients.system_ingredient import make_system

from examples.ingredients.simulation_ingredient import simulation_ingredient
from examples.ingredients.simulation_ingredient import simulate_system

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample
from examples.ingredients.sample_ingredient import generate_admissible_actions

from examples.ingredients.cwh_ingredient import cwh_ingredient
from examples.ingredients.cwh_ingredient import make_cost
from examples.ingredients.cwh_ingredient import make_constraint

from examples.ingredients.plotting_ingredient import plotting_ingredient
from examples.ingredients.plotting_ingredient import update_rc_params


@system_ingredient.config
def system_config():
    system_id = "CWH4DEnv-v0"

    sampling_time = 20

    action_space = {
        "lower_bound": -0.1,
        "upper_bound": 0.1,
    }


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "grid",
        "lower_bound": [-1.1, -1.1, -0.06, -0.06],
        "upper_bound": [1.1, 0.1, 0.06, 0.06],
        "grid_resolution": [10, 10, 5, 5],
    }

    sample_policy = {
        "sample_scheme": "random",
    }

    action_space = {
        "sample_scheme": "uniform",
        "lower_bound": -0.05,
        "upper_bound": 0.05,
        "sample_size": 500,
    }


@simulation_ingredient.config
def simulation_config():

    initial_condition = [-0.75, -0.75, 0, 0]


ex = Experiment(
    ingredients=[
        system_ingredient,
        simulation_ingredient,
        sample_ingredient,
        cwh_ingredient,
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

    sigma = 0.35  # Kernel bandwidth parameter.
    # Regularization parameter.
    # regularization_param = 1 / (sample["sample_space"]["sample_size"] ** 2)
    regularization_param = 1e-5

    time_horizon = 5

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
    simulation,
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
    cost_fn = make_cost(env=env)
    constraint_fn = make_constraint(time_horizon=time_horizon, env=env)

    if dynamic_programming is True:
        alg_class = KernelControlBwd
    else:
        alg_class = KernelControlFwd

    with ComputationTimer():

        # Compute policy.
        policy = alg_class(
            time_horizon=time_horizon,
            cost_fn=cost_fn,
            constraint_fn=constraint_fn,
            heuristic=heuristic,
            verbose=verbose,
            kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
            regularization_param=regularization_param,
            batch_size=batch_size,
        )

        policy.train(S=S, A=A)

    trajectory = simulate_system(time_horizon, env, policy)

    with open(results_filename, "wb") as f:
        np.save(f, trajectory)

    if not no_plot:
        plot_results()


@plotting_ingredient.config_hook
def plot_config(config, command_name, logger):
    if command_name in {"main", "plot_results", "plot_sample"}:
        return {
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
                "ylim": (-1.1, 0.1),
            },
        }


@ex.command(unobserved=True)
def plot_results(plot_cfg, _log):
    """Plot the results of the experiement."""

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    with open("results/data.npy", "rb") as f:
        trajectory = np.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, **plot_cfg["axes"])

    # plot constraint box
    verts = [(-1, -1), (1, -1), (0, 0), (-1, -1)]
    codes = [
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.CLOSEPOLY,
    ]

    path = matplotlib.path.Path(verts, codes)
    plt.gca().add_patch(matplotlib.patches.PathPatch(path, fc="none", ec="blue"))

    plt.gca().add_patch(plt.Rectangle((-0.2, -0.2), 0.4, 0.2, fc="none", ec="green"))

    # Plot generated trajectory.
    with plt.style.context(plot_cfg["trajectory_style"]):
        trajectory = np.array(trajectory, dtype=np.float32)
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label="System Trajectory",
        )

    plt.legend()

    plt.savefig(plot_cfg["plot_filename"])


@ex.command(unobserved=True)
def plot_sample(seed, plot_cfg):
    """Plot a sample taken from the system."""

    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    matplotlib.set_loglevel("notset")
    plt.set_loglevel("notset")

    # Make the system.
    env = make_system()

    # Set the random seed.
    set_system_seed(seed=seed, env=env)

    # # Generate the sample.
    S = generate_sample(seed=seed, env=env)
    X, U, Y = transpose_sample(S)
    X = np.array(X)
    U = np.array(U)
    Y = np.array(Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, **plot_cfg["axes"])

    # plot constraint box
    verts = [(-1, -1), (1, -1), (0, 0), (-1, -1)]
    codes = [
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.CLOSEPOLY,
    ]

    path = matplotlib.path.Path(verts, codes)
    plt.gca().add_patch(matplotlib.patches.PathPatch(path, fc="none", ec="blue"))

    plt.gca().add_patch(plt.Rectangle((-0.2, -0.2), 0.4, 0.2, fc="none", ec="green"))

    plt.scatter(X[:, 0], X[:, 1], marker=".")
    plt.scatter(Y[:, 0], Y[:, 1], marker=".")

    plt.savefig("results/plot_sample.png")


if __name__ == "__main__":
    ex.run_commandline()
