"""Linear system identification example.

This file demonstrates the linear system identification algorithm.

By default, it uses the CWH4D system dynamics. Try setting the regularization parameter
lower for higher accuracy. Note that this can introduce numerical instability if set too
low.

Example:

    To run the example, use the following command:

    .. code-block:: shell

        python -m examples.example_linear_id

"""

import gym
import gym_socks

import logging

import numpy as np

from functools import partial
from sacred import Experiment

from gym_socks.policies import ConstantPolicy

from gym_socks.algorithms.identification.kernel_linear_id import kernel_linear_id

from benchmarks._computation_timer import ComputationTimer

from benchmarks.ingredients.system_ingredient import system_ingredient
from benchmarks.ingredients.system_ingredient import set_system_seed
from benchmarks.ingredients.system_ingredient import make_system

from benchmarks.ingredients.sample_ingredient import sample_ingredient
from benchmarks.ingredients.sample_ingredient import generate_sample

from benchmarks.ingredients.simulation_ingredient import simulation_ingredient
from benchmarks.ingredients.simulation_ingredient import simulate_system

from benchmarks.ingredients.plotting_ingredient import plotting_ingredient
from benchmarks.ingredients.plotting_ingredient import update_rc_params


@system_ingredient.config
def system_config():

    system_id = "CWH4DEnv-v0"


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "random",
        "sample_size": 10,
    }


@simulation_ingredient.config
def simulation_config():

    initial_condition = [-0.75, -0.75, 0, 0]


ex = Experiment(
    ingredients=[
        system_ingredient,
        sample_ingredient,
        simulation_ingredient,
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

    # The regularization parameter.
    regularization_param = 1e-9

    time_horizon = 1000

    results_filename = "results/data.npy"
    no_plot = False


@ex.main
def main(
    simulation,
    seed,
    regularization_param,
    time_horizon,
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

    with ComputationTimer():

        # Compute the approximation.
        alg = kernel_linear_id(S=S, regularization_param=regularization_param)

    # Simulate the system using the actual dynamics.
    policy = ConstantPolicy(action_space=env.action_space, constant=[0.01, 0.01])
    actual_trajectory = simulate_system(time_horizon, env, policy)

    # Simulate the system using the approximated dynamics.
    estimated_trajectory = [simulation["initial_condition"]]
    for t in range(time_horizon):
        action = policy(time=t, state=[env.state])
        state = alg.predict(T=estimated_trajectory[t], U=action)

        estimated_trajectory.append(state)

    with open(results_filename, "wb") as f:
        np.save(f, actual_trajectory)
        np.save(f, estimated_trajectory)

    if not no_plot:
        plot_results()


@plotting_ingredient.config_hook
def plot_config(config, command_name, logger):
    if command_name in {"main", "plot_results"}:
        return {
            "actual_trajectory_style": {
                "lines.linestyle": "-",
                "lines.color": "C0",
            },
            "estimated_trajectory_style": {
                "lines.marker": ".",
                "lines.linestyle": "-",
                "lines.color": "C1",
            },
            "axes": {
                "xlabel": r"$x_1$",
                "ylabel": r"$x_2$",
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
        actual_trajectory = np.load(f)
        estimated_trajectory = np.load(f)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    # Plot actual trajectory.
    with plt.style.context(plot_cfg["actual_trajectory_style"]):
        actual_trajectory = np.array(actual_trajectory, dtype=np.float32)
        plt.plot(
            actual_trajectory[:, 0],
            actual_trajectory[:, 1],
            label="Actual Trajectory",
        )

    # Plot estimated trajectory.
    with plt.style.context(plot_cfg["estimated_trajectory_style"]):
        estimated_trajectory = np.array(estimated_trajectory, dtype=np.float32)
        plt.plot(
            estimated_trajectory[:, 0],
            estimated_trajectory[:, 1],
            label="Estimated Trajectory",
        )

    plt.legend()

    plt.savefig(plot_cfg["plot_filename"])


if __name__ == "__main__":
    ex.run_commandline()
