"""Simulation ingredient.

Used for experiments where the system needs to be simulated, such as after computing a
policy via one of the included control algorithms.

"""

from sacred import Ingredient

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import BasePolicy

import numpy as np

simulation_ingredient = Ingredient("simulation")


@simulation_ingredient.config
def config():
    """Simulation configuration variables."""

    initial_condition = None  # Initial condition of the system.

    filename = "results/data.npy"


@simulation_ingredient.capture
def simulate_system(
    env: DynamicalSystem, policy: BasePolicy, initial_condition: list
) -> list[list]:
    """Simulate the system from a given initial condition."""

    # Set the initial condition.
    env.reset()

    if initial_condition is None:
        initial_condition = env.state_space.sample()

    env.state = initial_condition

    trajectory = [env.state]

    # Simulate the env using the computed policy.
    for t in range(env.num_time_steps):

        action = np.array(policy(time=t, state=[env.state]))
        obs, cost, done, _ = env.step(action)

        trajectory.append(list(obs))

        if done:
            break

    return trajectory


@simulation_ingredient.capture
def save_simulation(trajectory: list[list], filename: str):
    """Save the result to NPY file."""
    with open(filename, "wb") as f:
        np.save(f, trajectory)


@simulation_ingredient.capture
def load_simulation(filename: str):
    """Load the result from NPY file."""
    with open("results/data.npy", "rb") as f:
        trajectory = np.load(f)

    return trajectory


@simulation_ingredient.config_hook
def _plot_config(config, command_name, logger):
    if command_name == "plot_results":
        return {
            "plot_marker": "None",
            "plot_markersize": 2.5,
            "plot_linewidth": 0.5,
            "plot_linestyle": "--",
        }


@simulation_ingredient.capture
def plot_simulation(
    plt,
    trajectory,
    plot_marker,
    plot_markersize,
    plot_linewidth,
    plot_linestyle,
    *args,
    **kwargs,
):
    trajectory = np.array(trajectory, dtype=np.float32)
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker=plot_marker,
        markersize=plot_markersize,
        linewidth=plot_linewidth,
        linestyle=plot_linestyle,
        *args,
        **kwargs,
    )
