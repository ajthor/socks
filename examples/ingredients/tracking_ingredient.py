from sacred import Ingredient

import gym

import numpy as np

tracking_ingredient = Ingredient("tracking")


@tracking_ingredient.config
def _config():

    path_amplitude = 0.5
    path_period = 2.0

    norm_order = 2
    squared = True


@tracking_ingredient.capture
def compute_target_trajectory(num_steps, path_amplitude, path_period):
    """Computes the traget trajectory to follow.

    The default trajectory is a V-shaped path based on a triangle function. The amplitude and period are set by the config.

    Args:
        amplitude : The amplitude of the triangle function.
        period : The period of the triangle function.
        sampling_time : The sampling time of the dynamical system. Ensures that there
            are a number of points in the target trajectory equal to the number of time
            steps in the simulation.
        time_horizon : The time horizon of the dynamical system. Ensures that there are
            a number of points in the target trajectory equal to the number of time
            steps in the simulation.

    Returns:
        target_trajectory : The target trajectory as a list of points.

    """

    a = path_amplitude
    p = path_period

    target_trajectory = [
        [
            (x * 0.1) - 1.0,
            4 * a / p * np.abs((((((x * 0.1) - 1.0) - p / 2) % p) + p) % p - p / 2) - a,
        ]
        for x in range(num_steps)
    ]

    return target_trajectory


@tracking_ingredient.capture
def make_cost(target_trajectory, norm_order, squared):
    def _tracking_cost(time: int = 0, state: np.ndarray = None) -> float:
        """Tracking cost function.

        The goal is to minimize the distance of the x/y position of the vehicle to the
        'state' of the target trajectory at each time step.

        Args:
            time : Time of the simulation. Used for time-dependent cost functions.
            state : State of the system.

        Returns:
            cost : Real-valued cost.

        """

        dist = state[:, :2] - np.array([target_trajectory[time]])
        result = np.linalg.norm(dist, ord=norm_order, axis=1)
        if squared is True:
            result = np.power(result, 2)
        return result

    return _tracking_cost


@tracking_ingredient.config_hook
def _plot_config(config, command_name, logger):
    if command_name == "plot_results":
        return {
            "plot_marker": "x",
            "plot_markersize": 2.5,
            "plot_linewidth": 0.5,
            "plot_linestyle": "--",
        }

    # plot_marker = "x"
    # plot_markersize = 2.5
    # plot_linewidth = 0.5
    # plot_linestyle = "--"


@tracking_ingredient.capture
def plot_target_trajectory(
    plt,
    target_trajectory,
    plot_marker,
    plot_markersize,
    plot_linewidth,
    plot_linestyle,
    *args,
    **kwargs,
):
    target_trajectory = np.array(target_trajectory, dtype=np.float32)
    plt.plot(
        target_trajectory[:, 0],
        target_trajectory[:, 1],
        marker=plot_marker,
        markersize=plot_markersize,
        linewidth=plot_linewidth,
        linestyle=plot_linestyle,
        *args,
        **kwargs,
    )
