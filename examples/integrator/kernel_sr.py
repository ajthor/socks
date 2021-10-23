"""Stochastic reachability of a double integrator.

This example shows the stochastic reachability algorithm on a double integrator (2D
stochastic chain of integrators) system.

Example:
    To run the example, use the following command:

        $ python -m examples.integrator.kernel_sr

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
from gym_socks.algorithms.reach.kernel_sr_rff import kernel_sr_rff
from gym_socks.envs.sample import sample

import numpy as np

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

from time import time

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

plt.set_loglevel("noteset")

ex = Experiment()


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

    system_id = "2DIntegratorEnv-v0"

    sigma = 0.1
    regularization_param = 1

    sample_space_lb = -1.1
    sample_space_ub = 1.1
    sample_size = 2500

    time_horizon = 4
    sampling_time = 0.25

    constraint_tube_lb = -1
    constraint_tube_ub = 1
    target_tube_lb = -0.5
    target_tube_ub = 0.5

    test_points_lb = -1
    test_points_ub = 1
    test_points_grid_resolution = 50

    problem = "THT"

    batch_size = None

    verbose = True

    result_data_filename = "results/data.npy"


@ex.main
def main(
    seed,
    _log,
    system_id,
    sigma,
    regularization_param,
    sample_space_lb,
    sample_space_ub,
    sample_size,
    time_horizon,
    sampling_time,
    constraint_tube_lb,
    constraint_tube_ub,
    target_tube_lb,
    target_tube_ub,
    test_points_lb,
    test_points_ub,
    test_points_grid_resolution,
    problem,
    batch_size,
    verbose,
    result_data_filename,
):
    """Main experiment."""

    system = gym.make(system_id)

    # Set the random seed.
    system.seed(seed=seed)
    system.observation_space.seed(seed=seed)
    system.state_space.seed(seed=seed)
    system.action_space.seed(seed=seed)

    system.time_horizon = time_horizon
    system.sampling_time = sampling_time

    num_time_steps = system.num_time_steps

    # We define the constraint tube such that at the final time step, the system is in a
    # box [-0.5, 0.5]^d, but that all prior time steps the system is in a box [-1, 1]^d.

    if np.isscalar(constraint_tube_lb) is False:
        constraint_tube_lb = np.array(constraint_tube_lb)
    if np.isscalar(constraint_tube_ub) is False:
        constraint_tube_ub = np.array(constraint_tube_ub)

    constraint_tube = [
        gym.spaces.Box(
            low=constraint_tube_lb,
            high=constraint_tube_ub,
            shape=system.state_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    if np.isscalar(target_tube_lb) is False:
        target_tube_lb = np.array(target_tube_lb)
    if np.isscalar(target_tube_ub) is False:
        target_tube_ub = np.array(target_tube_ub)

    target_tube = [
        gym.spaces.Box(
            low=target_tube_lb,
            high=target_tube_ub,
            shape=system.state_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    # Generate the sample.
    sample_space = gym.spaces.Box(
        low=sample_space_lb,
        high=sample_space_ub,
        shape=system.state_space.shape,
        dtype=np.float32,
    )

    S = sample(
        sampler=gym_socks.envs.sample.step_sampler(
            system=system,
            policy=gym_socks.envs.policy.ZeroPolicy(system),
            sample_space=sample_space,
        ),
        sample_size=sample_size,
    )

    # Generate the test points.
    x1 = np.linspace(test_points_lb, test_points_ub, test_points_grid_resolution)
    x2 = np.linspace(test_points_lb, test_points_ub, test_points_grid_resolution)
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    t0 = time()

    Pr = kernel_sr_rff(
        S=S,
        T=T,
        num_steps=system.num_time_steps,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem=problem,
        num_features=1000,
        sigma=sigma,
        regularization_param=regularization_param,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Pr = kernel_sr(
    #     S=S,
    #     T=T,
    #     num_steps=system.num_time_steps,
    #     constraint_tube=constraint_tube,
    #     target_tube=target_tube,
    #     problem=problem,
    #     regularization_param=regularization_param,
    #     kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
    #     batch_size=batch_size,
    #     verbose=verbose,
    # )

    t1 = time()
    _log.info(f"computation time: {t1 - t0} s")

    # Save the result to NPY file.
    with open(result_data_filename, "wb") as f:
        np.save(f, Pr)


@ex.config
def plot_config():

    fig_width = 3
    fig_height = 3

    colormap = "viridis"

    show_x_axis = True
    show_y_axis = True

    show_x_label = True
    show_y_label = True

    show_colorbar = True

    plot_time = 0

    plot_filename = "results/plot.png"

    plot_3d = False
    elev = 30
    azim = -45


@ex.command(unobserved=True)
def plot_results(
    test_points_lb,
    test_points_ub,
    test_points_grid_resolution,
    fig_width,
    fig_height,
    colormap,
    show_x_axis,
    show_y_axis,
    show_x_label,
    show_y_label,
    show_colorbar,
    plot_time,
    result_data_filename,
    plot_filename,
    plot_3d,
    elev,
    azim,
):
    """Plot the results of the experiement."""

    with open(result_data_filename, "rb") as f:
        Pr = np.load(f)

    x1 = np.linspace(test_points_lb, test_points_ub, test_points_grid_resolution)
    x2 = np.linspace(test_points_lb, test_points_ub, test_points_grid_resolution)
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = Pr[plot_time].reshape(XX.shape)

    if plot_3d is False:
        # Plot flat color map.
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)

        plt.pcolor(XX, YY, Z, cmap=colormap, vmin=0, vmax=1, shading="auto")

        if show_colorbar is True:
            plt.colorbar(ax=ax)

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
    plot_results()
