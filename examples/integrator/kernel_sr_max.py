"""Maximal stochastic reachability of a double integrator.

This example shows the maximal stochastic reachability algorithm on a double integrator
(2D stochastic chain of integrators) system.

Example:
    To run the example, use the following command:

        $ python -m examples.integrator.kernel_sr_max

"""

from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.reach.stochastic_reachability import KernelMaximalSR
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

ex = Experiment()


@ex.config
def config():
    """Experiment configuration variables.

    SOCKS uses sacred to run experiments in order to ensure repeatability. Configuration
    variables are parameters that are passed to the experiment, such as the random seed,
    and can be specified at the command-line.

    Example:
        To run the experiment normally, just run:

            $ python -m experiment.<experiment>

        To specify configuration variables, use `with variable=value`, e.g.

            $ python -m experiment.<experiment> with seed=123

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 0.1
    sample_size = 3125

    time_horizon = 4
    sampling_time = 0.25

    grid_resolution = 50

    test_points = [
        np.linspace(-1, 1, grid_resolution),
        np.linspace(-1, 1, grid_resolution),
    ]

    sr_problem = "THT"

    verbose = True


@ex.main
def main(
    seed,
    _log,
    sigma,
    sample_size,
    sampling_time,
    time_horizon,
    test_points,
    sr_problem,
    verbose,
):
    """Main experiment."""

    system = gym_socks.envs.NDIntegratorEnv(2)

    # Set the random seed.
    system.seed(seed=seed)
    system.observation_space.seed(seed=seed)
    system.state_space.seed(seed=seed)
    system.action_space.seed(seed=seed)

    system.time_horizon = time_horizon
    system.sampling_time = sampling_time

    num_time_steps = system.num_time_steps

    # We define the constraints such that at the final time step, the system is in a
    # box [-0.5, 0.5]^d, but that all prior time steps the system is in a box [-1, 1]^d.
    constraint_tube = [
        gym.spaces.Box(
            low=-1,
            high=1,
            shape=system.state_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    target_tube = [
        gym.spaces.Box(low=-1, high=1, shape=system.state_space.shape, dtype=np.float32)
        for i in range(num_time_steps)
    ]

    @gym_socks.envs.sample.sample_generator
    def multi_action_sampler() -> tuple:
        """Multi-action sampler.

        Generates a sample using multiple actions at a uniform grid of points taken from
        within the range specified by 'ranges'. Note that this is a simplification to
        make the result appear more uniform, but is not necessary for the correct
        operation of the algorithm. A random iid sample taken from the state space is
        sufficient.

        Yields:
            observation : Observation of input/output from the stochastic kernel.

        """

        ranges = [np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)]
        action_ranges = np.linspace(-1, 1, 5)

        xc = gym_socks.envs.sample.uniform_grid(ranges)

        for action_item in action_ranges:

            for point in xc:
                state = point
                action = [action_item]

                system.state = state
                next_state, cost, done, _ = system.step(action)

                yield (state, action, next_state)

    # Generate the sample.
    S = sample(
        sampler=multi_action_sampler,
        sample_size=sample_size,
    )

    # Generate the test points.
    x1 = test_points[0]
    x2 = test_points[1]
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    # Generate the admissible control actions.
    A = np.linspace(-1, 1, 10)
    A = np.expand_dims(A, axis=1)

    t0 = time()

    alg = KernelMaximalSR(kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))))

    # Run the algorithm.
    Pr, _ = alg.run(
        system=system,
        S=S,
        A=A,
        T=T,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem=sr_problem,
        verbose=verbose,
    )

    t1 = time()
    _log.info(f"computation time: {t1 - t0} s")

    # Save the result to NPY file.
    with open("results/data.npy", "wb") as f:
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

    x_ticks = [-1, -0.5, 0, 0.5, 1]
    y_ticks = [-1, -0.5, 0, 0.5, 1]

    show_colorbar = True

    plot_time = 0


@ex.command(unobserved=True)
def plot_results(
    test_points,
    fig_width,
    fig_height,
    colormap,
    show_x_axis,
    show_y_axis,
    show_x_label,
    show_y_label,
    show_colorbar,
    plot_time,
):
    """Plot the results of the experiement."""

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

    x1 = np.round(test_points[0], 3)
    x2 = np.round(test_points[1], 3)
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = Pr[plot_time].reshape(XX.shape)

    # Plot flat color map.
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    plt.pcolor(XX, YY, Z, cmap=colormap, vmin=0, vmax=1, shading="auto")
    if show_colorbar is True:
        plt.colorbar(ax=ax)

    if show_x_axis is False:
        ax.get_xaxis().set_visible(False)
    if show_y_axis is False:
        ax.get_yaxis().set_visible(False)

    if show_x_label is True:
        ax.set_xlabel(r"$x_1$")
    if show_y_label is True:
        ax.set_ylabel(r"$x_2$")

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/plot.pgf")

    # Plot 3D projection.
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(XX, YY, Z, cmap=colormap, linewidth=0, antialiased=False)
    ax.set_zlim(0, 1)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$\Pr$")

    plt.savefig("results/plot_3d.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/plot_3d.pgf")


if __name__ == "__main__":
    ex.run_commandline()
    plot_results()
