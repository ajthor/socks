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

# from gym_socks.algorithms.reach.stochastic_reachability import KernelSR

from gym_socks.algorithms.reach.stochastic_reachability import (
    KernelSR,
    kernel_stochastic_reachability,
)
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
    sample_size = 2500

    test_points_x1 = np.linspace(-1, 1, 50)
    test_points_x2 = np.linspace(-1, 1, 50)

    test_points = [test_points_x1, test_points_x2]

    verbose = True


@ex.main
def main(seed, sigma, sample_size, test_points, verbose):

    system = gym_socks.envs.NDIntegratorEnv(2)

    # Set the random seed.
    system.seed(seed=seed)
    system.observation_space.seed(seed=seed)
    system.state_space.seed(seed=seed)
    system.action_space.seed(seed=seed)

    num_time_steps = system.num_time_steps

    # We define the constraint tube such that at the final time step, the system is in a
    # box [-0.5, 0.5]^d, but that all prior time steps the system is in a box [-1, 1]^d.
    constraint_tube = [
        gym.spaces.Box(
            low=-1,
            high=1,
            shape=system.observation_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    target_tube = [
        gym.spaces.Box(
            low=-0.5, high=0.5, shape=system.observation_space.shape, dtype=np.float32
        )
        for i in range(num_time_steps)
    ]

    # Generate the sample.
    sample_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=system.observation_space.shape,
        dtype=np.float32,
    )

    S = sample(
        sampler=gym_socks.envs.sample.uniform_grid_step_sampler(
            [np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)],
            system=system,
            policy=gym_socks.envs.policy.ZeroPolicy(system),
            sample_space=sample_space,
        ),
        sample_size=sample_size,
    )

    # Generate the test points.
    x1 = test_points[0]
    x2 = test_points[1]
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    t0 = time()

    alg = KernelSR(kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))))

    # Run the algorithm.
    Pr, _ = alg.run(
        system=system,
        S=S,
        T=T,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem="THT",
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # Save the result to NPY file.
    with open("results/data.npy", "wb") as f:
        np.save(f, Pr)


@ex.command(unobserved=True)
def plot_results(test_points):
    """Plot the results of the experiement."""

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

    colormap = "viridis"

    x1 = np.round(test_points[0], 3)
    x2 = np.round(test_points[1], 3)
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = Pr[0].reshape(XX.shape)

    # Plot flat color map.
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    plt.pcolor(XX, YY, Z, cmap=colormap, vmin=0, vmax=1, shading="auto")
    plt.colorbar()

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")

    # Plot 3D projection.
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")

    ax.tick_params(direction="out", pad=-1)
    ax.plot_surface(XX, YY, Z, cmap=colormap, linewidth=0, antialiased=False)
    ax.set_zlim(0, 1)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$\Pr$")

    plt.savefig("results/plot_3d.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
    plot_results()
