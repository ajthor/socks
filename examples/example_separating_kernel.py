"""Forward reachability example.

This file demonstrates the forward reachability classifier on a set of dummy data. Note
that the data is not taken from a dynamical system, but can easily be adapted to data
taken from system observations via a simple substitution. The reason for the dummy data
is to showcase the technique on a non-convex forward reachable set.

Example:
    To run the example, use the following command:

        $ python -m examples.forward_reach.forward_reach

.. [1] `Learning Approximate Forward Reachable Sets Using Separating Kernels, 2021
        Adam J. Thorpe, Kendric R. Ortiz, Meeko M. K. Oishi
        Learning for Dynamics and Control,
        <https://arxiv.org/abs/2011.09678>`_

"""

import gym
import gym_socks

import numpy as np

from functools import partial
from sacred import Experiment

from gym_socks.algorithms.reach.separating_kernel import SeparatingKernelClassifier

from sklearn.metrics.pairwise import euclidean_distances

from gym_socks.envs.sample import sample

from examples._computation_timer import ComputationTimer
from examples.ingredients.forward_reach_ingredient import forward_reach_ingredient
from examples.ingredients.forward_reach_ingredient import generate_test_points


@forward_reach_ingredient.config
def forward_reach_config():

    test_points = {
        "lower_bound": -1,
        "upper_bound": 1,
        "grid_resolution": 100,
    }


ex = Experiment(ingredients=[forward_reach_ingredient])


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

    sigma = 0.1
    sample_size = 500

    regularization_param = 1 / sample_size

    filename = "results/data.npy"


@ex.main
def main(_log, seed, sigma, regularization_param, sample_size, filename):
    """Main experiment."""

    np.random.seed(seed)

    @gym_socks.envs.sample.sample_generator
    def donut_sampler() -> tuple:
        """Sample generator.

        Sample generator that generates points in a donut-shaped ring around the origin.
        An example of a non-convex region.

        Yields:
            sample : A sample taken iid from the region.

        """

        r = np.random.uniform(low=0.5, high=0.75, size=(1,))
        phi = np.random.uniform(low=0, high=2 * np.pi, size=(1,))
        point = np.array([r * np.cos(phi), r * np.sin(phi)])

        yield tuple(np.ravel(point))

    # Sample the distribution.
    S = sample(sampler=donut_sampler, sample_size=sample_size)

    # Generate the test points.
    T = generate_test_points((2,))

    with ComputationTimer():

        kernel_fn = partial(
            gym_socks.kernel.metrics.abel_kernel,
            sigma=sigma,
            distance_fn=euclidean_distances,
        )

        # Construct the algorithm.
        alg = SeparatingKernelClassifier(
            kernel_fn=kernel_fn,
            regularization_param=regularization_param,
        )

        # Train and classify the test points.
        alg.fit(S)
        labels = alg.predict(T)

    if not np.any(labels):
        _log.warning("No test points classified within reach set.")

    with open(filename, "wb") as f:
        np.save(f, S)
        np.save(f, T)
        np.save(f, labels)


@forward_reach_ingredient.config_hook
def _plot_config(config, command_name, logger):
    if command_name == "plot_results":
        return {
            "plot_marker": "None",
            "plot_markersize": 2.5,
            "plot_linewidth": 0.5,
            "plot_linestyle": "--",
            "dpi": 300,
        }


@ex.command(unobserved=True)
def plot_results(filename):
    """Plot the results of the experiement."""

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

    with open(filename, "rb") as f:
        S = np.array(np.load(f))
        T = np.array(np.load(f))
        labels = np.load(f)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    points_in = T[labels == True]

    plt.scatter(
        points_in[:, 0],
        points_in[:, 1],
        color="C0",
        marker=",",
        s=(300.0 / fig.dpi) ** 2,
    )

    plt.scatter(S[:, 0], S[:, 1], color="r", marker=".", s=1)

    # Plot support region.
    plt.gca().add_patch(plt.Circle((0, 0), 0.5, fc="none", ec="blue", lw=0.5))
    plt.gca().add_patch(plt.Circle((0, 0), 0.75, fc="none", ec="blue", lw=0.5))

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
