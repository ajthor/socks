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

from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.reach.forward_reach import KernelForwardReachClassifier
from gym_socks.envs.sample import sample

import numpy as np

from functools import partial
from sklearn.metrics.pairwise import euclidean_distances

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
    sample_size = 500


@ex.main
def main(sigma, sample_size):
    """Main experiment."""

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
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    # Construct the algorithm.
    alg = KernelForwardReachClassifier(
        kernel_fn=partial(
            gym_socks.kernel.metrics.abel_kernel,
            sigma=sigma,
            distance_fn=euclidean_distances,
        )
    )

    t0 = time()

    # Train and classify the test points.
    alg.train(S)
    classifications = alg.classify(T)

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    with open("results/forward_reach.npy", "wb") as f:
        np.save(f, S)
        np.save(f, T)
        np.save(f, classifications)


@ex.command(unobserved=True)
def plot_results():
    """Plot the results of the experiement."""

    with open("results/forward_reach.npy", "rb") as f:
        S = np.array(np.load(f))
        T = np.array(np.load(f))
        classifications = np.load(f)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    points_in = T[classifications == True]
    points_out = T[classifications == False]

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
    plot_results()
