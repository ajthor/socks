from gym_socks.algorithms.reach.forward_reach import KernelForwardReachClassifier

import gym
import gym_socks

import numpy as np

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample

# from gym_socks.envs.sample import uniform_initial_conditions
# from gym_socks.envs.sample import uniform_grid

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


def main():

    @gym_socks.envs.sample.sample_generator
    def donut_sampler():

        r = np.random.uniform(low=0.5, high=0.75, size=(1,))
        phi = np.random.uniform(low = 0, high=2*np.pi, size=(1,))
        point = np.array([r * np.cos(phi), r * np.sin(phi)])

        yield np.ravel(point)

    S = sample(sampler=donut_sampler, sample_size=1000)

    # generate the test points
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    alg = KernelForwardReachClassifier(kernel_fn=partial(kernel.abel_kernel, sigma=0.25))

    t0 = time()

    alg.train(S)
    classifications = alg.classify(T)

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    with open("results/sample_points.npy", "wb") as f:
        np.save(f, S)

    with open("results/test_points.npy", "wb") as f:
        np.save(f, T)

    with open("results/classifications.npy", "wb") as f:
        np.save(f, classifications)


def plot_results():

    with open("results/sample_points.npy", "rb") as f:
        S = np.load(f)
        S = np.array(S)

    with open("results/test_points.npy", "rb") as f:
        T = np.load(f)
        T = np.array(T)

    with open("results/classifications.npy", "rb") as f:
        classifications = np.load(f)

    colormap = "viridis"

    # flat color map
    fig = plt.figure(figsize=(3.33, 3.33))
    ax = fig.add_subplot(111)

    points_in = T[classifications == True]
    points_out = T[classifications == False]

    plt.scatter(points_in[:, 0], points_in[:, 1], color="C0", marker=",")
    plt.scatter(points_out[:, 0], points_out[:, 1], color="C1", marker=",")

    plt.scatter(S[:, 0], S[:, 1], color="b", marker=".")

    # plot support region
    plt.gca().add_patch(plt.Circle((0, 0), 0.5, fc="none", ec="blue", lw=0.5))
    plt.gca().add_patch(plt.Circle((0, 0), 0.75, fc="none", ec="blue", lw=0.5))

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
    plot_results()
