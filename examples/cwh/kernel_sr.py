from gym_socks.algorithms.reach.stochastic_reachability import KernelSR

import gym
import gym_socks

import numpy as np

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample
from gym_socks.envs.sample import random_initial_conditions
from gym_socks.envs.sample import uniform_grid

from gym_socks.envs.policy import ZeroPolicy

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


def CWHSafeSet(points):

    return np.array(
        (np.abs(points[:, 0]) < np.abs(points[:, 1]))
        & (np.abs(points[:, 2]) <= 0.05)
        & (np.abs(points[:, 3]) <= 0.05),
        dtype=bool,
    )


def main():

    # the system is a 4D CWH system
    system = gym_socks.envs.StochasticCWH4DEnv()

    num_time_steps = system.num_time_steps

    constraint_tube = [CWHSafeSet for i in range(num_time_steps)]

    target_tube = [
        gym.spaces.Box(
            low=np.array([-0.1, -0.1, -0.01, -0.01]),
            high=np.array([0.1, 0, 0.01, 0.01]),
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    # generate the sample
    initial_conditions = random_initial_conditions(
        system=system,
        sample_space=gym.spaces.Box(
            low=np.array([-1, -1, -0.06, -0.06]),
            high=np.array([1, 0, 0.06, 0.06]),
            dtype=np.float32,
        ),
        n=10000,
    )
    S, U = sample(
        system=system, initial_conditions=initial_conditions, policy=ZeroPolicy(system)
    )

    # generate the test points
    T, x = uniform_grid(
        sample_space=gym.spaces.Box(
            low=np.array([-1, -1, 0, 0]), high=np.array([1, 0, 0, 0]), dtype=np.float32
        ),
        n=[100, 100, 1, 1],
    )

    x1 = x[0]
    x2 = x[1]

    alg = KernelSR(kernel_fn=partial(rbf_kernel, gamma=50))

    t0 = time()

    # run the algorithm
    Pr, _ = alg.run(
        system=system,
        S=S,
        T=T,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem="FHT",
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # save the result to NPY file
    with open("results/data.npy", "wb") as f:
        np.save(f, Pr)

    # save the result to CSV file
    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = Pr[0].reshape(XX.shape)

    np.savetxt(
        "results/data.csv",
        np.column_stack((np.ravel(XX), np.ravel(YY), np.ravel(Z))),
        header="x, y, pr",
        comments="# ",
        delimiter=",",
        newline="\n",
    )


def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        colormap = "viridis"

        cm = 1 / 2.54

        # data
        x1 = np.round(np.linspace(-1, 1, 100), 3)
        x2 = np.round(np.linspace(-1, 1, 100), 3)
        XX, YY = np.meshgrid(x1, x2, indexing="ij")
        Z = Pr[-2].reshape(XX.shape)

        # flat color map
        fig = plt.figure(figsize=(5 * cm, 5 * cm))
        ax = fig.add_subplot(111)

        plt.pcolor(XX, YY, Z, cmap=colormap, vmin=0, vmax=1, shading="auto")
        plt.colorbar()

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")

        # 3D projection
        fig = plt.figure(figsize=(5 * cm, 5 * cm))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(XX, YY, Z, cmap=colormap, linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$\Pr$")

        plt.savefig("results/plot_3d.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
    plot_results()
