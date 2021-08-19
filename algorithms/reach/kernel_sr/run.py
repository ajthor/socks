# from algorithms.algorithm import AlgorithmRunner
from algorithms.reach.kernel_sr.kernel_sr import KernelSR

import gym
import systems

import numpy as np

import kernel.metrics as kernel
from systems.sample import generate_sample
from systems.sample import generate_uniform_sample

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

from time import time


def main():

    # the system is a 2D integrator with no action space
    system = systems.envs.integrator.StochasticNDIntegratorEnv(2)
    system.action_space = gym.spaces.Box(
        low=0, high=0, shape=system.action_space.shape, dtype=np.float32
    )

    num_time_steps = system.num_time_steps

    # we define the constraint tube such that at the final time step, the system is in a
    # box [-0.5, 0.5]^d, but that all prior time steps the system is in a box [-1, 1]^d.
    constraint_tube = [
        gym.spaces.Box(
            low=-1,
            high=1,
            shape=system.observation_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps - 1)
    ]
    constraint_tube = [
        *constraint_tube,
        gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=system.observation_space.shape,
            dtype=np.float32,
        ),
    ]

    # define the sample space to generate the sample from the stochastic kernel
    sample_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=system.observation_space.shape,
        dtype=np.float32,
    )

    # generate the sample
    S = generate_sample(sample_space, system, 2500)
    # S, _ = generate_uniform_sample(sample_space, system, [50, 50])

    # generate the test points
    # rounding to avoid numpy floating point precision errors
    x1 = np.round(np.linspace(-1, 1, 100), 3)
    x2 = np.round(np.linspace(-1, 1, 100), 3)
    T = [(xx1, xx2) for xx1 in x1 for xx2 in x2]
    # _, T = generate_uniform_sample(sample_space, system, [21, 21])

    t0 = time()

    alg = KernelSR()

    # run the algorithm
    Pr, _ = alg.run(
        system=system, sample=S, test_points=T, constraint_tube=constraint_tube
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

    # plot the result
    plot_results()


def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        fig = plt.figure()
        cm = "viridis"

        # ax = fig.add_subplot(111)
        #
        # x1 = np.round(np.linspace(-1, 1, 100), 3)
        # x2 = np.round(np.linspace(-1, 1, 100), 3)
        # XX, YY = np.meshgrid(x1, x2, indexing="ij")
        # Z = Pr[0].reshape(XX.shape)
        #
        # plt.pcolor(XX, YY, Z, cmap=cm, vmin=0, vmax=1, shading='auto')
        # plt.colorbar()
        #
        # plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")

        ax = fig.add_subplot(111, projection="3d")

        x1 = np.round(np.linspace(-1, 1, 100), 3)
        x2 = np.round(np.linspace(-1, 1, 100), 3)
        XX, YY = np.meshgrid(x1, x2, indexing="ij")
        Z = Pr[0].reshape(XX.shape)

        ax.plot_surface(XX, YY, Z, cmap=cm, linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$\Pr$")
        # ax.view_init(azim=15., elev=30.)

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()