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


def main():

    system = systems.envs.integrator.StochasticNDIntegratorEnv(2)

    num_time_steps = system.num_time_steps

    constraint_tube = [
        gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=system.observation_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    sample_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=system.observation_space.shape,
        dtype=np.float32,
    )

    # S = generate_sample(sample_space, system, 5000)
    S, _ = generate_uniform_sample(sample_space, system, [71, 71])

    # rounding to avoid numpy floating point precision errors
    x1 = np.round(np.linspace(-1, 1, 81), 3)
    x2 = np.round(np.linspace(-1, 1, 81), 3)
    T = [(xx1, xx2) for xx1 in x1 for xx2 in x2]
    # _, T = generate_uniform_sample(sample_space, system, [21, 21])

    alg = KernelSR()

    Pr, _ = alg.run(
        system=system, sample=S, test_points=T, constraint_tube=constraint_tube
    )

    with open("results/data.npy", "wb") as f:
        np.save(f, Pr)

    plot_results()


def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        fig = plt.figure()
        cm = "viridis"

        # ax = fig.add_subplot(111)
        #
        # XX, YY = np.meshgrid(x1, x2, indexing="ij")
        # Z = Pr[5].reshape(XX.shape)
        #
        # plt.pcolor(XX, YY, Z, cmap=cm, vmin=0, vmax=1)
        # plt.colorbar()
        #
        # plt.savefig("plot.png", dpi=300, bbox_inches="tight")

        ax = fig.add_subplot(111, projection="3d")

        x1 = np.round(np.linspace(-1, 1, 81), 3)
        x2 = np.round(np.linspace(-1, 1, 81), 3)
        XX, YY = np.meshgrid(x1, x2, indexing="ij")
        Z = Pr[0].reshape(XX.shape)

        ax.plot_surface(XX, YY, Z, cmap=cm, linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$\Pr$")
        # ax.view_init(elev=90., azim=0.)

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
