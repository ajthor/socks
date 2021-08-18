from algorithms.reach.monte_carlo.monte_carlo import MonteCarloSR

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

    num_discrete_steps = system.num_time_steps

    constraint_tube = [
        gym.spaces.Box(
            low=-1,
            high=1,
            shape=system.observation_space.shape,
            dtype=np.float32,
        )
        for i in range(num_discrete_steps)
    ]

    # rounding to avoid numpy floating point precision errors
    x1 = np.round(np.linspace(-1, 1, 21), 3)
    x2 = np.round(np.linspace(-1, 1, 21), 3)
    T = [(xx1, xx2) for xx1 in x1 for xx2 in x2]

    alg = MonteCarloSR()

    Pr = alg.run(system=system, test_points=T, constraint_tube=constraint_tube)

    with open("results/data.npy", "wb") as f:
        np.save(f, Pr)

    plot_results()


def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        fig = plt.figure()
        cm = "viridis"

        ax = fig.add_subplot(111, projection="3d")

        x1 = np.round(np.linspace(-1, 1, 21), 3)
        x2 = np.round(np.linspace(-1, 1, 21), 3)
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
