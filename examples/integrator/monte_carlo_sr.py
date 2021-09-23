from gym_socks.algorithms.reach.monte_carlo import MonteCarloSR

import gym
import gym_socks

import numpy as np

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample
from gym_socks.envs.sample import uniform_initial_conditions
from gym_socks.envs.sample import uniform_grid

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

    # the system is a 2D integrator with no action space
    system = gym_socks.envs.StochasticNDIntegratorEnv(2)

    def policy(time, state):
        return [0]

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
        for i in range(num_time_steps)
    ]

    target_tube = [
        gym.spaces.Box(
            low=-0.5, high=0.5, shape=system.observation_space.shape, dtype=np.float32
        )
        for i in range(num_time_steps)
    ]

    # generate the test points
    T, x = uniform_grid(
        sample_space=gym.spaces.Box(
            low=-1, high=1, shape=system.observation_space.shape, dtype=np.float32
        ),
        n=[10, 10],
    )

    alg = MonteCarloSR()

    t0 = time()

    Pr = alg.run(
        system=system,
        T=T,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        num_iterations=50,
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    with open("results/data.npy", "wb") as f:
        np.save(f, Pr)

    plot_results()


def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        colormap = "viridis"

        cm = 1 / 2.54

        x1 = np.round(np.linspace(-1, 1, 10), 3)
        x2 = np.round(np.linspace(-1, 1, 10), 3)
        XX, YY = np.meshgrid(x1, x2, indexing="ij")
        Z = Pr[0].reshape(XX.shape)

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
