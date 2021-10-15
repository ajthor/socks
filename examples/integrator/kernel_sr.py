from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.reach.stochastic_reachability import KernelSR
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
    """Experiment configuration variables."""
    sigma = 0.1
    sample_size = 2500


@ex.main
def main(sigma, sample_size):

    # the system is a 2D integrator with no action space
    system = gym_socks.envs.StochasticNDIntegratorEnv(2)

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

    # generate the sample
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

    # generate the test points
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    alg = KernelSR(kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))))

    t0 = time()

    # run the algorithm
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

        # data
        x1 = np.round(np.linspace(-1, 1, 100), 3)
        x2 = np.round(np.linspace(-1, 1, 100), 3)
        XX, YY = np.meshgrid(x1, x2, indexing="ij")
        Z = Pr[0].reshape(XX.shape)

        # flat color map
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_subplot(111)

        plt.pcolor(XX, YY, Z, cmap=colormap, vmin=0, vmax=1, shading="auto")
        plt.colorbar()

        # fig.tight_layout()

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")

        # 3D projection
        fig = plt.figure(figsize=(3.33, 3.33))
        ax = fig.add_subplot(111, projection="3d")

        ax.tick_params(direction="out", pad=-1)
        ax.plot_surface(XX, YY, Z, cmap=colormap, linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$\Pr$")

        fig.tight_layout()

        plt.savefig("results/plot_3d.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
    plot_results()
