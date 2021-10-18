from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.reach.stochastic_reachability import KernelMaximalSR
from gym_socks.envs.sample import sample, uniform_grid

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


def CWHSafeSet(points):

    return np.array(
        (np.abs(points[:, 0]) < np.abs(points[:, 1]))
        & (np.abs(points[:, 2]) <= 0.05)
        & (np.abs(points[:, 3]) <= 0.05),
        dtype=bool,
    )


@ex.config
def config():
    """Experiment configuration variables."""
    sigma = 0.1

    sampling_time = 20
    time_horizon = 100

    initial_condition = [-0.5, -0.75, 0, 0]

    # Sample size.
    sample_size = 2000

    target_state = [0, 0, 0, 0]


@ex.main
def main(
    sigma, sampling_time, time_horizon, initial_condition, sample_size, target_state
):

    # the system is a 4D CWH system
    system = gym_socks.envs.StochasticCWH4DEnv()

    system.action_space = gym.spaces.Box(
        low=-0.1, high=0.1, shape=system.action_space.shape, dtype=np.float32
    )

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

    """
    Generate the sample.

    For this example, we choose points randomly from the region of interest. We choose
    points which have position and velocity just outside of the safe set region, to
    ensure we have examples of infeasible states.
    """
    sample_space = gym.spaces.Box(
        low=np.array([-1.1, -1.1, -0.06, -0.06]),
        high=np.array([1.1, 0.1, 0.06, 0.06]),
        dtype=np.float32,
    )

    S = sample(
        gym_socks.envs.sample.step_sampler(
            system=system,
            policy=gym_socks.envs.policy.RandomizedPolicy(system),
            sample_space=sample_space,
        ),
        sample_size=sample_size,
    )

    # Generate a sample of admissible control actions.
    u1 = np.linspace(-0.1, 0.1, 5)
    u2 = np.linspace(-0.1, 0.1, 5)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    # generate the test points
    x1 = np.linspace(-0.1, 0.1, 25)
    x2 = np.linspace(-0.1, 0.1, 25)
    T = gym_socks.envs.sample.uniform_grid([x1, x2, [0], [0]])

    # T, x = uniform_grid(
    #     sample_space=gym.spaces.Box(
    #         low=np.array([-1, -1, 0, 0]), high=np.array([1, 0, 0, 0]), dtype=np.float32
    #     ),
    #     n=[25, 25, 1, 1],
    # )

    # x1 = x[0]
    # x2 = x[1]

    alg = KernelMaximalSR(kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))))

    t0 = time()

    # run the algorithm
    Pr, _ = alg.run(
        system=system,
        S=S,
        A=A,
        T=T,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem="FHT",
        # batch_size=10,
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


@ex.command(unobserved=True)
def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        colormap = "viridis"

        cm = 1 / 2.54

        # data
        x1 = np.round(np.linspace(-1, 1, 25), 3)
        x2 = np.round(np.linspace(-1, 1, 25), 3)
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
    ex.run_commandline()
    plot_results()
