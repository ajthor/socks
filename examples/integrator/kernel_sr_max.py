from gym_socks.algorithms.reach.stochastic_reachability import KernelMaximalSR

import gym
import gym_socks

import numpy as np

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample

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


def main():

    # the system is a 2D integrator
    system = gym_socks.envs.StochasticNDIntegratorEnv(2)

    # system.action_space = gym.spaces.Box(low=-1.1, high=1.1, shape=system.action_space.shape, dtype=np.float32)

    num_time_steps = system.num_time_steps

    # we define the constraints such that at the final time step, the system is in a
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
            low=-1, high=1, shape=system.observation_space.shape, dtype=np.float32
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

    @gym_socks.envs.sample.sample_generator
    def multi_action_sampler():

        ranges = [np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)]
        action_ranges = np.linspace(-1, 1, 5)

        xc = gym_socks.envs.sample.uniform_grid(ranges)

        for action_item in action_ranges:

            for point in xc:
                state = point
                action = [action_item]

                system.state = state
                next_state, cost, done, _ = system.step(action)

                yield (state, action, next_state)

    S = sample(
        sampler=multi_action_sampler,
        sample_size=3125,
    )

    # generate the test points
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    T = gym_socks.envs.sample.uniform_grid([x1, x2])

    # generate the admissible control actions
    A = np.linspace(-1, 1, 10)
    A = np.expand_dims(A, axis=1)

    alg = KernelMaximalSR(kernel_fn=partial(rbf_kernel, gamma=50))

    t0 = time()

    # run the algorithm
    Pr, _ = alg.run(
        system=system,
        S=S,
        A=A,
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

    # plot the result
    plot_results()


def plot_results():

    with open("results/data.npy", "rb") as f:
        Pr = np.load(f)

        colormap = "viridis"

        cm = 1 / 2.54

        # data
        x1 = np.round(np.linspace(-1, 1, 50), 3)
        x2 = np.round(np.linspace(-1, 1, 50), 3)
        XX, YY = np.meshgrid(x1, x2, indexing="ij")
        Z = Pr[0].reshape(XX.shape)

        # flat color map
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_subplot(111)

        plt.pcolor(XX, YY, Z, cmap=colormap, vmin=0, vmax=1, shading="auto")
        plt.colorbar()

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")
        plt.savefig("results/plot.pgf")

        # 3D projection
        fig = plt.figure(figsize=(5 * cm, 5 * cm))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(XX, YY, Z, cmap=colormap, linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$\Pr$")

        plt.savefig("results/plot_3d.png", dpi=300, bbox_inches="tight")
        plt.savefig("results/plot_3d.pgf")


if __name__ == "__main__":
    main()
