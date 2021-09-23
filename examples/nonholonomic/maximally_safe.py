from gym_socks.algorithms.reach.maximally_safe import MaximallySafePolicy

import gym
import gym_socks

import numpy as np
from numpy.linalg import norm

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample
from gym_socks.envs.sample import random_initial_conditions
from gym_socks.envs.sample import uniform_grid

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

    system = gym_socks.envs.StochasticNonholonomicVehicleEnv()

    system.action_space = gym.spaces.Box(
        low=np.array([-0.1, -10.1]),
        high=np.array([1.1, 10.1]),
        # shape=system.action_space.shape,
        dtype=np.float32,
    )

    system.sampling_time = 0.1
    system.time_horizon = 2
    num_time_steps = system.num_time_steps

    constraint_tube = [
        gym.spaces.Box(
            low=np.array([-1, -1, -np.inf]),
            high=np.array([1, 1, np.inf]),
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    target_tube = [
        gym.spaces.Box(
            low=np.array(
                [
                    -1.2 + i * system.sampling_time,
                    -1.2 + i * system.sampling_time,
                    -np.inf,
                ]
            ),
            high=np.array(
                [
                    -0.8 + i * system.sampling_time,
                    -0.8 + i * system.sampling_time,
                    np.inf,
                ]
            ),
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    # generate the sample
    initial_conditions = random_initial_conditions(
        system=system,
        sample_space=gym.spaces.Box(
            low=np.array([-1.1, -1.1, -2 * np.pi]),
            high=np.array([1.1, 1.1, 2 * np.pi]),
            # shape=system.observation_space.shape,
            dtype=np.float32,
        ),
        n=1800,
    )
    S, U = sample(
        system=system,
        initial_conditions=initial_conditions,
    )

    A, _ = uniform_grid(
        sample_space=gym.spaces.Box(
            low=np.array([0, -10]),
            high=np.array([1, 10]),
            # shape=system.observation_space.shape,
            dtype=np.float32,
        ),
        n=[10, 20],
    )

    A = np.expand_dims(A, axis=1)

    t0 = time()

    # compute policy
    policy = MaximallySafePolicy(kernel_fn=partial(rbf_kernel, gamma=50))
    policy.train_batch(
        system=system,
        S=S,
        U=U,
        A=A,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem="FHT",
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # initial condition
    system.state = [-0.8, 0.5, np.pi]
    trajectory = [system.state]

    for t in range(num_time_steps - 1):

        action = np.array(policy(time=t, state=[system.state]))
        obs, reward, done, _ = system.step(action)

        trajectory.append(list(obs))

        if done:
            break

    # save the result to NPY file
    with open("results/data.npy", "wb") as f:
        np.save(f, trajectory)

    # plot the result
    plot_results()


def plot_results():

    with open("results/data.npy", "rb") as f:

        trajectory = np.load(f)

        colormap = "viridis"

        cm = 1 / 2.54

        # flat color map
        fig = plt.figure(figsize=(5 * cm, 5 * cm))
        ax = fig.add_subplot(111)

        # plot target trajectory
        target_trajectory = np.array([[-1 + i * 0.1, -1 + i * 0.1] for i in range(21)])
        plt.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            marker="X",
            markersize=2.5,
            linewidth=0.5,
            linestyle="--",
        )

        # plot generated trajectory
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            marker="^",
            markersize=2.5,
            linewidth=0.5,
            linestyle="--",
        )

        # plot constraint box
        # plt.gca().add_patch(plt.Rectangle((-0.2, -0.2), 0.4, 0.4, fc='none', ec="red"))
        for i in range(21):
            plt.gca().add_patch(
                plt.Rectangle(
                    (-1.2 + (i * 0.1), -1.2 + (i * 0.1)),
                    0.4,
                    0.4,
                    fc="none",
                    ec="green",
                    linewidth=0.25,
                )
            )

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
