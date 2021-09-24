from gym_socks.algorithms.control import KernelControlFwd

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

    system = gym_socks.envs.StochasticNDPointMassEnv(2)

    system.action_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=system.action_space.shape,
        dtype=np.float32,
    )

    system.sampling_time = 0.25
    system.time_horizon = 2
    num_time_steps = system.num_time_steps

    # generate the sample
    initial_conditions = random_initial_conditions(
        system=system,
        sample_space=gym.spaces.Box(
            low=-1.1,
            high=1.1,
            shape=system.observation_space.shape,
            dtype=np.float32,
        ),
        n=5000,
    )

    S, U = sample(
        system=system,
        initial_conditions=initial_conditions,
    )

    A, _ = uniform_grid(
        sample_space=gym.spaces.Box(
            low=-1,
            high=1,
            shape=system.action_space.shape,
            dtype=np.float32,
        ),
        n=[21, 21],
    )

    A = np.expand_dims(A, axis=1)

    # define the cost function
    def tracking_cost(time=0, state=None):

        return np.power(
            norm(
                state
                - np.array(
                    [
                        [
                            -1 + time * system.sampling_time,
                            -1 + time * system.sampling_time,
                        ]
                    ]
                ),
                ord=2,
                axis=1,
            ),
            2,
        )

    t0 = time()

    # compute policy
    policy = KernelControlFwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (0.15 ** 2))), l=1 / (len(S) ** 2)
    )
    policy.train(system=system, S=S, U=U, A=A, cost_fn=tracking_cost)

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # initial condition
    system.state = [-0.5, 0]
    trajectory = [system.state]

    for t in range(num_time_steps):

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

        # trajectory plot
        fig = plt.figure(figsize=(5 * cm, 5 * cm))
        ax = fig.add_subplot(111)

        # plot target trajectory
        target_trajectory = np.array([[-1 + i * 0.25, -1 + i * 0.25] for i in range(9)])
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

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
