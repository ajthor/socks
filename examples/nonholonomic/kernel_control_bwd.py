from gym_socks.algorithms.control import KernelControlBwd

import gym
import gym_socks

import numpy as np
from numpy.linalg import norm

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample

# from gym_socks.envs.sample import random_initial_conditions
# from gym_socks.envs.sample import uniform_grid

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
        dtype=np.float32,
    )

    system.sampling_time = 0.1
    system.time_horizon = 2
    num_time_steps = system.num_time_steps

    # generate the sample
    sample_space = gym.spaces.Box(
        low=np.array([-1.1, -1.1, -2 * np.pi]),
        high=np.array([1.1, 1.1, 2 * np.pi]),
        dtype=np.float32,
    )

    S = sample(
        sampler=gym_socks.envs.sample.step_sampler(
            system=system,
            policy=gym_socks.envs.policy.RandomizedPolicy(system),
            sample_space=sample_space,
        ),
        sample_size=1500,
    )

    # generate the test points
    u1 = np.linspace(0, 1, 10)
    u2 = np.linspace(-10, 10, 20)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    # # generate the sample
    # initial_conditions = random_initial_conditions(
    #     system=system,
    #     sample_space=gym.spaces.Box(
    #         low=np.array([-1.1, -1.1, -2 * np.pi]),
    #         high=np.array([1.1, 1.1, 2 * np.pi]),
    #         dtype=np.float32,
    #     ),
    #     n=1500,
    # )
    # S, U = sample(
    #     system=system,
    #     initial_conditions=initial_conditions,
    # )
    #
    # A, _ = uniform_grid(
    #     sample_space=gym.spaces.Box(
    #         low=np.array([0, -10]),
    #         high=np.array([1, 10]),
    #         dtype=np.float32,
    #     ),
    #     n=[10, 20],
    # )
    #
    # A = np.expand_dims(A, axis=1)

    def tracking_cost(time=0, state=None):

        return np.power(
            norm(
                state[:, :2]
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

    def tracking_constraint(time=0, state=None):

        return np.array(
            np.all(state[:, :2] >= -0.2, axis=1) & np.all(state[:, :2] <= 0.2, axis=1),
            dtype=np.float32,
        )

    t0 = time()

    # compute policy
    policy = KernelControlBwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (3 ** 2))), l=1 / (len(S) ** 2)
    )
    policy.train_batch(
        system=system,
        S=S,
        A=A,
        cost_fn=tracking_cost,
        # constraint_fn=tracking_constraint,
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # initial condition
    system.state = [-0.8, 0.5, np.pi]
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

        # flat color map
        fig = plt.figure(figsize=(5 * cm, 5 * cm))
        ax = fig.add_subplot(111)

        # plot target trajectory
        target_trajectory = np.array([[-1 + i * 0.1, -1 + i * 0.1] for i in range(21)])
        plt.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            marker="x",
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
        plt.gca().add_patch(plt.Rectangle((-0.2, -0.2), 0.4, 0.4, fc="none", ec="red"))

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
