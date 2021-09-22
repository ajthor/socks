# from algorithms.algorithm import AlgorithmRunner
from gym_socks.algorithms.control.kernel_control.kernel_control import KernelControlBwd

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

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
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

    # generate the sample
    initial_conditions = random_initial_conditions(
        system=system,
        sample_space=gym.spaces.Box(
            low=np.array([-1.1, -1.1, -2 * np.pi - 0.1]),
            high=np.array([1.1, 1.1, 2 * np.pi + 0.1]),
            # shape=system.observation_space.shape,
            dtype=np.float32,
        ),
        n=4000,
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
                ord=1,
                axis=1,
            ),
            2,
        )

    # compute policy
    policy = KernelControlBwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (3 ** 2))), l=1 / (len(S) ** 2)
    )
    policy.train(system=system, S=S, U=U, A=A, cost_fn=tracking_cost)

    # initial condition
    system.state = [-0.8, 0, -np.pi]
    trajectory = [system.state]

    for t in range(num_time_steps):

        action = np.array(policy(time=t, state=[system.state]))

        obs, reward, done, _ = system.step(action[0])

        # action = system.action_space.sample()
        # obs, reward, done, _ = system.step(action)

        # print(obs)

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

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
