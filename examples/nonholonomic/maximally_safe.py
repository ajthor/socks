from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.reach.maximally_safe import MaximallySafePolicy
from gym_socks.envs.sample import sample, uniform_grid

import numpy as np
from numpy.linalg import norm

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
    sigma = 3

    sampling_time = 0.1
    time_horizon = 2

    initial_condition = [-0.8, 0.5, np.pi]

    sample_size = 1500


@ex.main
def main(seed, sigma, sampling_time, time_horizon, initial_condition, sample_size):

    system = gym_socks.envs.StochasticNonholonomicVehicleEnv()

    system.action_space = gym.spaces.Box(
        low=np.array([-0.1, -10.1]),
        high=np.array([1.1, 10.1]),
        dtype=np.float32,
    )

    # Set the random seed.
    system.seed(seed=seed)
    system.observation_space.seed(seed=seed)
    system.action_space.seed(seed=seed)
    system.disturbance_space.seed(seed=seed)

    system.sampling_time = sampling_time
    system.time_horizon = time_horizon
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
    sample_space = gym.spaces.Box(
        low=np.array([-1.1, -1.1, -2 * np.pi]),
        high=np.array([1.1, 1.1, 2 * np.pi]),
        dtype=np.float32,
    )

    sample_space.seed(seed=seed)

    S = sample(
        sampler=gym_socks.envs.sample.step_sampler(
            system=system,
            policy=gym_socks.envs.policy.RandomizedPolicy(system),
            sample_space=sample_space,
        ),
        sample_size=sample_size,
    )

    # generate the test points
    u1 = np.linspace(0, 1, 10)
    u2 = np.linspace(-10, 10, 20)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    t0 = time()

    # compute policy
    policy = MaximallySafePolicy(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2)))
    )

    policy.train_batch(
        system=system,
        S=S,
        A=A,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem="FHT",
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # initial condition
    system.state = initial_condition
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
    ex.run_commandline()
    plot_results()
