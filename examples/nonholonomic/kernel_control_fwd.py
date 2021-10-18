from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.control import KernelControlFwd
from gym_socks.envs.sample import sample

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

    initial_condition = [-0.8, 0, 0]

    sample_size = 1500

    amplitude = 0.5
    period = 2.0


@ex.capture
def compute_target_trajectory(amplitude, period, sampling_time, time_horizon):

    a = amplitude
    p = period

    target_trajectory = [
        [
            (x * 0.1) - 1.0,
            4 * a / p * np.abs((((((x * 0.1) - 1.0) - p / 2) % p) + p) % p - p / 2) - a,
        ]
        for x in range(int(time_horizon / sampling_time) + 1)
    ]

    return target_trajectory


@ex.main
def main(seed, sigma, sampling_time, time_horizon, initial_condition, sample_size):

    system = gym_socks.envs.StochasticNonholonomicVehicleEnv()

    system.action_space = gym.spaces.Box(
        low=np.array([0.1, -10.1]),
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

    # generate the sample
    sample_space = gym.spaces.Box(
        low=np.array([-1.2, -1.2, -2 * np.pi]),
        high=np.array([1.2, 1.2, 2 * np.pi]),
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
    u1 = np.linspace(0.1, 1, 10)
    u2 = np.linspace(-10, 10, 20)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    target_trajectory = compute_target_trajectory()

    def tracking_cost(time=0, state=None):

        return np.power(
            norm(
                state[:, :2] - np.array([target_trajectory[time]]),
                ord=2,
                axis=1,
            ),
            2,
        )

    t0 = time()

    # compute policy
    policy = KernelControlFwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
        l=1 / (sample_size ** 2),
    )

    policy.train(
        system=system,
        S=S,
        A=A,
        cost_fn=tracking_cost,
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # initial condition
    system.state = initial_condition
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

    pass


def plot_results():

    with open("results/data.npy", "rb") as f:

        trajectory = np.load(f)

        colormap = "viridis"

        # flat color map
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)

        # plot target trajectory
        # target_trajectory = np.array([[-1 + i * 0.1, -1 + i * 0.1] for i in range(21)])
        target_trajectory = np.array(compute_target_trajectory())
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
            marker="None",
            markersize=2.5,
            linewidth=0.5,
            linestyle="--",
        )

        paper_airplane = [(0, -0.25), (0.5, -0.5), (0, 1), (-0.5, -0.5), (0, -0.25)]

        for x in trajectory:
            angle = -np.rad2deg(x[2])

            t = matplotlib.markers.MarkerStyle(marker=paper_airplane)
            t._transform = t.get_transform().rotate_deg(angle)

            plt.plot(x[0], x[1], marker=t, markersize=4, linestyle="None", color="C1")

        ax.set_xlabel(r"$x_{1}$")
        ax.set_ylabel(r"$x_{2}$")

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
    plot_results()
