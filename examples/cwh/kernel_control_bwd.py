from gym_socks.algorithms.control import KernelControlBwd

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

    # Define a 4D CWH system.
    system = gym_socks.envs.StochasticCWH4DEnv()

    # Set the action space such that the control actions are bounded by -0.1 and 0.1.
    system.action_space = gym.spaces.Box(
        low=-0.1, high=0.1, shape=system.action_space.shape, dtype=np.float32
    )

    # We define a short time horizon for the problem. The sampling time is Ts = 20s.
    system.time_horizon = 100
    num_time_steps = system.num_time_steps

    """
    Generate the sample.

    For this example, we choose points randomly from the region of interest. We choose
    points which have velocity and position just outside of the safe set region, to
    ensure we have examples of infeasible states.
    """
    initial_conditions = random_initial_conditions(
        system=system,
        sample_space=gym.spaces.Box(
            low=np.array([-1.1, -1.1, -0.06, -0.06]),
            high=np.array([1.1, 0.1, 0.06, 0.06]),
            dtype=np.float32,
        ),
        n=2000,
    )
    S, U = sample(system=system, initial_conditions=initial_conditions)

    # Generate a sample of admissible control actions.
    A, _ = uniform_grid(
        sample_space=gym.spaces.Box(
            low=-0.1,
            high=0.1,
            shape=system.action_space.shape,
            dtype=np.float32,
        ),
        n=[25, 25],
    )

    A = np.expand_dims(A, axis=1)

    # Define the cost and constraint functions.

    def cost_fn(time=0, state=None):
        """
        The cost is defined such that we seek to minimize the distance from the system
        to the origin. This would indicate a fully "docked" spacecraft with zero
        terminal velocity.
        """
        return np.power(
            norm(
                state
                - np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                ord=2,
                axis=1,
            ),
            2,
        )

    def constraint_fn(time=0, state=None):
        """
        The CWH constraint function is defined as the line of sight (LOS) cone from the
        spacecraft, where the velocity components are sufficiently small.
        """

        return np.array(
            (np.abs(state[:, 0]) < np.abs(state[:, 1]))
            & (np.abs(state[:, 2]) <= 0.05)
            & (np.abs(state[:, 3]) <= 0.05),
            dtype=bool,
        )

    # Run the algorithm.

    t0 = time()

    # Compute policy.
    policy = KernelControlBwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (0.35 ** 2))), l=1 / (len(S) ** 2)
    )
    policy.train_batch(
        system=system,
        S=S,
        U=U,
        A=A,
        cost_fn=cost_fn,
        # constraint_fn=constraint_fn,
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    """
    We now simulate the system under the computed policy from a pre-defined initial
    condition.
    """

    # Define an initial condition.
    system.state = [-0.7, -0.75, 0, 0]
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

        fig = plt.figure(figsize=(3.33, 3.33))
        ax = fig.add_subplot(111)

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
        verts = [(-1, -1), (1, -1), (0, 0), (-1, -1)]
        codes = [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.CLOSEPOLY,
        ]

        path = matplotlib.path.Path(verts, codes)
        plt.gca().add_patch(
            matplotlib.patches.PathPatch(path, fc="none", ec="green", lw=0.1)
        )

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 0.1)

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
