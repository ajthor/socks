from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.control import KernelControlFwd
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
    sigma = 0.35

    sampling_time = 20
    time_horizon = 100

    initial_condition = [-0.5, -0.75, 0, 0]

    # Sample size.
    sample_size = 1500

    target_state = [0, 0, 0, 0]

    action_space_lb = -0.05
    action_space_ub = 0.05


@ex.main
def main(
    seed,
    sigma,
    sampling_time,
    time_horizon,
    initial_condition,
    sample_size,
    target_state,
    action_space_lb,
    action_space_ub,
):

    # Define a 4D CWH system.
    system = gym_socks.envs.StochasticCWH4DEnv()

    # Set the action space such that the control actions are bounded by -0.1 and 0.1.
    system.action_space = gym.spaces.Box(
        low=action_space_lb,
        high=action_space_ub,
        shape=system.action_space.shape,
        dtype=np.float32,
    )

    # Set the random seed.
    system.seed(seed=seed)
    system.observation_space.seed(seed=seed)
    system.action_space.seed(seed=seed)
    system.disturbance_space.seed(seed=seed)

    # We define a short time horizon for the problem. The sampling time is Ts = 20s.
    system.sampling_time = sampling_time
    system.time_horizon = time_horizon
    num_time_steps = system.num_time_steps

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

    sample_space.seed(seed=seed)

    S = sample(
        gym_socks.envs.sample.step_sampler(
            system=system,
            policy=gym_socks.envs.policy.RandomizedPolicy(system),
            sample_space=sample_space,
        ),
        sample_size=sample_size,
    )

    # Generate a sample of admissible control actions.
    u1 = np.linspace(action_space_lb, action_space_ub, 25)
    u2 = np.linspace(action_space_lb, action_space_ub, 25)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    # Define the cost and constraint functions.

    def cost_fn(time=0, state=None):
        """
        The cost is defined such that we seek to minimize the distance from the system
        to the origin. This would indicate a fully "docked" spacecraft with zero
        terminal velocity.
        """
        return np.power(
            norm(
                state - np.array([target_state]),
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
    policy = KernelControlFwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
        l=1 / (sample_size ** 2),
    )

    policy.train(
        system=system,
        S=S,
        A=A,
        cost_fn=cost_fn,
        constraint_fn=constraint_fn,
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    """
    We now simulate the system under the computed policy from a pre-defined initial
    condition.
    """

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


def plot_results():

    with open("results/data.npy", "rb") as f:

        trajectory = np.load(f)

        colormap = "viridis"

        cm = 1 / 2.54

        fig = plt.figure(figsize=(3, 3))
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
            matplotlib.patches.PathPatch(path, fc="none", ec="blue", lw=0.1)
        )

        plt.gca().add_patch(plt.Rectangle((-0.1, -0.1), 0.2, 0.1, fc="none", ec="green"))

        ax.set_xlabel(r"$x_{1}$")
        ax.set_ylabel(r"$x_{2}$")

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 0.1)

        plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
    plot_results()
