"""Stochastic Optimal Control for a point mass system.

This example demonstrates the constrained optimal controller synthesis algorithm
(dynamic programming) on a point mass dynamical system.

Example:
    To run the example, use the following command:

        $ python -m examples.point_mass.kernel_control_bwd

.. [1] `Stochastic Optimal Control via
        Hilbert Space Embeddings of Distributions, 2021
        Adam J. Thorpe, Meeko M. K. Oishi
        IEEE Conference on Decision and Control,
        <https://arxiv.org/abs/2103.12759>`_

"""

from sacred import Experiment

import gym
import gym_socks

from gym_socks.algorithms.control import KernelControlBwd
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
    """Experiment configuration variables.

    SOCKS uses sacred to run experiments in order to ensure repeatability. Configuration
    variables are parameters that are passed to the experiment, such as the random seed,
    and can be specified at the command-line.

    Example:
        To run the experiment normally, just run:

            $ python -m experiment.<experiment>

        To specify configuration variables, use `with variable=value`, e.g.

            $ python -m experiment.<experiment> with seed=123

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 1

    sampling_time = 0.1
    time_horizon = 2

    initial_condition = [-0.8, 0]

    sample_size = 2500


@ex.main
def main(seed, sigma, sampling_time, time_horizon, initial_condition, sample_size):
    """Main experiment."""

    system = gym_socks.envs.NDPointMassEnv(2)

    system.action_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=system.action_space.shape,
        dtype=np.float32,
    )

    # Set the random seed.
    system.seed(seed=seed)
    system.observation_space.seed(seed=seed)
    system.state_space.seed(seed=seed)
    system.action_space.seed(seed=seed)

    system.sampling_time = sampling_time
    system.time_horizon = time_horizon
    num_time_steps = system.num_time_steps

    # Generate the sample.
    sample_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=system.observation_space.shape,
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

    # Generate the set of admissible control actions.
    u1 = np.linspace(-1, 1, 21)
    u2 = np.linspace(-1, 1, 21)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    def tracking_cost(time: int = 0, state: np.ndarray = None) -> float:
        """Tracking cost function.

        The goal is to minimize the distance of the x/y position of the vehicle to the
        'state' of the target trajectory at each time step.

        Args:
            time : Time of the simulation. Used for time-dependent cost functions.
            state : State of the system.

        Returns:
            cost : Real-valued cost.

        """

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

    def tracking_constraint(time: int = 0, state: np.ndarray = None) -> bool:
        """Tracking constraint function.

        The constraint is defined as a box centered around the origin. The goal is to
        track the target trajectory as closely as possible while avoiding the obstacle.

        Args:
            time : Time of the simulation. Used for time-dependent constraint functions.
            state : State of the system.

        Returns:
            satisfies_constraints : A boolean vector indicating whether the constraint
                is satisfied or not.

        """

        return np.array(
            np.all(state >= -0.2, axis=1) & np.all(state <= 0.2, axis=1),
            dtype=bool,
        )

    t0 = time()

    # Compute policy.
    policy = KernelControlBwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
        regularization_param=1 / (sample_size ** 2),
    )

    policy.train_batch(
        system=system,
        S=S,
        A=A,
        cost_fn=tracking_cost,
        constraint_fn=tracking_constraint,
    )

    t1 = time()
    print(f"Total time: {t1 - t0} s")

    # Set the initial condition.
    system.state = initial_condition
    trajectory = [system.state]

    # Simulate the system using the computed policy.
    for t in range(num_time_steps):

        action = np.array(policy(time=t, state=[system.state]))

        obs, cost, done, _ = system.step(action)

        trajectory.append(list(obs))

        if done:
            break

    # Save the result to NPY file.
    with open("results/data.npy", "wb") as f:
        np.save(f, trajectory)


@ex.command(unobserved=True)
def plot_results():
    """Plot the results of the experiement."""

    with open("results/data.npy", "rb") as f:
        trajectory = np.load(f)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    # Plot target trajectory.
    target_trajectory = np.array([[-1 + i * 0.1, -1 + i * 0.1] for i in range(21)])
    plt.plot(
        target_trajectory[:, 0],
        target_trajectory[:, 1],
        marker="x",
        markersize=2.5,
        linewidth=0.5,
        linestyle="--",
    )

    # Plot generated trajectory.
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="^",
        markersize=2.5,
        linewidth=0.5,
        linestyle="--",
    )

    # Plot the constraint box.
    plt.gca().add_patch(plt.Rectangle((-0.2, -0.2), 0.4, 0.4, fc="none", ec="red"))

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ex.run_commandline()
    plot_results()
