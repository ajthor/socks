"""Stochastic Optimal Control for a nonholonomic vehicle system.

This file demonstrates the optimal controller synthesis algorithm on a nonlinear
dynamical system with nonholonomic vehicle dynamics.

Example:
    To run the example, use the following command:

        $ python -m examples.nonholonomic.kernel_control_fwd

.. [1] `Stochastic Optimal Control via
        Hilbert Space Embeddings of Distributions, 2021
        Adam J. Thorpe, Meeko M. K. Oishi
        IEEE Conference on Decision and Control,
        <https://arxiv.org/abs/2103.12759>`_

"""

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
    sigma = 3

    sampling_time = 0.1
    time_horizon = 2

    initial_condition = [-0.8, 0, 0]

    sample_size = 1500

    amplitude = 0.5
    period = 2.0

    verbose = True


@ex.capture
def compute_target_trajectory(
    amplitude: float, period: float, sampling_time: float, time_horizon: float
) -> list[list]:
    """Computes the traget trajectory to follow.

    The default trajectory is a V-shaped path based on a triangle function. The amplitude and period are set by the config.

    Args:
        amplitude : The amplitude of the triangle function.
        period : The period of the triangle function.
        sampling_time : The sampling time of the dynamical system. Ensures that there
            are a number of points in the target trajectory equal to the number of time
            steps in the simulation.
        time_horizon : The time horizon of the dynamical system. Ensures that there are
            a number of points in the target trajectory equal to the number of time
            steps in the simulation.

    Returns:
        target_trajectory : The target trajectory as a list of points.

    """

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
def main(
    seed, sigma, sampling_time, time_horizon, initial_condition, sample_size, verbose
):
    """Main experiment."""

    system = gym_socks.envs.NonholonomicVehicleEnv()

    system.action_space = gym.spaces.Box(
        low=np.array([0.1, -10.1]),
        high=np.array([1.1, 10.1]),
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

    # Generate the set of admissible control actions.
    u1 = np.linspace(0.1, 1, 10)
    u2 = np.linspace(-10, 10, 20)
    A = gym_socks.envs.sample.uniform_grid([u1, u2])

    # Compute the target trajectory.
    target_trajectory = compute_target_trajectory()

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
                state[:, :2] - np.array([target_trajectory[time]]),
                ord=2,
                axis=1,
            ),
            2,
        )

    t0 = time()

    # Compute policy.
    policy = KernelControlFwd(
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
        regularization_param=1 / (sample_size ** 2),
    )

    policy.train(
        system=system,
        S=S,
        A=A,
        cost_fn=tracking_cost,
        verbose=verbose,
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
    target_trajectory = np.array(compute_target_trajectory())
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
        marker="None",
        markersize=2.5,
        linewidth=0.5,
        linestyle="--",
    )

    # Plot the markers as arrows, showing vehicle heading.
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
