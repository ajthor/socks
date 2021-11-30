"""Stochastic optimal control.

This example demonstrates the optimal controller synthesis algorithm.

By default, it uses a nonlinear dynamical system with nonholonomic vehicle dynamics. Other dynamical systems can also be used, by modifying the configuration as needed.

Several configuration files are included in the `examples/configs` folder, and can be used by running the example using the `with` syntax, e.g.

    $ python -m <experiment> with examples/configs/<config_file>

Example:
    To run the example, use the following command:

        $ python -m examples.benchmark_tracking_problem

.. [1] `Stochastic Optimal Control via
        Hilbert Space Embeddings of Distributions, 2021
        Adam J. Thorpe, Meeko M. K. Oishi
        IEEE Conference on Decision and Control,
        <https://arxiv.org/abs/2103.12759>`_

"""


import gym
import gym_socks

import matplotlib.pyplot as plt

import logging

import numpy as np
from numpy.linalg import norm

from sacred import Experiment

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from gym_socks.kernel.metrics import abel_kernel

from gym_socks.algorithms.control import KernelControlCC

from gym_socks.envs.sample import sample as _sample, transpose_sample

# from gym_socks.envs.sample import trajectory_sampler
from gym_socks.envs.sample import reshape_trajectory_sample, sample_generator

from gym_socks.envs.policy import RandomizedPolicy, BasePolicy, PDController

from gym_socks.envs.dynamical_system import DynamicalSystem

from examples._computation_timer import ComputationTimer

from examples.ingredients.system_ingredient import system_ingredient
from examples.ingredients.system_ingredient import set_system_seed
from examples.ingredients.system_ingredient import make_system

from examples.ingredients.simulation_ingredient import simulation_ingredient
from examples.ingredients.simulation_ingredient import simulate_system

from examples.ingredients.sample_ingredient import sample_ingredient
from examples.ingredients.sample_ingredient import generate_sample
from examples.ingredients.sample_ingredient import generate_admissible_actions
from examples.ingredients.sample_ingredient import _get_sample_size

from examples.ingredients.plotting_ingredient import plotting_ingredient
from examples.ingredients.plotting_ingredient import update_rc_params

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


@system_ingredient.config
def system_config():

    system_id = "RepeatedIntegratorEnv-v0"

    sampling_time = 0.1

    action_space = {
        "lower_bound": [-1.1, -1.1],
        "upper_bound": [1.1, 1.1],
    }


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "uniform",
        "lower_bound": [-0.6, 0.1, -0.6, 0.1],
        "upper_bound": [-0.4, 0.1, -0.4, 0.1],
        "sample_size": 500,
    }

    action_space = {
        "sample_scheme": "uniform",
        "lower_bound": [-1.1, -1.1],
        "upper_bound": [1.1, 1.1],
        "sample_size": 500,
    }


@simulation_ingredient.config
def simulation_config():

    initial_condition = [-0.5, 0.1, -0.5, 0.1]


def cc_trajectory_sampler(
    time_horizon: int,
    env: DynamicalSystem = None,
    policy: BasePolicy = None,
    sample_space: gym.Space = None,
    # init_state = np.array(4),
    # B_initialize_from_init_state = False
):
    """Default trajectory sampler.

    Args:
        env: The system to sample from.
        policy: The policy applied to the system during sampling.
        sample_space: The space where initial conditions are drawn from.

    Returns:
        A generator function that yields system observations as tuples.

    """

    @sample_generator
    def _sample_generator():
        # if not(B_initialize_from_init_state):
        #     state = sample_space.sample()
        # else:
        #     state = init_state.copy()
        # print("sample_space.sample()=",sample_space.sample().shape)
        state = np.array([-0.5, 0.1, -0.5, 0.1])

        state_sequence = []
        action_sequence = []

        env.state = state

        time = 0
        for t in range(time_horizon):
            action = policy(time=time, state=env.state)
            next_state, cost, done, _ = env.step(time=t, action=action)

            state_sequence.append(next_state)
            action_sequence.append(action)

            time += 1

        yield (state, action_sequence, state_sequence)

    return _sample_generator


# def normalize_trajectory_sample(sample):
#     _S = transpose_sample(sample)
#     return zip(*[item / np.sum(np.abs(item)) for item in _S])


ex = Experiment(
    ingredients=[
        system_ingredient,
        simulation_ingredient,
        sample_ingredient,
        plotting_ingredient,
    ]
)


@ex.config
def config(sample):
    """Experiment configuration variables.

    SOCKS uses sacred to run experiments in order to ensure repeatability. Configuration
    variables are parameters that are passed to the experiment, such as the random seed,
    and can be specified at the command-line.

    Example:
        To run the experiment normally, use:

            $ python -m <experiment>

        The full configuration can be viewed using:

            $ python -m <experiment> print_config

        To specify configuration variables, use `with variable=value`, e.g.

            $ python -m <experiment> with seed=123 system.time_horizon=5

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 3  # Kernel bandwidth parameter.
    # Regularization parameter.
    regularization_param = 1e-5

    # Probability of violation.
    delta = 0.1

    time_horizon = 15

    verbose = True

    gen_sample = True

    results_filename = "results/data.npy"
    no_plot = False


@ex.main
def main(
    seed,
    sigma,
    regularization_param,
    delta,
    time_horizon,
    verbose,
    gen_sample,
    results_filename,
    no_plot,
    sample,
    simulation,
    _log,
):
    """Main experiment."""

    # Make the system.
    env = make_system()

    # Set the random seed.
    set_system_seed(seed=seed, env=env)

    if gen_sample is True:
        # Generate the sample.
        _log.debug("Generating sample.")
        sample_space = gym.spaces.Box(
            low=np.array(sample["sample_space"]["lower_bound"], dtype=np.float32),
            high=np.array(sample["sample_space"]["upper_bound"], dtype=np.float32),
            shape=env.state_space.shape,
            dtype=np.float32,
        )
        sample_space.seed(seed=seed)

        # TODO define goal state and pass through
        PD_gains = -1 * np.array([[3, 0.5, 0, 0], [0, 0, 3, 0.5]])
        # ---------------------------------------------------------
        # DEBUGGING CODE
        # dt = 0.1
        # A = np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
        # B = np.array([[dt**2/2,0],[dt,0],[0,dt**2/2],[0,dt]])
        # np.linalg.eig(A+B@K)[0]
        # ---------------------------------------------------------
        ClosedLoopPDPolicy = PDController(
            action_space=env.action_space,
            state_space=env.state_space,
            goal_state=np.array([0.5, 0.0, 0.5, 0]),
            PD_gains=PD_gains,
        )
        S = _sample(
            sampler=cc_trajectory_sampler(
                time_horizon=time_horizon,
                env=env,
                policy=ClosedLoopPDPolicy,
                sample_space=sample_space,
                # init_state = simulation["initial_condition"],
                # B_initialize_from_init_state=True
            ),
            sample_size=sample["sample_space"]["sample_size"],
        )

        # S is a list of tuples
        # S = [(x_0, u, x),
        #      (x_0, u, x),
        #      (x_0, u, x),]
        #   x_0 - np.array (x_dim,)  # vector
        #    u  - np.array (N,u_dim) # matrix (N is time_horizon)
        #    x  - np.array (N,x_dim) # matrix (N is time_horizon)
        # where x[k,:] = f(u[:k,:], x_0, uncertainty) + noise

        # T is a list of tuples
        # T = [(x_0, u, x),
        #      (x_0, u, x),
        #      (x_0, u, x),
        #      (x_0, u, x),]
        #   x_0 - np.array (x_dim,)  # vector
        #    u  - np.array (N*u_dim) # vector (N is time_horizon)
        #    x  - np.array (N*x_dim) # vector (N is time_horizon)

        # -------------------------------------------------
        # DEBUG
        X, U, Y = transpose_sample(S)
        # trajs = np.array(Y)
        # plt.figure(0)
        # for i in range(trajs.shape[0]):
        #     plt.plot(trajs[i, :, 0], trajs[i, :, 2], alpha=0.2)
        # plt.show()
        # -------------------------------------------------

        # Generate the set of admissible control actions.
        _log.debug("Generating admissible control actions.")
        _S = _sample(
            sampler=cc_trajectory_sampler(
                time_horizon=time_horizon,
                env=env,
                policy=ClosedLoopPDPolicy,
                sample_space=sample_space,
            ),
            sample_size=_get_sample_size(env.action_space, sample["action_space"]),
        )

        _T = reshape_trajectory_sample(_S)
        _, A, _ = transpose_sample(_T)

        with open("results/sample.npy", "wb") as f:
            np.save(f, np.array(X))
            np.save(f, np.array(U))
            np.save(f, np.array(Y))
            np.save(f, np.array(A))

    with open("results/sample.npy", "rb") as f:
        X = np.load(f).tolist()
        U = np.load(f).tolist()
        Y = np.load(f).tolist()
        A = np.load(f).tolist()

    S = list(zip(X, U, Y))

    T = reshape_trajectory_sample(S)
    # T = normalize_trajectory_sample(T)
    # Define the cost and constraint functions.

    def _cost_fn(time, state):
        state = np.reshape(state, (-1, time_horizon, 4))
        dist = np.abs(state[:, -1, [0, 2]] - np.array([[0.5, 0.5]]))
        result = np.linalg.norm(dist, ord=2, axis=1)
        return result
        # return np.power(result, 2)

    def _constraint_fn(time, state):
        state = np.reshape(state, (-1, time_horizon, 4))
        dist = state[:, :, [0, 2]] - np.array([[0, 0]])
        result = np.linalg.norm(dist, ord=2, axis=2)
        indicator = np.all(result >= 0.2, axis=1)
        return indicator
        # return 0.2 - np.power(result, 2) - 1 + delta
        # dist >= 0.2
        # 0 >= 0.2 - dist

        # E[f(X, U)] >= 1 - delta

    with ComputationTimer():

        # Compute policy.
        policy = KernelControlCC(
            cost_fn=_cost_fn,
            constraint_fn=_constraint_fn,
            delta=delta,
            verbose=verbose,
            kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
            # kernel_fn=partial(
            #     abel_kernel, sigma=sigma, distance_fn=euclidean_distances
            # ),
            regularization_param=regularization_param,
            seed=seed,
        )

        policy.train(S=T, A=A)

    env.reset()
    env.state = simulation["initial_condition"]
    trajectory = [env.state]

    action_sequence = np.array(policy(state=[env.state]), dtype=np.float32)
    action_sequence = np.reshape(action_sequence, (-1, 2))

    # Simulate the env using the computed policy.
    for t in range(time_horizon):

        obs, *_ = env.step(time=t, action=action_sequence[t])
        next_state = env.state

        trajectory.append(list(next_state))

    with open(results_filename, "wb") as f:
        np.save(f, trajectory)

    if not no_plot:
        plot_results()


@plotting_ingredient.config_hook
def plot_config(config, command_name, logger):
    if command_name in {"main", "plot_results", "plot_sample"}:
        return {
            "trajectory_style": {
                "lines.linewidth": 0.5,
                "lines.linestyle": "-",
                "lines.color": "C1",
            },
            "axes": {
                "xlabel": r"$x_1$",
                "ylabel": r"$x_2$",
                "xlim": (-1.1, 1.1),
                "ylim": (-1.1, 1.1),
            },
        }


@ex.command(unobserved=True)
def plot_results(system, plot_cfg):
    """Plot the results of the experiement."""

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    with open("results/data.npy", "rb") as f:
        trajectory = np.load(f)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    # Plot generated trajectory.
    with plt.style.context(plot_cfg["trajectory_style"]):
        trajectory = np.array(trajectory, dtype=np.float32)
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 2],
            label="System Trajectory",
        )

    plt.gca().add_patch(plt.Circle((0, 0), 0.2, fc="none", ec="red"))

    plt.legend()

    plt.savefig(plot_cfg["plot_filename"])


@ex.command(unobserved=True)
def plot_sample(time_horizon, plot_cfg):

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    with open("results/sample.npy", "rb") as f:
        X = np.load(f).tolist()
        U = np.load(f).tolist()
        Y = np.load(f).tolist()
        A = np.load(f).tolist()

    def _constraint_fn(time, state):
        state = np.reshape(state, (-1, time_horizon, 4))
        dist = state[:, :, [0, 2]]
        result = np.linalg.norm(dist, ord=2, axis=2)
        indicator = np.all(result >= 0.2, axis=1)
        return indicator

    satisfies_constraints = _constraint_fn(time=0, state=Y)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    for i, trajectory in enumerate(Y):

        trajectory = np.array(trajectory, dtype=np.float32)
        if satisfies_constraints[i] == True:
            plt_color = "C0"
        else:
            plt_color = "C1"

        with plt.style.context(plot_cfg["trajectory_style"]):
            plt.plot(trajectory[:, 0], trajectory[:, 2], color=plt_color, marker="")

    plt.gca().add_patch(plt.Circle((0, 0), 0.2, fc="none", ec="red"))

    plt.scatter(0.5, 0.5, s=2.5, c="C2")

    plt.savefig(plot_cfg["plot_filename"])


if __name__ == "__main__":
    ex.run_commandline()