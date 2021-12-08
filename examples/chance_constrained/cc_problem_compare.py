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

Todo:
    * Matrix of kernel parameter vs. sample size.
    * Update to Mixed paper obstacles.
    * Validate against mixed paper soln.

"""


import gym
import gym_socks

import logging

import numpy as np
from numpy.linalg import norm

from sacred import Experiment
from sacred import Ingredient

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, laplacian_kernel
from gym_socks.kernel.metrics import abel_kernel

from gym_socks.algorithms.control import KernelControlCC

from gym_socks.envs.sample import sample as _sample, transpose_sample, uniform_grid

from gym_socks.envs.sample import sample_generator
from gym_socks.envs.sample import trajectory_sampler
from gym_socks.envs.sample import reshape_trajectory_sample

from gym_socks.envs.policy import (
    RandomizedPolicy,
    BasePolicy,
    PDController,
    ConstantPresampledPolicy,
)

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

from examples.chance_constrained.cost_ingredient import cost_ingredient
from examples.chance_constrained.cost_ingredient import make_cost_x
from examples.chance_constrained.cost_ingredient import make_cost_u

from examples.chance_constrained.constraint_ingredient import constraint_ingredient
from examples.chance_constrained.constraint_ingredient import make_constraint_x
from examples.chance_constrained.constraint_ingredient import make_constraint_u

from examples.chance_constrained.cc_kernel import cc_kernel

import matplotlib.pyplot as plt


@system_ingredient.config
def system_config():

    # system_id = "RepeatedIntegratorEnv-v0"
    system_id = "NonMarkovIntegratorEnv-v0"

    sampling_time = 1.0

    action_space = {
        "lower_bound": [-1.1, -1.1],
        "upper_bound": [1.1, 1.1],
    }


@sample_ingredient.config
def sample_config():

    sample_space = {
        "sample_scheme": "uniform",
        # "lower_bound": [-0, -0, -0, -0],
        # # "upper_bound": [0, 0, 0, 0],
        "lower_bound": [-0.5, -0.05, -0.5, -0.05],
        "upper_bound": [0.5, 0.05, 0.5, 0.05],
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

    initial_condition = [0, 0, 0, 0]


ex = Experiment(
    ingredients=[
        system_ingredient,
        simulation_ingredient,
        sample_ingredient,
        plotting_ingredient,
        cost_ingredient,
        constraint_ingredient,
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

            $ python -m <experiment> with seed=123 time_horizon=5

    .. _sacred:
        https://sacred.readthedocs.io/en/stable/index.html

    """

    sigma = 5  # Kernel bandwidth parameter.
    # Regularization parameter.
    regularization_param = 1e-5

    # Probability of violation.
    delta = 0.01

    time_horizon = 15

    verbose = True

    gen_sample = True
    pd_gains = [[1, 2.25, 0, 0], [0, 0, 1, 2.25]]

    results_filename = "results/data.npy"
    no_plot = False

    mc_validation = True
    num_monte_carlo = 1000


# def _admissible_sampler(env, time_horizon, sample_space, sample, pd_gains, goal_state):

#     A = env.state_matrix
#     B = env.input_matrix

#     ux = np.linspace(0, 1, 10)  # sample["action_space"]["grid_resolution"][0])
#     uy = np.linspace(0, 1, 10)  # sample["action_space"]["grid_resolution"][1])

#     ui = uniform_grid([ux, uy])

#     # def get_next_action():
#     #     for i in range(len(ui)):
#     #         yield ui[i]

#     # def action_generator():
#     #     while True:
#     #         yield from get_next_action()

#     def policy(time, state, *args, **kwargs):
#         if time == 0:
#             action = env.action_space.sample()
#         else:
#             action = -pd_gains @ (state - goal_state)
#             action = np.clip(action, -1, 1)

#         return action

#     @sample_generator
#     def _sample_generator():
#         # env.reset()
#         initial_state = sample_space.sample()

#         state = initial_state

#         state_sequence = []
#         action_sequence = []

#         # env.state = state

#         # for item in ui:

#         time = 0
#         for t in range(time_horizon):

#             # if t == 0:
#             #     action = list(item)
#             # else:
#             # action = -pd_gains @ (state - goal_state)
#             # action = np.clip(action, -1, 1)
#             action = policy(time=t, state=state)

#             next_state = np.matmul(A, state) + np.matmul(B, action)

#             state_sequence.append(next_state)
#             action_sequence.append(action)

#             state = next_state

#             time += 1

#         yield (initial_state, action_sequence, state_sequence)

#     return _sample_generator


# @ex.capture
# def gen_admissible_control_sequences(seed, env, time_horizon, sample):
#     pass


@ex.capture
def generate_cc_sample(seed, env, time_horizon, cost, pd_gains, sample, _log):
    # Generate the sample.
    _log.debug("Generating sample.")
    sample_space = gym.spaces.Box(
        low=np.array(sample["sample_space"]["lower_bound"], dtype=np.float32),
        high=np.array(sample["sample_space"]["upper_bound"], dtype=np.float32),
        shape=env.state_space.shape,
        dtype=np.float32,
    )
    sample_space.seed(seed=seed)

    # # PD controller
    PD_gains = np.array(pd_gains)
    ClosedLoopPDPolicy = PDController(
        action_space=env.action_space,
        state_space=env.state_space,
        goal_state=np.array(cost["goal"]),
        PD_gains=PD_gains,
    )

    # Sample controls
    _log.debug("Sampling controls for dataset to approximate Q.")
    _S = _sample(
        sampler=trajectory_sampler(
            time_horizon=time_horizon,
            env=env,
            policy=ClosedLoopPDPolicy,
            sample_space=sample_space,
        ),
        sample_size=sample["sample_space"]["sample_size"],
        # _get_sample_size(env.action_space, sample["action_space"]),
    )
    _T = reshape_trajectory_sample(_S)
    _, U, _ = transpose_sample(_T)

    # sample trajectories
    print("U=", U[0].shape)
    _log.debug("Sampling trajectories for dataset to approximate Q.")
    S = _sample(
        sampler=trajectory_sampler(
            time_horizon=time_horizon,
            env=env,
            # policy=ClosedLoopPDPolicy,
            policy=ConstantPresampledPolicy(controls=U, action_space=env.action_space),
            sample_space=sample_space,
        ),
        sample_size=sample["sample_space"]["sample_size"],
    )
    X, U, Y = transpose_sample(S)

    # # # -------------------------------------------------
    # # # DEBUG
    # trajs = np.array(Y)
    # plt.figure(0)
    # for i in range(trajs.shape[0]):
    #     plt.plot(trajs[i, :, 0], trajs[i, :, 2], alpha=0.2)
    # # plt.show()

    # plt.savefig("results/plot_sample.png")
    # # -------------------------------------------------

    # ---------------
    def sample_controls(action_space, M, T, x0=np.zeros(4), xg=np.zeros(4), u_dim=2):
        dt = 1.0
        A = np.array(
            [
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        B = np.array(
            [
                [(dt ** 2) / 2, 0],
                [dt, 0],
                [0, (dt ** 2) / 2],
                [0, dt],
            ],
            dtype=np.float32,
        )

        print("(M, u_dim, T)=", (M, u_dim, T))
        us = np.zeros((M, u_dim, T))
        xs = np.zeros((M, x0.shape[0], T + 1))
        xs[:, :, 0] = np.repeat(x0[np.newaxis, :], M, axis=0)
        # print(f"xs shape: {np.shape(xs)}")
        for t in range(T):
            if t <= 2:
                M_us = int(np.sqrt(M)) ** 2
                ux = np.linspace(0, 1, int(np.sqrt(M)))
                uy = np.linspace(0, 1, int(np.sqrt(M)))
                usk = np.meshgrid(ux, uy)
                usk = np.reshape(np.meshgrid(ux, uy), (2, -1)).T

                xs[:M_us, :, t + 1] = (A @ xs[:M_us, :, t].T + B @ usk.T).T
                us[:M_us, :, t] = usk
            if t > 2:
                dxk = xs[:, :, t] - np.repeat(xg[np.newaxis, :], M, axis=0)
                usk = -PD_gains @ dxk.T

                for i in range(M):
                    usk[:, i] = np.clip(usk[:, i], action_space.low, action_space.high)

                xs[:, :, t + 1] = (A @ xs[:, :, t].T + B @ usk).T
                us[:, :, t] = usk.T

            for i in range(M):
                alpha = 0.005
                drag_vector = -alpha * np.array(
                    [
                        (dt ** 2) * np.abs(xs[i, :, t][1]) * xs[i, :, t][1] / 2,
                        dt * np.abs(xs[i, :, t][1]) * xs[i, :, t][1],
                        (dt ** 2) * np.abs(xs[i, :, t][3]) * xs[i, :, t][3] / 2,
                        dt * np.abs(xs[i, :, t][3]) * xs[i, :, t][3],
                    ],
                    dtype=np.float32,
                )
                xs[i, :, t + 1] += drag_vector

        us = np.full(
            us.shape,
            us,
            dtype=action_space.dtype,
        )
        for t in range(T):
            for i in range(M):
                us[i, :, t] = np.clip(us[i, :, t], action_space.low, action_space.high)
        us_list = []
        for i in range(M):
            us_list.append(np.reshape(us[i, :, :].T, u_dim * T))
        return xs, us_list

    x0 = np.array([0, 0, 0, 0])
    xs, us = sample_controls(
        env.action_space,
        _get_sample_size(env.action_space, sample["action_space"]),
        time_horizon,
        x0,  # simulation["initial_condition"],
        np.array(cost["goal"]),
        env.action_space.shape[0],
    )
    print("us=", us[0].shape)

    # # -------------------------------------------------
    # # DEBUG
    ###-------------
    plt.figure(0)
    for i in range(xs.shape[0]):
        plt.plot(xs[i, 0, :], xs[i, 2, :], alpha=0.2)
    plt.show()
    ###-------------

    plt.savefig("results/plot_sample.png")
    # -------------------------------------------------

    # # sample trajectories
    # # print("U=", U[0].shape)
    # _log.debug("Sampling trajectories for dataset to approximate Q.")
    # S = _sample(
    #     sampler=trajectory_sampler(
    #         time_horizon=time_horizon,
    #         env=env,
    #         # policy=ClosedLoopPDPolicy,
    #         policy=ConstantPresampledPolicy(controls=us, action_space=env.action_space),
    #         sample_space=sample_space,
    #     ),
    #     sample_size=sample["sample_space"]["sample_size"],
    # )
    # X, U, Y = transpose_sample(S)

    ###-------------
    # Generate the set of admissible control actions.
    _log.debug("Generating admissible control actions (dataset A).")
    _S = _sample(
        sampler=trajectory_sampler(
            time_horizon=time_horizon,
            env=env,
            # policy=ClosedLoopPDPolicy,
            policy=ConstantPresampledPolicy(controls=us, action_space=env.action_space),
            sample_space=sample_space,
        ),
        sample_size=_get_sample_size(env.action_space, sample["action_space"]),
    )

    # _S = _sample(
    #     sampler=_admissible_sampler(
    #         env=env,
    #         time_horizon=time_horizon,
    #         sample_space=sample_space,
    #         sample=sample,
    #         pd_gains=np.array(pd_gains),
    #         goal_state=np.array(cost["goal"]),
    #     ),
    #     sample_size=_get_sample_size(env.action_space, sample["action_space"]),
    # )

    _, _, YA = transpose_sample(_S)
    _T = reshape_trajectory_sample(_S)
    _, A, _ = transpose_sample(_T)

    print(f"X shape {np.shape(X)}")
    print(f"U shape {np.shape(U)}")
    print(f"Y shape {np.shape(Y)}")
    print(f"A shape {np.shape(A)}")
    print(f"YA shape {np.shape(YA)}")

    with open("results/sample.npy", "wb") as f:
        np.save(f, np.array(X))
        np.save(f, np.array(U))
        np.save(f, np.array(Y))
        np.save(f, np.array(A))

    with open("results/test_sample.npy", "wb") as f:
        np.save(f, np.array(YA))


@ex.main
def main(
    seed,
    sigma,
    regularization_param,
    delta,
    time_horizon,
    verbose,
    gen_sample,
    cost,
    pd_gains,
    results_filename,
    no_plot,
    mc_validation,
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
        generate_cc_sample(seed=seed, env=env)

    with open("results/sample.npy", "rb") as f:
        X = np.load(f).tolist()
        U = np.load(f).tolist()
        Y = np.load(f).tolist()
        A = np.load(f).tolist()

    Y = np.array(Y)
    print(f"max v {Y[:, :, [1, 3]].max()}")
    print(f"min v {Y[:, :, [1, 3]].min()}")

    S = list(zip(X, U, Y))

    T = reshape_trajectory_sample(S)
    # T = S

    # Define the cost and constraint functions.
    cost_x = make_cost_x(time_horizon=time_horizon)
    cost_u = make_cost_u(time_horizon=time_horizon)

    constraint_x = make_constraint_x(time_horizon=time_horizon)
    constraint_u = make_constraint_u(time_horizon=time_horizon)

    with ComputationTimer():

        # Compute policy.
        policy = KernelControlCC(
            cost_fn_x=cost_x,
            cost_fn_u=cost_u,
            constraint_fn_x=constraint_x,
            constraint_fn_u=constraint_u,
            delta=delta,
            verbose=verbose,
            # kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (0.01 ** 2))),
            kernel_fn_x=rbf_kernel,
            kernel_fn_u=rbf_kernel,
            # kernel_fn_u=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
            # kernel_fn=partial(cc_kernel, gamma=1 / (2 * (sigma ** 2))),
            # kernel_fn=partial(abel_kernel, sigma=1 / (2 * (sigma ** 2))),
            # kernel_fn=partial(laplacian_kernel, gamma=1 / (2 * (sigma ** 2))),
            regularization_param=regularization_param,
            # regularization_param=1,
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

    if mc_validation:
        probability_vector = policy.probability_vector
        plot_mc_validation(seed=seed, env=env, probability_vector=probability_vector)


@plotting_ingredient.config_hook
def plot_config(config, command_name, logger):
    if command_name in {"main", "plot_results", "plot_sample", "plot_mc_validation"}:
        return {
            "trajectory_style": {
                "lines.linewidth": 0.5,
                "lines.linestyle": "-",
                "lines.color": "C1",
            },
            "axes": {
                "xlabel": r"$p_x$",
                "ylabel": r"$p_y$",
                "xlim": (-0.5, 12.5),
                "ylim": (-0.5, 12.5),
            },
        }


@ex.command(unobserved=True)
def plot_results(system, cost, obstacle, plot_cfg):
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

    # plt.gca().add_patch(
    #     plt.Circle(
    #         obstacle["center"],
    #         obstacle["radius"],
    #         fc="none",
    #         ec="red",
    #         label="Obstacle",
    #     )
    # )

    obstacles_vertices = [np.array([[2.7, 2], [8, 2], [8, 7.3]])]
    obstacles_vertices.append(np.array([[3, 3.7], [6.3, 7], [3, 7]]))

    for obs_verts in obstacles_vertices:
        plt.gca().add_patch(plt.Polygon(obs_verts, fc="black", ec="none"))

    x_goal = (10, 10)
    eps_goal = 2.5  # todo make this a variable
    # goal_vertices = np.array([[8.5,8.5],[8.5,11.5],[11.5,11.5],[11.5,8.5]])
    # plt.gca().add_patch(plt.Polygon(goal_vertices, fc="black", ec="none"))
    plt.gca().add_patch(
        plt.Circle(
            x_goal,
            eps_goal,
            fc="black",
            ec="none",
            alpha=0.2,
            label=r"$\mathcal{X}_{goal}$",
        )
    )

    plt.scatter(cost["goal"][0], cost["goal"][2], s=2.5, c="C2", label="Goal State")

    plt.legend()

    plt.savefig(plot_cfg["plot_filename"])


@ex.command(unobserved=True)
def plot_sample(seed, time_horizon, delta, cost, obstacle, plot_cfg):

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt

    # Make the system.
    env = make_system()

    # Set the random seed.
    set_system_seed(seed=seed, env=env)

    S = generate_cc_sample(seed=seed, env=env)

    with open("results/sample.npy", "rb") as f:
        X = np.load(f).tolist()
        U = np.load(f).tolist()
        Y = np.load(f).tolist()
        A = np.load(f).tolist()

    with open("results/test_sample.npy", "rb") as f:
        YA = np.load(f).tolist()

    constraint_fn = make_constraint_x(time_horizon=time_horizon)
    satisfies_constraints = constraint_fn(time=0, state=Y)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    for i, trajectory in enumerate(Y):

        trajectory = np.array(trajectory, dtype=np.float32)

        if satisfies_constraints[i] == True:
            plt_color = "C0"
        else:
            plt_color = "C1"

        # if i <= 10:
        with plt.style.context(plot_cfg["trajectory_style"]):
            plt.plot(trajectory[:, 0], trajectory[:, 2], color=plt_color, marker="")

        # plt.gca().add_patch(
        #     plt.Circle(
        #         obstacle["center"],
        #         obstacle["radius"],
        #         fc="none",
        #         ec="red",
        #         label="Obstacle",
        #     )
        # )

    for i, trajectory in enumerate(Y):
        trajectory = np.array(trajectory)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 2], s=3, color="red")

    obstacles_vertices = [np.array([[2.7, 2], [8, 2], [8, 7.3]])]
    obstacles_vertices.append(np.array([[3, 3.7], [6.3, 7], [3, 7]]))

    for obs_verts in obstacles_vertices:
        plt.gca().add_patch(plt.Polygon(obs_verts, fc="black", ec="none"))

    plt.scatter(cost["goal"][0], cost["goal"][2], s=2.5, c="C2", label="Goal State")

    plt.savefig(plot_cfg["plot_filename"])


@ex.capture
def plot_mc_validation(
    seed,
    env,
    probability_vector,
    simulation,
    delta,
    time_horizon,
    num_monte_carlo,
    cost,
    obstacle,
    plot_cfg,
):

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Dynamically load for speed.
    import matplotlib

    matplotlib.use("Agg")
    update_rc_params(matplotlib, plot_cfg["rc_params"])

    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib import rcParams

    # rc('text', usetex=True)
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 16
    rcParams["text.usetex"] = True

    with open("results/sample.npy", "rb") as f:
        X = np.load(f)
        U = np.load(f)
        Y = np.load(f)
        A = np.load(f)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    # cost_fn = make_cost(time_horizon=time_horizon)
    constraint_fn = make_constraint_x(time_horizon=time_horizon)

    num_violations = 0

    for _ in range(max(num_monte_carlo, 1000)):

        env.reset()
        # env.mass = 1
        # env.alpha = 0.05
        env.state = simulation["initial_condition"]
        # trajectory = [env.state]
        trajectory = []

        idx = np.random.choice(len(probability_vector), size=None, p=probability_vector)
        action_sequence = np.array(A[idx], dtype=np.float32)
        action_sequence = np.reshape(action_sequence, (-1, 2))

        # Simulate the env using the computed policy.
        for t in range(time_horizon):

            obs, *_ = env.step(time=t, action=action_sequence[t])
            next_state = env.state

            trajectory.append(list(next_state))

        # Compute whether the simulated trajectory satisfies constraints.
        satisfies_constraints = constraint_fn(time=None, state=trajectory)

        trajectory.insert(0, simulation["initial_condition"])
        trajectory = np.array(trajectory, dtype=np.float32)
        if satisfies_constraints[0] == True:
            plt_color = (0, 0.3, 0.8)  # "blue"
            plt_alpha = 0.05
        else:
            plt_color = (0.9, 0, 0)  # "red"
            plt_alpha = 0.2

        # Plot the trajectory.
        with plt.style.context(plot_cfg["trajectory_style"]):
            plt.plot(
                trajectory[:, 0],
                trajectory[:, 2],
                color=plt_color,
                marker="",
                alpha=plt_alpha,
            )

        num_violations += ~satisfies_constraints[0]

    print(
        f"% violation: {num_violations / num_monte_carlo} [{num_violations}/{num_monte_carlo}]"
    )

    obstacles_vertices = [np.array([[2.7, 2], [8, 2], [8, 7.3]])]
    obstacles_vertices.append(np.array([[3, 3.7], [6.3, 7], [3, 7]]))

    for obs_verts in obstacles_vertices:
        plt.gca().add_patch(plt.Polygon(obs_verts, fc="red", ec="none", alpha=0.4))
    plt.text(3.5, 5.5, r"$\mathcal{O}$", color=(0.7, 0, 0))

    x_init = simulation["initial_condition"]
    plt.scatter(x_init[0], x_init[1], s=30, c="k", marker="+")
    plt.text(0.5, 0, r"$x_0$")

    x_goal = (cost["goal"][0], cost["goal"][2])
    eps_goal = 2.5  # todo make this a variable
    plt.gca().add_patch(plt.Circle(x_goal, eps_goal, fc="black", ec="none", alpha=0.2))
    plt.text(x_goal[0] - 1, x_goal[1] - 3.5, r"$\mathcal{X}_{goal}$")
    # plt.text(7.5, 0, r"$\delta = {{{delta}}}$".format(delta=delta))

    plt.xlabel(r"$p_x$")
    plt.ylabel(r"$p_y$").set_rotation(0)

    plt.savefig("results/plot_mc.png")


if __name__ == "__main__":
    ex.run_commandline()
