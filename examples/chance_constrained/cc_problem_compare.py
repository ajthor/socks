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

import matplotlib.pyplot as plt

import logging

import numpy as np
from numpy.linalg import norm

from shapely.geometry import Polygon
from shapely.geometry import LineString

from sacred import Experiment
from sacred import Ingredient

from functools import partial
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from gym_socks.kernel.metrics import abel_kernel

from gym_socks.algorithms.control import KernelControlCC

from gym_socks.envs.sample import sample as _sample, transpose_sample

# from gym_socks.envs.sample import trajectory_sampler
from gym_socks.envs.sample import sample_generator
from gym_socks.envs.sample import trajectory_sampler
from gym_socks.envs.sample import reshape_trajectory_sample

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


@system_ingredient.config
def system_config():

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
        "lower_bound": [0, 0, 0, 0],
        "upper_bound": [0, 0, 0, 0],
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


cost_ingredient = Ingredient("cost")


@cost_ingredient.config
def cost_config():

    goal = [10, 0, 10, 0]


@cost_ingredient.capture
def make_cost(time_horizon, goal):

    goal = np.array(goal)

    def _cost_fn(time, state, action):
        action = np.reshape(action, (-1, time_horizon, 2))
        result = np.linalg.norm(action, ord=1, axis=2)
        return np.sum(result, axis=1)

        # state = np.reshape(state, (-1, time_horizon, 4))
        # # dist = state[:, -1, [0, 2]] - np.array([goal[[0, 2]]])
        # dist = state[:, -1, :] - np.array([goal])
        # result = np.linalg.norm(dist, ord=2, axis=1)
        # return result
        # return np.power(result, 2)

    return _cost_fn


obstacle_ingredient = Ingredient("obstacle")


@obstacle_ingredient.config
def obstacle_config():

    center = [0, 0]
    radius = 0.2


constraint_ingredient = Ingredient("constraint", ingredients=[obstacle_ingredient])


@constraint_ingredient.capture
def make_constraint(time_horizon, obstacle):

    center = np.array(obstacle["center"])

    # obstacle1 = Polygon([(3, 2), (8, 2), (8, 7)])
    # obstacle2 = Polygon([(3, 4), (6, 7), (3, 7)])

    O1A = np.array([[-1, 1], [1, 0], [0, -1]])
    O2A = np.array([[0, 1], [-1, 0], [1, -1]])

    O1b = np.array([-1, 8, -2])
    O2b = np.array([7, -3, -1])

    def _constraint_fn(time, state, action):
        state = np.reshape(state, (-1, time_horizon, 4))

        state_shape = np.shape(state)
        indicator = np.zeros((state_shape[0],), dtype=bool)

        # # dist = state[:, :, [0, 2]] - np.array([obstacle["center"]])
        # # result = np.linalg.norm(dist, ord=2, axis=2)
        # # indicator = np.all(result >= obstacle["radius"], axis=1)
        # for i in range(state_shape[0]):
        #     traj = state[i, :, [0, 2]].T
        #     ls = LineString(traj)

        #     # in_obstacle1 = ls.contains(obstacle1)
        #     # in_obstacle2 = ls.contains(obstacle2)

        #     in_obstacle1 = obstacle1.contains(ls)
        #     in_obstacle2 = obstacle2.contains(ls)

        #     # in_obstacle1 = ls.within(obstacle1)
        #     # in_obstacle2 = ls.within(obstacle2)

        #     indicator[i] = ~in_obstacle1 and ~in_obstacle2

        #     print(in_obstacle1)
        #     print(in_obstacle2)
        #     print(~in_obstacle1 and ~in_obstacle2)

        #     break

        #     # indicator[i] = np.random.randint(2)

        # return indicator

        for i in range(state_shape[0]):

            for j in range(time_horizon):

                in_obstacle = True

                for k in range(3):
                    h_ij = -np.array([O1A[k, 0], 0, O1A[k, 1], 0])
                    g_ij = -O1b[k]

                    if h_ij @ state[i, j, :] <= g_ij:
                        in_obstacle = False

                indicator[i] = indicator[i] or in_obstacle

                in_obstacle = True

                for k in range(3):
                    h_ij = -np.array([O2A[k, 0], 0, O2A[k, 1], 0])
                    g_ij = -O2b[k]

                    if h_ij @ state[i, j, :] <= g_ij:
                        in_obstacle = False

                indicator[i] = indicator[i] or in_obstacle

        dist = state[:, -1, [0, 2]] - np.array([10, 10])
        result = np.linalg.norm(dist, ord=2, axis=1)
        is_close = result <= 1.0

        return ~indicator & is_close

    return _constraint_fn


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
    regularization_param = 1e-7

    # Probability of violation.
    delta = 0.01

    time_horizon = 15

    verbose = True

    gen_sample = True
    pd_gains = [[0.5, 2, 0, 0], [0, 0, 0.5, 2]]

    results_filename = "results/data.npy"
    no_plot = False

    mc_validation = True
    num_monte_carlo = 1000


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

    # TODO define goal state and pass through
    PD_gains = np.array(pd_gains)
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
        goal_state=np.array(cost["goal"]),
        PD_gains=PD_gains,
    )
    S = _sample(
        sampler=trajectory_sampler(
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
        sampler=trajectory_sampler(
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

    S = list(zip(X, U, Y))

    T = reshape_trajectory_sample(S)
    # T = normalize_trajectory_sample(T)
    # Define the cost and constraint functions.

    with ComputationTimer():

        # Compute policy.
        policy = KernelControlCC(
            cost_fn=make_cost(time_horizon=time_horizon),
            constraint_fn=make_constraint(time_horizon=time_horizon),
            delta=delta,
            verbose=verbose,
            kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
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

    if mc_validation:
        probability_vector = policy.probability_vector
        plot_mc_validation(seed=seed, env=env, probability_vector=probability_vector)


@ex.capture
def plot_mc_validation(
    seed,
    env,
    probability_vector,
    simulation,
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

    with open("results/sample.npy", "rb") as f:
        X = np.load(f)
        U = np.load(f)
        Y = np.load(f)
        A = np.load(f)

    fig = plt.figure()
    ax = plt.axes(**plot_cfg["axes"])

    cost_fn = make_cost(time_horizon=time_horizon)
    constraint_fn = make_constraint(time_horizon=time_horizon)

    num_violations = 0

    for _ in range(num_monte_carlo):

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
        satisfies_constraints = constraint_fn(
            time=None,
            state=trajectory,
            action=action_sequence,
        )

        trajectory.insert(0, simulation["initial_condition"])
        trajectory = np.array(trajectory, dtype=np.float32)
        if satisfies_constraints[0] == True:
            plt_color = "C0"
        else:
            plt_color = "C1"

        # Plot the trajectory.
        with plt.style.context(plot_cfg["trajectory_style"]):
            plt.plot(trajectory[:, 0], trajectory[:, 2], color=plt_color, marker="")

        num_violations += ~satisfies_constraints[0]

    print(
        f"% violation: {num_violations / num_monte_carlo} [{num_violations}/{num_monte_carlo}]"
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

    obstacles_vertices = [np.array([[3, 2], [8, 2], [8, 7]])]
    obstacles_vertices.append(np.array([[3, 4], [6, 7], [3, 7]]))

    for obs_verts in obstacles_vertices:
        plt.gca().add_patch(plt.Polygon(obs_verts, fc="black", ec="none"))

    plt.scatter(cost["goal"][0], cost["goal"][2], s=2.5, c="C2", label="Goal State")

    plt.savefig("results/plot_mc.png")


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
                "xlabel": r"$x_1$",
                "ylabel": r"$x_2$",
                "xlim": (-0.5, 10.5),
                "ylim": (-0.5, 10.5),
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

    obstacles_vertices = [np.array([[3, 2], [8, 2], [8, 7]])]
    obstacles_vertices.append(np.array([[3, 4], [6, 7], [3, 7]]))

    for obs_verts in obstacles_vertices:
        plt.gca().add_patch(plt.Polygon(obs_verts, fc="black", ec="none"))

    plt.scatter(cost["goal"][0], cost["goal"][2], s=2.5, c="C2", label="Goal State")

    plt.legend()

    plt.savefig(plot_cfg["plot_filename"])


@ex.command(unobserved=True)
def plot_sample(seed, time_horizon, cost, obstacle, plot_cfg):

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

    constraint_fn = make_constraint(time_horizon=time_horizon)
    satisfies_constraints = constraint_fn(time=0, state=Y, action=U)

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

    obstacles_vertices = [np.array([[3, 2], [8, 2], [8, 7]])]
    obstacles_vertices.append(np.array([[3, 4], [6, 7], [3, 7]]))

    for obs_verts in obstacles_vertices:
        plt.gca().add_patch(plt.Polygon(obs_verts, fc="black", ec="none"))

    plt.scatter(cost["goal"][0], cost["goal"][2], s=2.5, c="C2", label="Goal State")

    plt.savefig(plot_cfg["plot_filename"])


if __name__ == "__main__":
    ex.run_commandline()
