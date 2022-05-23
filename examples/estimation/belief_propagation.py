from functools import partial

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from gym.spaces import Box

from gym_socks.algorithms.estimation import kernel_belief_propagation
from gym_socks.algorithms.estimation import kernel_bayes_filter
from gym_socks.algorithms.estimation import kernel_bayes_sampler

from gym_socks.envs.point_mass import NDPointMassEnv

from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.sampling import sample_fn
from gym_socks.sampling import space_sampler
from gym_socks.sampling import transition_sampler

from gym_socks.sampling.transform import transpose_sample

from sklearn.metrics.pairwise import linear_kernel

try:
    import networkx as nx
except ImportError as no_networkx:
    raise no_networkx

A = np.array(
    [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

G = nx.DiGraph(A)
G_moral = nx.moral_graph(G)  # Moralize the graph. Changes directed to undirected.

# Plot the graph for demonstration purposes.
fig = plt.figure()
ax = plt.axes()
nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True)
plt.savefig("results/plot_graph.png")

fig = plt.figure()
ax = plt.axes()
nx.draw_networkx(G_moral, pos=nx.spring_layout(G_moral), with_labels=True)
plt.savefig("results/plot_graph.png")

# Generate a sample.
num_nodes = G.number_of_nodes()


class PartiallyObservableEnv(NDPointMassEnv):
    def generate_observation(self, time, state, action):
        v = self.np_random.standard_normal(size=self.observation_space.shape)
        return np.array(state, dtype=np.float32) + 1e-3 * np.array(v)

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-3 * np.array(w)


# We store the agents in a numpy array of `objects`.
agents = [
    PartiallyObservableEnv(2),
    PartiallyObservableEnv(2),
    PartiallyObservableEnv(2),
]

# Size of the sample collected for each agent.
sample_sizes = [
    1000,
    1200,
    800,
]

S = []  # List of samples taken from each agent.
state_estimators = []  # List of Bayes state estimators for each agent.

# Initial state of each agent.
initial_conditions = [
    [-0.5, -0.5],
    [0.25, -0.25],
    [0, 0.5],
]

# Precompute the sample taken from each agent,
for i, agent in enumerate(agents):

    # Define the sampling regions.
    state_sample_space = Box(
        low=-2,
        high=2,
        shape=agents[i].state_space.shape,
        dtype=agents[i].state_space.dtype,
    )

    action_sample_space = Box(
        low=-1,
        high=1,
        shape=agents[i].action_space.shape,
        dtype=agents[i].action_space.dtype,
    )

    # Generate a sample from the agent.
    state_sampler = space_sampler(state_sample_space)
    action_sampler = space_sampler(action_sample_space)

    # The agent sample s is a list of tuples [(x1, u1, y1, z1), ..., (xM, uM, yM, zM)],
    # where xn is the initial state, un is the control action, yn is the resulting state
    # after one time step, and zn is the observation.
    sampler = kernel_bayes_sampler(
        env=agent,
        state_sampler=state_sampler,
        action_sampler=action_sampler,
    )

    S.append(sampler.sample(size=sample_sizes[i]))

    # Reset the agents to initial conditions.
    agent.reset(initial_conditions[i])

    # Define the state estimators used by each agent.
    # Note that the Bayes filters pre-compute many of the matrices we need for BP.
    state_estimators.append(
        kernel_bayes_filter(
            S=S[i],
            initial_condition=initial_conditions[i],
            kernel_fn=partial(rbf_kernel, sigma=0.5),
            regularization_param=1e-5,
        )
    )

# Simulate the multi-agent system.
time_horizon = 20

actual_trajectories = [[] for _ in range(len(agents))]
estimated_trajectories = [[] for _ in range(len(agents))]

# Simulation loop.
for t in range(time_horizon):
    for i, agent in enumerate(agents):

        action = [0.5, 0.5]
        obs, *_ = agent.step(action=action, time=t)

        est_state = state_estimators[i].predict(action=[action], observation=[obs])

        actual_trajectories[i].append(agent.state)
        estimated_trajectories[i].append(est_state)

# Plot the results.
fig = plt.figure()
ax = plt.axes()

for i, agent in enumerate(agents):
    actual_trajectory = np.asarray(actual_trajectories[i])
    estimated_trajectory = np.asarray(estimated_trajectories[i])

    plt.plot(
        actual_trajectory[:, 0],
        actual_trajectory[:, 1],
        marker="x",
        label=f"Actual Agent {i}",
    )
    plt.plot(
        estimated_trajectory[:, 0],
        estimated_trajectory[:, 1],
        marker="o",
        label=f"Estimated Agent {i}",
    )

plt.legend()
plt.savefig("results/plot.png")
