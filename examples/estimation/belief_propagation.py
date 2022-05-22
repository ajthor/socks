from ast import Import
from functools import partial

import numpy as np

from gym.spaces import Box

from gym_socks.algorithms.estimation import kernel_belief_propagation
from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.sampling import sample_fn
from gym_socks.sampling import space_sampler
from gym_socks.sampling.transform import transpose_sample

from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv

from sklearn.metrics.pairwise import linear_kernel

import matplotlib
import matplotlib.pyplot as plt

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
