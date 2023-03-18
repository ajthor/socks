# %% [markdown]
"""
# Target Tracking (Dynamic Programming)

This example demonstrates the kernel-based dynamic programming algorithm. By default, it
uses a nonholonomic vehicle system (unicycle dynamics), and seeks to track a v-shaped
trajectory.

To run the example, use the following command:

```shell
    python examples/control/tracking.py
```

"""

# %%
from functools import partial

import numpy as np

from gym_socks.envs.spaces import Box
from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv
from gym_socks.algorithms.control.kernel_control_bwd import KernelControlBwd

from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.sampling import space_sampler
from gym_socks.sampling import grid_sampler
from gym_socks.sampling import transition_sampler
from gym_socks.sampling.transform import transpose_sample

from gym_socks.utils.grid import boxgrid

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["animation.html"] = "jshtml"

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from scipy.interpolate import griddata

from time import perf_counter

# %% [markdown]
# Configuration variables.

# %%
sigma = 3  # Kernel bandwidth parameter.
regularization_param = 1e-3  # Regularization parameter.

time_horizon = 20

# For controlling randomness.
seed = 12345

# %% [markdown]
# ## Generate the Sample
#
# We generate a random sample from the system, and choose random control actions and
# random initial conditions.

# %%
env = NonholonomicVehicleEnv()
env.sampling_time = 0.1
env.seed(seed)
sample_size = 2000

state_sample_space = Box(
    low=np.array([-1.2, -1.2, -np.pi], dtype=float),
    high=np.array([1.2, 1.2, np.pi], dtype=float),
    shape=(3,),
    dtype=float,
    seed=seed,
)

action_sample_space = Box(
    low=np.array([0.1, -10.1], dtype=float),
    high=np.array([1.1, 10.1], dtype=float),
    shape=(2,),
    dtype=float,
    seed=seed,
)

state_sampler = space_sampler(state_sample_space)
action_sampler = space_sampler(action_sample_space)
S = transition_sampler(env, state_sampler, action_sampler).sample(size=sample_size)

A = boxgrid(action_sample_space, (10, 21))

# %% [markdown]
# We define the cost as the norm distance to the target at each time step.

# %%
a = 0.5  # Path amplitude.
p = 2.0  # Path period.
target_trajectory = [
    [
        (x * 0.1) - 1.0,
        4 * a / p * np.abs((((((x * 0.1) - 1.0) - p / 2) % p) + p) % p - p / 2) - a,
    ]
    for x in range(time_horizon)
]


def _tracking_cost(time: int = 0, state: np.ndarray = None) -> float:
    """Tracking cost function."""

    dist = state[:, :2] - np.array([target_trajectory[time]])
    result = np.linalg.norm(dist, ord=2, axis=1)
    result = np.power(result, 2)
    return result


# %% [markdown]
# ## Algorithm
#
# Now, we can compute the policy using the algorithm, and then simulate the system
# forward in time using the computed policy.

# %%
start = perf_counter()

# Compute the policy.
policy = KernelControlBwd(
    time_horizon=time_horizon,
    cost_fn=_tracking_cost,
    kernel_fn=partial(rbf_kernel, sigma=sigma),
    regularization_param=regularization_param,
)

policy.train(S=S, A=A)

# Simulate the controlled system.
env.reset()
initial_condition = [-0.8, 0, 0]
env.reset(initial_condition)
trajectory = [initial_condition]

for t in range(time_horizon):
    action = policy(time=t, state=env.state)
    state, *_ = env.step(time=t, action=action)

    trajectory.append(list(state))

print(perf_counter() - start)

# %% [markdown]
# ## Value Functions
#
# In order to better visualize what is happening with the kernel-based approximation, we
# plot the value functions at each time step. Note that these are interpolated based on
# the data points we have in the sample `S`.

# %%
*_, Y = transpose_sample(S)
Y = np.array(Y)
grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

fig = plt.figure()
ax = plt.axes()
div = make_axes_locatable(ax)
cax = div.append_axes("right", "5%", "5%")

M = griddata(Y[:, [0, 1]], policy.value_functions[0], (grid_x, grid_y))
im = ax.imshow(M.T, extent=(-1, 1, -1, 1), origin="lower")
cb = fig.colorbar(im, cax=cax)
tx = ax.set_title(f"Time: {0}")


def animate(i):
    cax.cla()
    M = griddata(Y[:, [0, 1]], policy.value_functions[i], (grid_x, grid_y))
    vmin = np.min(M)
    vmax = np.max(M)
    im.set_data(M.T)
    im.set_clim(vmin, vmax)
    tx.set_text(f"Time {i}")


ani = FuncAnimation(fig, animate, frames=time_horizon)

# For saving the gif.
# from matplotlib.animation import PillowWriter
# ani.save("results/value_functions.gif", dpi=300, writer=PillowWriter(fps=2))

ani

# %% [markdown]
# ## Results
#
# We then plot the simulated trajectories of the actual system alongside the predicted
# state trajectory using the approximated dynamics.

# %%
fig = plt.figure()
ax = plt.axes()

target_trajectory = np.array(target_trajectory, dtype=float)
plt.plot(
    target_trajectory[:, 0],
    target_trajectory[:, 1],
    marker="o",
    color="C0",
    label="Target Trajectory",
)

trajectory = np.array(trajectory, dtype=float)
plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    color="C1",
    label="System Trajectory",
)

# Plot the markers as arrows, showing vehicle heading.
paper_airplane = [(0, -0.25), (0.5, -0.5), (0, 1), (-0.5, -0.5), (0, -0.25)]

for x in trajectory:
    angle = -np.rad2deg(x[2])

    t = matplotlib.markers.MarkerStyle(marker=paper_airplane)
    t._transform = t.get_transform().rotate_deg(angle)

    plt.plot(x[0], x[1], marker=t, markersize=15, linestyle="None", color="C1")

plt.legend()
plt.show()
