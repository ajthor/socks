# %% [markdown]
"""
# State estimation example.

This example demonstrates nonparametric state estimation for partially observable
dynamical system models.

To run the example, use the following command:

```shell
    python examples/estimation/bayes_filter.py
```

"""

# %%
from functools import partial

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel

from gym.spaces import Box

from gym_socks.algorithms.estimation import kernel_bayes_filter

from gym_socks.sampling import sample_generator
from gym_socks.sampling import random_sampler
from gym_socks.sampling import sample

from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv

# %% [markdown]
# ## Define the Partially Observable System
#
# The environments in `gym_socks` are fully observable by default. In order to make them
# partially observable, we override the `generate_observation` function and incorporate
# a Gaussian additive noise.

# %%
class PartiallyObservableEnv(NonholonomicVehicleEnv):
    def generate_observation(self, time, state, action):
        v = self.np_random.standard_normal(size=self.observation_space.shape)
        return np.array(state, dtype=np.float32) + 1e-3 * np.array(v)


# Define the system.
env = PartiallyObservableEnv()

# %% [markdown]
# ## Generate a Sample
#
# In order to use the nonparametric filtering algorithm, we require a sample that has
# the true state information. This is a significant drawback, since it requires us to
# have true knowledge of the state in a partially observable system, but is applicable
# when data of the true state is available, e.g. from a few expensive measurements of
# the system or from high-fidelity simulations.

# %%
sample_size = 1500

state_sampler = random_sampler(
    sample_space=Box(
        low=np.array([-1.2, -1.2, -2 * np.pi]),
        high=np.array([1.2, 1.2, 2 * np.pi]),
        shape=env.state_space.shape,
        dtype=env.state_space.dtype,
    )
)

action_sampler = random_sampler(
    sample_space=Box(
        low=np.array([0.1, -10]),
        high=np.array([1.1, 10]),
        shape=env.action_space.shape,
        dtype=env.action_space.dtype,
    )
)


@sample_generator
def sampler():
    state = next(state_sampler)
    action = next(action_sampler)

    env.state = state
    observation, *_ = env.step(action=action)
    next_state = env.state

    yield (state, action, next_state, observation)


S = sample(sampler=sampler, sample_size=sample_size)

# %% [markdown]
# ## Fit the Estimator to the Data
#
# We then fit the estimator to the sample data. This computes an implicit estimate of
# the stochastic kernels that describe the state transitions and the observations. In
# MDP literature, these are sometimes called the *state transition kernel* and the
# *observation kernel*.

# %%
sigma = 3
kernel_fn = partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2)))
regularization_param = 1 / (sample_size ** 2)

estimator = kernel_bayes_filter(
    S, kernel_fn=kernel_fn, regularization_param=regularization_param
)

# %%
time_horizon = 20

initial_condition = [-0.8, -0.8, np.pi / 4]
env.state = initial_condition

actual_trajectory = []
estimated_trajectory = []

for t in range(time_horizon):
    action = [0.75, 0]
    obs, *_ = env.step(action=action, time=t)

    est_state = estimator.predict(action=[action], observation=[obs])

    actual_trajectory.append(env.state)
    estimated_trajectory.append(est_state)

# %% [markdown]
# ## Results
#
# We then plot the simulated trajectories of the actual system alongside the predicted
# state trajectory.

# %%
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes()

actual_trajectory = np.array(actual_trajectory)
estimated_trajectory = np.array(estimated_trajectory)

print(actual_trajectory)
print(estimated_trajectory)

actual_trajectory = np.array(actual_trajectory, dtype=np.float32)
plt.plot(
    actual_trajectory[:, 0],
    actual_trajectory[:, 1],
    label="Actual Trajectory",
)

estimated_trajectory = np.array(estimated_trajectory, dtype=np.float32)
plt.plot(
    estimated_trajectory[:, 0],
    estimated_trajectory[:, 1],
    label="Estimated Trajectory",
)

plt.legend()

plt.savefig("results/plot.png")
