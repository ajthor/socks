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

from gym.spaces import Box

from gym_socks.algorithms.estimation import kernel_bayes_filter
from gym_socks.kernel.metrics import rbf_kernel

from gym_socks.sampling import sample_fn
from gym_socks.sampling import space_sampler
from gym_socks.sampling.transform import transpose_sample

from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv

from sklearn.metrics.pairwise import linear_kernel

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
        return np.array(state, dtype=np.float32) + 1e-5 * np.array(v)

    def generate_disturbance(self, time, state, action):
        w = self.np_random.standard_normal(size=self.state_space.shape)
        return 1e-3 * np.array(w)


# Define the system.
seed = 12345
env = PartiallyObservableEnv(seed=seed)

# %% [markdown]
# ## Generate a Sample
#
# In order to use the nonparametric filtering algorithm, we require a sample that has
# the true state information. This is a significant drawback, since it requires us to
# have true knowledge of the state in a partially observable system, but is applicable
# when data of the true state is available, e.g. from a few expensive measurements of
# the system or from high-fidelity simulations.

# %%
sample_size = 2000

state_sampler = space_sampler(
    space=Box(
        low=np.array([-0.75, -0.75, -2 * np.pi]),
        high=np.array([0.75, 0.75, 2 * np.pi]),
        shape=env.state_space.shape,
        dtype=env.state_space.dtype,
        seed=seed,
    )
)

action_sampler = space_sampler(
    space=Box(
        low=np.array([0.1, -10]),
        high=np.array([1.1, 10]),
        shape=env.action_space.shape,
        dtype=env.action_space.dtype,
        seed=seed,
    )
)


@sample_fn
def sampler(env, state_sampler, action_sampler):
    state = next(state_sampler)
    action = next(action_sampler)

    env.state = state
    observation, *_ = env.step(action=action)
    next_state = env.state

    yield state, action, next_state, observation


S = sampler(env, state_sampler, action_sampler).sample(size=sample_size)

# %% [markdown]
# ## Fit the Estimator to the Data
#
# We then fit the estimator to the sample data. This computes an implicit estimate of
# the stochastic kernels that describe the state transitions and the observations. In
# MDP literature, these are sometimes called the *state transition kernel* and the
# *observation kernel*.

# %%
sigma = 3
gamma = gamma = 1 / (2 * (sigma ** 2))
kernel_fn = partial(rbf_kernel, sigma=sigma)
regularization_param = 1 / (sample_size ** 2)

# %%
time_horizon = 50

initial_condition = [-0.5, 0, 0]
env.reset(initial_condition)

estimator = kernel_bayes_filter(
    S,
    initial_condition=initial_condition,
    kernel_fn=kernel_fn,
    regularization_param=regularization_param,
)

actual_trajectory = []
estimated_trajectory = []

for t in range(time_horizon):
    action = [0.5, 1]  # Constant forward velocity and turn rate.
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
