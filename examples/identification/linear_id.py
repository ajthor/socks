# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,.pct.py:percent,.lgt.py:light,.spx.py:sphinx,md,Rmd,.pandoc.md:pandoc
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Linear System Identification

This file demonstrates the linear system identification algorithm.

By default, it uses the CWH4D system dynamics. Try setting the regularization
parameter lower for higher accuracy. Note that this can introduce numerical
instability if set too low.

To run the example, use the following command:

```shell
    python -m examples.identification.linear_id
```

"""

# %%
import gym_socks

import numpy as np

from gym.envs.registration import make

from gym_socks.algorithms.identification.kernel_linear_id import kernel_linear_id

from gym_socks.policies import ConstantPolicy
from gym_socks.sampling import sample
from gym_socks.sampling import sample_generator
from gym_socks.sampling import random_sampler

# %% [markdown]
# Configuration variables.

# %%
system_id = "CWH4DEnv-v0"
regularization_param = 1e-9
time_horizon = 1000

# %% [markdown]
# Generate the system sample.

# %%
env = make(system_id)

sample_size = 10

state_sampler = random_sampler(sample_space=env.state_space)
policy = ConstantPolicy(action_space=env.action_space, constant=-1.0)


@sample_generator
def sampler():
    state = next(state_sampler)
    action = policy()

    env.state = state
    next_state, *_ = env.step(action=action)

    yield (state, action, next_state)


S = sample(sampler=sampler, sample_size=sample_size)


# %% [markdown]
# Run the algorithm.

# %%
alg = kernel_linear_id(S=S, regularization_param=regularization_param)

# %% [markdown]
# Validate the output.

# %%
simulation_policy = ConstantPolicy(action_space=env.action_space, constant=[0.01, 0.01])

env.reset()

# Set a specific initial condition for simulation.
initial_condition = [-0.75, -0.75, 0, 0]

# Simulate the system using the actual dynamics.
env.state = initial_condition
actual_trajectory = [env.state]
for t in range(time_horizon):
    action = np.array(policy(time=t, state=[env.state]), dtype=np.float32)
    obs, *_ = env.step(time=t, action=action)
    next_state = env.state

    actual_trajectory.append(list(next_state))

# Simulate the system using the approximated dynamics.
estimated_trajectory = [initial_condition]
for t in range(time_horizon):
    action = policy(time=t, state=[env.state])
    state = alg.predict(T=estimated_trajectory[t], U=action)

    estimated_trajectory.append(state)

# %% [markdown]
# Plot the results.

# %%
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes()

actual_trajectory = np.array(actual_trajectory, dtype=np.float32)
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], label="Actual Trajectory")

estimated_trajectory = np.array(estimated_trajectory, dtype=np.float32)
plt.plot(
    estimated_trajectory[:, 0], estimated_trajectory[:, 1], label="Estimated Trajectory"
)

plt.legend()

plt.show()
