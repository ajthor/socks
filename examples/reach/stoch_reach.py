# %% [markdown]
"""
# Stochastic Reachability

This example shows the stochastic reachability algorithm.

By default, the system is a double integrator (2D stochastic chain of integrators).

To run the example, use the following command:

```shell
    python -m examples.reach.stoch_reach
```

"""

# %%
import gym_socks

import numpy as np

from gym.envs.registration import make

from gym_socks.algorithms.reach.kernel_sr import kernel_sr

from gym_socks.sampling import sample
from gym_socks.sampling import sample_generator
from gym_socks.sampling import default_sampler
from gym_socks.sampling import random_sampler

# %% [markdown]
# Configuration variables.

# %%
system_id = "2DIntegratorEnv-v0"
regularization_param = 1e-7
time_horizon = 10

# %%

env = make(system_id)

sample_size = 1000

state_sampler = random_sampler(sample_space=env.state_space)
action_sampler = random_sampler(sample_space=env.action_space)

S = sample(
    sampler=default_sampler(
        state_sampler=state_sampler, action_sampler=action_sampler, env=env
    ),
    sample_size=sample_size,
)
