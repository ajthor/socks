
import gym

import systems.envs

import numpy as np

from systems.sample import generate_sample
from systems.sample import generate_sample_trajectories

from kernel.metrics import rbf_kernel
from kernel.metrics import regularized_inverse

# env = systems.envs.integrator.NDIntegratorEnv(2)
env = systems.envs.integrator.StochasticNDIntegratorEnv(2)

# env = systems.envs.point_mass.StochasticNDPointMassEnv(2)

# env = systems.envs.cwh.CWH6DEnv()

env.seed(0)

sample_space = gym.spaces.Box(
    low=-0.1,
    high=0.1,
    shape=env.observation_space.shape,
    dtype=np.float32,
)

S = generate_sample(sample_space, env, 5)

K = rbf_kernel(S[:,0,:])
W = regularized_inverse(S[:,0,:])

print(K)
print(W)

# print(env.__class__.__mro__)
