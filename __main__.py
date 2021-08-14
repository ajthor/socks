# from kernel_basic import Kernel

# if __name__ == '__main__':

import gym

import gym_basic.envs

import numpy as np

from gym_basic.envs.sample import generate_sample
from gym_basic.envs.sample import generate_sample_trajectories

from kernel_basic.kernel import rbf_kernel
from kernel_basic.kernel import regularized_inverse

# env = gym_basic.envs.integrator.NDIntegratorEnv(2)
env = gym_basic.envs.integrator.StochasticNDIntegratorEnv(2)

# env = gym_basic.envs.point_mass.StochasticNDPointMassEnv(2)

# env = gym_basic.envs.cwh.CWH6DEnv()

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
