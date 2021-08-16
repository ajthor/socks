import gym

import systems.envs

import numpy as np

from systems.sample import generate_sample
from systems.sample import generate_uniform_sample
from systems.sample import generate_sample_trajectories

from kernel.metrics import rbf_kernel
from kernel.metrics import regularized_inverse

env = systems.envs.integrator.NDIntegratorEnv(2)
# env = systems.envs.integrator.StochasticNDIntegratorEnv(2)
#
# # env = systems.envs.point_mass.StochasticNDPointMassEnv(2)
#
# # env = systems.envs.cwh.CWH6DEnv()
#
# env.seed(0)

sample_space = gym.spaces.Box(
    low=-0.1,
    high=0.1,
    shape=env.observation_space.shape,
    dtype=np.float32,
)

S, x = generate_uniform_sample(sample_space, env, [21, 21])

print(S)

# K = rbf_kernel(S[:,0,:])
# W = regularized_inverse(S[:,0,:])
#
# print(K)
# print(W)
#
# # print(env.__class__.__mro__)

# env = systems.envs.temperature.StochasticTemperatureRegEnv(3)
#
# # print(env.observation_space.sample())
# # print(env.action_space.sample())
# #
# # env.reset()
# #
# # print(env.state)
#
# obs = env.reset()
#
# for i in range(20):
#
#     print(obs)
#
#     # get action
#     # action = env.action_space.sample()
#
#     # t = tuple(0 for i in range(env.dim - 1))
#     # action = (1, *t)
#
#     action = tuple(0 for i in range(env.dim))
#
#     # apply action
#     obs, reward, done, _ = env.step(action)
#
#     if done:
#         break
#
# env.close()
