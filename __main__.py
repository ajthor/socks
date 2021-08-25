import gym

import systems.envs

import numpy as np

from systems.sample import generate_sample
from systems.sample import generate_uniform_sample
from systems.sample import generate_sample_trajectories

from kernel.metrics import rbf_kernel
from kernel.metrics import regularized_inverse

# env = systems.envs.integrator.NDIntegratorEnv(2)
# # env = systems.envs.integrator.StochasticNDIntegratorEnv(2)
# #
# # # env = systems.envs.point_mass.StochasticNDPointMassEnv(2)
# #
# # env.seed(0)
#
# sample_space = gym.spaces.Box(
#     low=-0.1,
#     high=0.1,
#     shape=env.observation_space.shape,
#     dtype=np.float32,
# )
#
# S, U = generate_sample(sample_space, env, 5)

# print(S)


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


# env = systems.envs.cwh.CWH4DEnv()
#
# print(env.state_matrix)

# Monte-Carlo
# Control
# Do that one paragrah on scalability.


# np.random.seed(0)
#
# def get_pt():
#     return np.random.rand(2), np.random.rand(1)
#
# S = [get_pt() for i in range(10)]
#
# Sx, Sy = zip(*[get_pt() for i in range(10)])
#
# # Sx, Sy = [(x, y) for x, y in [get_pt() for i in range(10)]]
#
# print(S)
#
# print(Sx)
# print(Sy)
