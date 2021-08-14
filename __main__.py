# from kernel_basic import Kernel

# if __name__ == '__main__':

import gym

import gym_basic.envs

import numpy as np

from gym_basic.envs.sample import generate_sample
from gym_basic.envs.sample import generate_sample_trajectories

# env = gym_basic.envs.integrator.NDIntegratorEnv(2)
# env = gym_basic.envs.integrator.StochasticNDIntegratorEnv(2)

# env = gym_basic.envs.point_mass.StochasticNDPointMassEnv(2)

env = gym_basic.envs.cwh.CWH6DEnv()

# sample_space = gym.spaces.Box(
#     low=-0.1,
#     high=0.1,
#     shape=env.observation_space.shape,
#     dtype=np.float32,
# )
#
# S = generate_sample_trajectories(sample_space, env, 5)
#
# print(env.__class__.__mro__)


# env.disturbance_space = gym.spaces.Box(low=0, high=0, shape=(2,), dtype=np.float32)

obs = env.reset()
# env.state = np.array([0.1, 0.1])

for i in range(5):

    # get action
    action = env.action_space.sample()

    # print(obs)

    # apply action
    obs, reward, done, _ = env.step(action)

    if done:
        print(f"Terminated after {i+1} iterations.")
        break

env.close()
