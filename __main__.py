# from kernel_basic import Kernel

# if __name__ == '__main__':

import gym

import gym_basic.envs

import numpy as np

# get environment
# env = gym.make("doubleIntegrator-v0")
# env.env.seed(1)
env = gym_basic.envs.integrator_system.DoubleIntegratorEnv()



# from stable_baselines.common.env_checker import check_env
#
# env = CustomEnv(arg1, ...)
# # It will check your custom environment and output additional warnings if needed
# check_env(env)



obs = env.reset()
env.state = np.array([0.1, 0.1])

for i in range(10):

    # env.render()
    print(env.state)

    # get action
    # action = controller(obs)
    action = env.action_space.sample()
    # action = np.array([0])

    # apply action
    obs, reward, done, _ = env.step(action)

    if done:
        print(f"Terminated after {i+1} iterations.")
        break

env.close()
