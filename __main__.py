# from kernel_basic import Kernel

# if __name__ == '__main__':

import gym

from gym_basic.envs.double_integrator import DoubleIntegratorEnv

# get environment
env = gym.make("doubleIntegrator-v0")
# env.env.seed(1)



# from stable_baselines.common.env_checker import check_env
#
# env = CustomEnv(arg1, ...)
# # It will check your custom environment and output additional warnings if needed
# check_env(env)



obs = env.reset()

for i in range(1000):

    # env.render()
    print(obs)

    # get action
    # action = controller(obs)
    action = env.action_space.sample()

    # apply action
    obs, reward, done, _ = env.step(action)

    if done:
        print(f"Terminated after {i+1} iterations.")
        break

env.close()
