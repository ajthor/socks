import unittest

import gym

import gym_socks.envs

import numpy as np

system_list = [
    gym_socks.envs.NDIntegratorEnv(1),
    gym_socks.envs.NDIntegratorEnv(2),
    gym_socks.envs.NDIntegratorEnv(3),
    gym_socks.envs.NDIntegratorEnv(4),
    gym_socks.envs.StochasticNDIntegratorEnv(1),
    gym_socks.envs.StochasticNDIntegratorEnv(2),
    gym_socks.envs.StochasticNDIntegratorEnv(3),
    gym_socks.envs.StochasticNDIntegratorEnv(4),
    gym_socks.envs.NDPointMassEnv(1),
    gym_socks.envs.NDPointMassEnv(2),
    gym_socks.envs.NDPointMassEnv(3),
    gym_socks.envs.NDPointMassEnv(4),
    gym_socks.envs.StochasticNDPointMassEnv(1),
    gym_socks.envs.StochasticNDPointMassEnv(2),
    gym_socks.envs.StochasticNDPointMassEnv(3),
    gym_socks.envs.StochasticNDPointMassEnv(4),
    gym_socks.envs.NonholonomicVehicleEnv(),
    gym_socks.envs.CWH4DEnv(),
    gym_socks.envs.CWH6DEnv(),
    gym_socks.envs.StochasticCWH4DEnv(),
    gym_socks.envs.StochasticCWH6DEnv(),
    gym_socks.envs.QuadrotorEnv(),
    gym_socks.envs.StochasticQuadrotorEnv(),
]


class TestEnvironmentsRun(unittest.TestCase):
    def test_envs_run(cls):
        """
        Assert that environments can run.
        """

        for env in system_list:
            with cls.subTest(msg=f"Testing with {type(env)}."):

                try:
                    obs = env.reset()

                    for i in range(env.num_time_steps):

                        # get action
                        action = env.action_space.sample()

                        # apply action
                        obs, reward, done, _ = env.step(action)

                        if done:
                            break

                    env.close()

                except ExceptionType:
                    self.fail("myFunc() raised ExceptionType")
