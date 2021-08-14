import unittest

import gym

import gym_basic.envs

import numpy as np

system_list = [
    gym_basic.envs.integrator.NDIntegratorEnv(1),
    gym_basic.envs.integrator.NDIntegratorEnv(2),
    gym_basic.envs.integrator.NDIntegratorEnv(3),
    gym_basic.envs.integrator.NDIntegratorEnv(4),
    gym_basic.envs.integrator.StochasticNDIntegratorEnv(1),
    gym_basic.envs.integrator.StochasticNDIntegratorEnv(2),
    gym_basic.envs.integrator.StochasticNDIntegratorEnv(3),
    gym_basic.envs.integrator.StochasticNDIntegratorEnv(4),
    gym_basic.envs.point_mass.NDPointMassEnv(1),
    gym_basic.envs.point_mass.NDPointMassEnv(2),
    gym_basic.envs.point_mass.NDPointMassEnv(3),
    gym_basic.envs.point_mass.NDPointMassEnv(4),
    gym_basic.envs.point_mass.StochasticNDPointMassEnv(1),
    gym_basic.envs.point_mass.StochasticNDPointMassEnv(2),
    gym_basic.envs.point_mass.StochasticNDPointMassEnv(3),
    gym_basic.envs.point_mass.StochasticNDPointMassEnv(4),
    gym_basic.envs.nonholonomic.NonholonomicVehicleEnv(),
    gym_basic.envs.cwh.CWH4DEnv(),
    gym_basic.envs.cwh.CWH6DEnv(),
    gym_basic.envs.cwh.StochasticCWH4DEnv(),
    gym_basic.envs.cwh.StochasticCWH6DEnv(),
    gym_basic.envs.QUAD20.QuadrotorEnv(),
    gym_basic.envs.QUAD20.StochasticQuadrotorEnv(),
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

                    for i in range(5):

                        # get action
                        action = env.action_space.sample()

                        # apply action
                        obs, reward, done, _ = env.step(action)

                        if done:
                            break

                    env.close()

                except ExceptionType:
                    self.fail("myFunc() raised ExceptionType")
