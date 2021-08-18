import unittest

import gym

import systems.envs

import numpy as np

system_list = [
    systems.envs.integrator.NDIntegratorEnv(1),
    systems.envs.integrator.NDIntegratorEnv(2),
    systems.envs.integrator.NDIntegratorEnv(3),
    systems.envs.integrator.NDIntegratorEnv(4),
    systems.envs.integrator.StochasticNDIntegratorEnv(1),
    systems.envs.integrator.StochasticNDIntegratorEnv(2),
    systems.envs.integrator.StochasticNDIntegratorEnv(3),
    systems.envs.integrator.StochasticNDIntegratorEnv(4),
    systems.envs.point_mass.NDPointMassEnv(1),
    systems.envs.point_mass.NDPointMassEnv(2),
    systems.envs.point_mass.NDPointMassEnv(3),
    systems.envs.point_mass.NDPointMassEnv(4),
    systems.envs.point_mass.StochasticNDPointMassEnv(1),
    systems.envs.point_mass.StochasticNDPointMassEnv(2),
    systems.envs.point_mass.StochasticNDPointMassEnv(3),
    systems.envs.point_mass.StochasticNDPointMassEnv(4),
    systems.envs.nonholonomic.NonholonomicVehicleEnv(),
    systems.envs.cwh.CWH4DEnv(),
    systems.envs.cwh.CWH6DEnv(),
    systems.envs.cwh.StochasticCWH4DEnv(),
    systems.envs.cwh.StochasticCWH6DEnv(),
    systems.envs.QUAD20.QuadrotorEnv(),
    systems.envs.QUAD20.StochasticQuadrotorEnv(),
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
