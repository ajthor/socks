import systems.envs

from systems.sample import generate_sample
from systems.sample import generate_sample_trajectories

from gym.envs.registration import register

register(
    id="doubleIntegrator-v0",
    entry_point="systems.envs:DoubleIntegratorEnv",
)

register(
    id="CWH4DEnv-v0",
    entry_point="systems.envs:CWH4DEnv",
)

register(
    id="CWH6DEnv-v0",
    entry_point="systems.envs:CWH6DEnv",
)

register(
    id="StochasticNDIntegratorEnv-v0",
    entry_point="systems.envs:StochasticNDIntegratorEnv",
)

register(
    id="NonholonomicVehicleEnv-v0",
    entry_point="systems.envs:NonholonomicVehicleEnv",
)

register(
    id="NDPointMassEnv-v0",
    entry_point="systems.envs:NDPointMassEnv",
)

register(
    id="StochasticNDPointMassEnv-v0",
    entry_point="systems.envs:StochasticNDPointMassEnv",
)

register(
    id="QuadrotorEnv-v0",
    entry_point="systems.envs:QuadrotorEnv",
)

register(
    id="StochasticQuadrotorEnv-v0",
    entry_point="systems.envs:StochasticQuadrotorEnv",
)
