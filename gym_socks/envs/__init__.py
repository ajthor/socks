__all__ = ["cwh", "integrator", "nonholonomic", "point_mass", "QUAD20", "tora"]

from gym_socks.envs.cwh import CWH4DEnv
from gym_socks.envs.cwh import StochasticCWH4DEnv

from gym_socks.envs.cwh import CWH6DEnv
from gym_socks.envs.cwh import StochasticCWH6DEnv

from gym_socks.envs.integrator import NDIntegratorEnv
from gym_socks.envs.integrator import StochasticNDIntegratorEnv

from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv
from gym_socks.envs.nonholonomic import StochasticNonholonomicVehicleEnv

from gym_socks.envs.point_mass import NDPointMassEnv
from gym_socks.envs.point_mass import StochasticNDPointMassEnv

from gym_socks.envs.QUAD20 import QuadrotorEnv
from gym_socks.envs.QUAD20 import StochasticQuadrotorEnv

from gym_socks.envs.tora import TORAEnv
from gym_socks.envs.tora import StochasticTORAEnv

# ---- Hybrid system models ----

from gym_socks.envs.temperature import TemperatureRegEnv

# ---- Sample generation functions ----

from gym_socks.envs.sample import random_initial_conditions
from gym_socks.envs.sample import uniform_initial_conditions

from gym_socks.envs.sample import generate_sample
from gym_socks.envs.sample import generate_sample_trajectories

# ---- Register gym envs ----

from gym.envs.registration import register

register(
    id="CWH4DEnv-v0",
    entry_point="gym_socks.envs:CWH4DEnv",
)

register(
    id="CWH6DEnv-v0",
    entry_point="gym_socks.envs:CWH6DEnv",
)

register(
    id="NonholonomicVehicleEnv-v0",
    entry_point="gym_socks.envs:NonholonomicVehicleEnv",
)

register(
    id="QuadrotorEnv-v0",
    entry_point="gym_socks.envs:QuadrotorEnv",
)

register(
    id="StochasticQuadrotorEnv-v0",
    entry_point="gym_socks.envs:StochasticQuadrotorEnv",
)
