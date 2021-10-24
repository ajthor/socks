__all__ = [
    # Systems
    "cwh",
    "integrator",
    "nonholonomic",
    "point_mass",
    "QUAD20",
    "tora",
    # Classes
    "policy",
    "sample",
]

from gym_socks.envs.cwh import CWH4DEnv
from gym_socks.envs.cwh import CWH6DEnv

from gym_socks.envs.integrator import NDIntegratorEnv

from gym_socks.envs.nonholonomic import NonholonomicVehicleEnv

from gym_socks.envs.planar_quad import PlanarQuadrotorEnv

from gym_socks.envs.point_mass import NDPointMassEnv

from gym_socks.envs.QUAD20 import QuadrotorEnv

from gym_socks.envs.tora import TORAEnv

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
    id="2DIntegratorEnv-v0",
    entry_point=NDIntegratorEnv,
    kwargs={"dim": 2},
    order_enforce=False,
)

register(
    id="NonholonomicVehicleEnv-v0",
    entry_point="gym_socks.envs:NonholonomicVehicleEnv",
)

register(
    id="PlanarQuadrotorEnv-v0",
    entry_point="gym_socks.envs:PlanarQuadrotorEnv",
)

register(
    id="2DPointMassEnv-v0",
    entry_point=NDPointMassEnv,
    kwargs={"dim": 2},
    order_enforce=False,
)

register(
    id="QuadrotorEnv-v0",
    entry_point="gym_socks.envs:QuadrotorEnv",
)

register(
    id="TORAEnv-v0",
    entry_point="gym_socks.envs:TORAEnv",
)
