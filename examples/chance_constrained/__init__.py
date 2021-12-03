from examples.chance_constrained.cc_env import NonMarkovIntegratorEnv

from gym.envs.registration import register

register(
    id="NonMarkovIntegratorEnv-v0",
    entry_point=NonMarkovIntegratorEnv,
    order_enforce=False,
)
