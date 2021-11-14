"""Simulation ingredient.

Used for experiments where the system needs to be simulated, such as after computing a
policy via one of the included control algorithms.

"""

from sacred import Ingredient

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import BasePolicy

import numpy as np

simulation_ingredient = Ingredient("simulation")


@simulation_ingredient.config
def config():
    """Simulation configuration variables."""

    initial_condition = None  # Initial condition of the system.

    filename = "results/data.npy"


@simulation_ingredient.capture
def simulate_system(
    env: DynamicalSystem, policy: BasePolicy, initial_condition: list, _log
) -> list:
    """Simulate the system from a given initial condition."""

    _log.debug("Simulating the system.")

    # Set the initial condition.
    env.reset()

    if initial_condition is None:
        initial_condition = env.state_space.sample()

    env.state = initial_condition

    trajectory = [env.state]

    # Simulate the env using the computed policy.
    for t in range(env.num_time_steps):

        action = np.array(policy(time=t, state=[env.state]), dtype=np.float32)
        obs, cost, done, _ = env.step(time=t, action=action)
        next_state = env.state

        trajectory.append(list(next_state))

        if done:
            break

    return trajectory
