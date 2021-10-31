import gym

import numpy as np

from sacred import Ingredient

from gym_socks.envs.dynamical_system import DynamicalSystem

cwh_ingredient = Ingredient("cwh")


@cwh_ingredient.config
def _config():

    norm_order = 2
    squared = False

    target_state = [0, 0, 0, 0]

    terminal_constraint = {
        "lower_bound": [-0.2, -0.2, -0.05, -0.05],
        "upper_bound": [-0.2, 0, 0.05, 0.05],
    }


@cwh_ingredient.capture
def make_cost(env: DynamicalSystem, target_state, norm_order, squared):
    def _cost_fn(time: int = 0, state: np.ndarray = None) -> float:
        """CWH cost function.

        The cost is defined such that we seek to minimize the distance from the system
        to the origin. This would indicate a fully "docked" spacecraft with zero
        terminal velocity.

        """

        dist = state - np.array([target_state], dtype=np.float32)
        result = np.linalg.norm(dist, ord=norm_order, axis=1)
        if squared is True:
            result = np.power(result, 2)
        return result

    return _cost_fn


@cwh_ingredient.capture
def make_constraint(env: DynamicalSystem):
    def _constraint_fn(time: int = 0, state: np.ndarray = None) -> float:
        """CWH constraint function.

        The CWH constraint function is defined as the line of sight (LOS) cone from the
        spacecraft, where the velocity components are sufficiently small.

        Note:
            The constraints are written in terms of indicator functions, which return a
            one if the sample is in the LOS cone (or in the target space) and a zero if
            it is not, but the optimal control problem is defined such that the
            constraints are satisfied if the function is less than or equal to zero.
            Thus, we use the following algebraic manipulation::

                1[A](x) >= 1
                - 1[A](x) <= -1
                - 1[A](x) + 1 <= 0

        """

        # Terminal constraint.
        if time < env.num_time_steps - 1:
            satisfies_constraints = (
                -np.array(
                    (np.abs(state[:, 0]) < np.abs(state[:, 1]))
                    & (np.abs(state[:, 2]) <= 0.05)
                    & (np.abs(state[:, 3]) <= 0.05),
                    dtype=np.float32,
                )
                + 1
            )

            return np.round(satisfies_constraints, decimals=2)

        # LOS constraint.
        else:
            satisfies_constraints = (
                -np.array(
                    (np.abs(state[:, 0]) < 0.2)
                    & (state[:, 1] >= -0.2)
                    & (state[:, 1] <= 0)
                    & (np.abs(state[:, 2]) <= 0.05)
                    & (np.abs(state[:, 3]) <= 0.05),
                    dtype=np.float32,
                )
                + 1
            )

            return np.round(satisfies_constraints, decimals=2)

    return _constraint_fn
