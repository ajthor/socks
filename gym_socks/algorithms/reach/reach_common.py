import gym

import numpy as np

from gym_socks.utils import indicator_fn


def _fht_step(Y, V, constraint_set, target_set):
    r"""First-hitting time problem backward recursion step.

    This function implements the backward recursion step for the first-hitting time
    problem, given by:

    .. math::

        V_{t}(x) = 1_{\mathcal{T}}(x)
        + 1_{\mathcal{K} \backslash \mathcal{T}}(x) \mathbb{E}[V_{t+1}]

    Args:
        Y: Vector of data points in the sample.
        V: Vector of coefficients from the kernel-based approximation.
        constraint_set: A space or function which determines whether a point is in the
            constraint set. This can be either a space from gym.spaces or a function
            which acts as a 0/1 classifier. See the `indicator_fn` implementation for
            more information.
        target_set: A space or function which determines whether a point is in the
            target set. This can be either a space from gym.spaces or a function which
            acts as a 0/1 classifier. See the `indicator_fn` implementation for more
            information.

    Returns:
        The kernel-based approximation of the value function.

    """

    Y_in_constraint_set = indicator_fn(Y, constraint_set)
    Y_in_target_set = indicator_fn(Y, target_set)

    return Y_in_target_set + (Y_in_constraint_set & ~Y_in_target_set) * V


def _tht_step(Y, V, constraint_set, target_set):
    r"""Terminal-hitting time problem backward recursion step.

    This function implements the backward recursion step for the terminal-hitting time
    problem, given by:

    .. math::

        V_{t}(x) = 1_{\mathcal{K}}(x) \mathbb{E}[V_{t+1}]

    Args:
        Y: Vector of data points in the sample.
        V: Vector of coefficients from the kernel-based approximation.
        constraint_set: A space or function which determines whether a point is in the
            constraint set. This can be either a space from gym.spaces or a function
            which acts as a 0/1 classifier. See the `indicator_fn` implementation for
            more information.
        target_set: This argument is unused in the terminal-hitting time problem. The
            reson it is allowed is to avoid extra if statements in the implementation.

    Returns:
        The kernel-based approximation of the value function.

    """

    Y_in_constraint_set = indicator_fn(Y, constraint_set)

    return Y_in_constraint_set * V


def generate_tube(time_horizon: int, low, high):
    """Generate a stochastic reachability tube using config.

    This function computes a stochastic reachability tube using the tube configuration.

    Args:
        env: The dynamical system model.
        bounds: The bounds of the tube. Specified as a dictionary.

    Returns:
        A list of spaces indexed by time.

    """

    tube_lb = bounds["lower_bound"]
    tube_ub = bounds["upper_bound"]

    tube = []
    for i in range(time_horizon):
        tube_t = gym.spaces.Box(
            low[i],
            high[i],
            shape=shape,
            dtype=np.float32,
        )
        tube.append(tube_t)

    return tube
