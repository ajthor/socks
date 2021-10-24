from gym_socks.utils import indicator_fn


def _fht_step(Y, V, constraint_set, target_set):
    """First-hitting time problem backward recursion step.

    This function implements the backward recursion step for the first-hitting time
    problem, given by::

        V_t(x) = 1[T](x) + 1[K\T](x) E[V_t+1]

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
    """Terminal-hitting time problem backward recursion step.

    This function implements the backward recursion step for the terminal-hitting time
    problem, given by::

        V_t(x) = 1[K](x) E[V_t+1]

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
