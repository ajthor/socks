from gym_socks.utils import indicator_fn


def _fht_step(Y, V, constraint_set, target_set):
    Y_in_constraint_set = indicator_fn(Y, constraint_set)
    Y_in_target_set = indicator_fn(Y, target_set)

    return Y_in_target_set + (Y_in_constraint_set & ~Y_in_target_set) * V


def _tht_step(Y, V, constraint_set, target_set):
    Y_in_constraint_set = indicator_fn(Y, constraint_set)

    return Y_in_constraint_set * V
