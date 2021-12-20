import gym

import numpy as np

from operator import itemgetter


def box_factory(config: dict) -> gym.spaces.Box:

    _config = dict()
    _defaults = {
        "low": -np.inf,
        "high": np.inf,
        "shape": (1,),
        "dtype": np.float32,
    }
    # Merge dictionaries, being careful not to overwrite existing entries.
    _config = {**_defaults, **config}

    low, high, shape, dtype = itemgetter("low", "high", "shape", "dtype")(_config)

    if low is str:
        if low == "inf":
            low = np.inf

        elif low == "-inf":
            low = -np.inf

    if dtype is str:
        dtype = np.dtype(dtype=dtype)

    _space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    # from scipy.spatial import HalfspaceIntersection

    # halfspaces = np.array(
    #     [[-1, 0.0, 0.0], [0.0, -1.0, 0.0], [2.0, 1.0, -4.0], [-0.5, 1.0, -2.0]]
    # )
    # feasible_point = np.array([0.5, 0.5])
    # hs = HalfspaceIntersection(halfspaces, feasible_point)

    return _space


# def convex_polytope_factory(config:dict) ->
