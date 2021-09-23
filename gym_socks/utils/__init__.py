__all__ = ["batch"]

import numpy as np


def normalize(v):
    return v / np.sum(v, axis=0)


def indicator_fn(points, set):
    _l = set.low
    _h = set.high

    return np.array(
        np.all(points >= _l, axis=1) & np.all(points <= _h, axis=1), dtype=np.bool
    )
