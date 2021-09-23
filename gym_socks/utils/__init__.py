import numpy as np


def normalize(v):
    return v / np.sum(v, axis=0)


def indicator_fn(points, set):
    _l = set.low
    _h = set.high

    return np.array(
        np.all(points >= _l, axis=1) & np.all(points <= _h, axis=1), dtype=np.bool
    )


def generate_batches(num_elements=1, batch_size=1):

    start = 0

    for _ in range(num_elements // batch_size):
        end = start + batch_size

        yield slice(start, end)

        start = end

    if start < num_elements:
        yield slice(start, n)
