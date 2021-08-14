
import numpy as np

def as_col(arr):
    shape = arr.shape

    err_msg = "%r (%s) invalid" % (action, type(action))
    assert np.squeeze(arr).ndim == 1, err_msg

    if shape[0] == 1:
        return arr.T
    else:
        return arr
