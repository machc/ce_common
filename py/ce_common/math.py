""" Math utilities """

import numpy as np


def softmax(x, axis=0):
    assert axis in [0, 1]

    min_ax = x.min(axis=axis)
    min_ax = min_ax[:, np.newaxis] if axis == 1 else min_ax
    den = np.exp(x - min_ax).sum(axis=axis)
    den = den[:, np.newaxis] if axis == 1 else den

    return np.exp(x - min_ax) / den


def angdiff(x, y):
    d = x - y
    return ((d + np.pi) % (2*np.pi)) - np.pi
