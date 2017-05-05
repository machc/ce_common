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


def rotx(ang):
    """ Return rotation around x.

    >>> np.allclose(rotx(0), np.eye(3))
    True
    >>> np.allclose(rotx(np.pi/2), np.array([[1,0,0], [0, 0, -1], [0, 1, 0]]))
    True
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(ang), -np.sin(ang)],
                     [0, np.sin(ang), np.cos(ang)]])


def roty(ang):
    """ See also: rotx. """
    return np.array([[np.cos(ang), 0, np.sin(ang)],
                     [0, 1, 0],
                     [-np.sin(ang), 0, np.cos(ang)]])


def rotz(ang):
    """ See also: rotx. """
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang), 0],
                     [0, 0, 1]])


def vec2skew(v):
    """ Vector to skew symmetric matrix. """
    assert v.shape == (3,)
    S = np.array([[    0, -v[2],  v[1]],
                  [ v[2],     0, -v[0]],
                  [-v[1],  v[0],    0]])

    return S
