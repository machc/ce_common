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


def rotvol(V, R, p=None):
    """ Rotate volume around a point p.

    Note: (0,0,0) is top-left-front of the cube.
          x points down, y right, z back.

    Args:
        V ((x,y,z) ndarray) - input volumetric representation (binary occupancy grid)
        R ((3,3) ndarray) - rotation matrix
        p ((3,) ndarray) - center of rotation
    """
    if p is None:
        p = np.zeros(3)

    assert V.ndim == 3
    assert R.shape == (3, 3)
    assert p.shape == (3,)

    p = p[:, np.newaxis]
    x, y, z = np.meshgrid(*[range(s) for s in V.shape])
    x, y, z = [_.ravel() for _ in [x, y ,z]]
    # inverse rotation
    rotpix = (R.T.dot(np.vstack([x.ravel(), y.ravel(), z.ravel()]) - p) + p + 0.5).astype('int')
    Vout = np.zeros_like(V)

    # source
    x_s, y_s, z_s = rotpix
    valid = ((0 <= x_s) & (x_s < V.shape[0]) &
             (0 <= y_s) & (y_s < V.shape[1]) &
             (0 <= z_s) & (z_s < V.shape[2]))

    Vout[x[valid], y[valid], z[valid]] = V[x_s[valid], y_s[valid], z_s[valid]]

    return Vout


def vec2skew(v):
    """ Vector to skew symmetric matrix. """
    assert v.shape == (3,)
    S = np.array([[    0, -v[2],  v[1]],
                  [ v[2],     0, -v[0]],
                  [-v[1],  v[0],    0]])

    return S
