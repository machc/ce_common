import numpy as np
import pytest

from ce_common import math


test_softmax_A = [np.random.rand(20, 20),
                  np.random.rand(20, 19),
                  np.random.rand(19, 20)]


@pytest.mark.parametrize("A", test_softmax_A)
def test_softmax(A):
    assert math.softmax(A).shape == A.shape
    assert np.allclose(math.softmax(A, axis=0).sum(axis=0), 1)
    assert np.allclose(math.softmax(A, axis=1).sum(axis=1), 1)


def test_rotvol():
    V = np.zeros((10, 10, 10))
    V[0, 0, 0] = 1
    mid = (np.array(V.shape) - 1)/2.

    R = math.rotx(np.pi/2)
    out_gt = np.zeros((10, 10, 10))
    out_gt[0, -1, 0] = 1
    out = math.rotvol(V, R, mid)
    assert np.allclose(out, out_gt)

    R = math.roty(np.pi)
    out_gt = np.zeros((10, 10, 10))
    out_gt[-1, 0, -1] = 1
    out = math.rotvol(V, R, mid)
    assert np.allclose(out, out_gt)

    R = math.rotz(np.pi/2)
    out_gt = np.zeros((10, 10, 10))
    out_gt[-1, 0, 0] = 1
    out = math.rotvol(V, R, mid)
    assert np.allclose(out, out_gt)


def test_absmax():
    for _ in range(3):
        x = np.random.rand(20, 25)
        res = math.absmax(x)
        ref = abs(x).max()

        assert np.allclose(abs(res), ref)

    for _ in range(3):
        x = np.random.rand(20, 25)

        res = math.absmax(x, axis=0)
        ref = x.max(axis=0)
        assert np.allclose(abs(res), ref)

        res = math.absmax(x, axis=1)
        ref = x.max(axis=1)
        assert np.allclose(abs(res), ref)
        
