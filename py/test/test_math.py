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
