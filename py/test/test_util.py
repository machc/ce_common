import numpy as np
import pytest

from ce_common import util


def test_combine_params():
    par = {'a': [1, 2, 3], 'b': [4, 5]}
    plist = util.combine_params(par)
    assert len(plist) == 6
    assert {'a': 2, 'b': 5} in plist

    plist = util.combine_params(par, add_runid=True)
    assert ({'a': 2, 'b': 5, 'run_id': 'a_2_b_5'} in plist or
            {'a': 2, 'b': 5, 'run_id': 'b_5_a_2'} in plist)


def test_list_params():
    common = {'c': 'blabla', 'd': 9}
    par = [{'a': 1, 'b': 4}, {'a':1, 'b':5}, {'a': 2, 'b':3}]
    plist = util.list_params(common, par)

    assert len(plist) == 3
    assert {'a': 2, 'b': 3, 'c': 'blabla', 'd': 9} in plist

    common = {'c': 'blabla'}
    par = [{'a': 1, 'b': 2}, {'a': 2, 'b': 1}]
    plist = util.list_params(common, par, add_runid=True)
    assert len(plist) == 2
    assert ({'a': 1, 'b': 2, 'c': 'blabla', 'run_id': 'a_1_b_2'} in plist or
            {'a': 1, 'b': 2, 'c': 'blabla', 'run_id': 'b_2_a_1'} in plist)
    

def test_rescale():
    assert np.allclose(util.rescale(0, (-1, 1), (0, 96)), 48)
    assert np.allclose(util.rescale(-1, (-1, 1), (0, 96)), 0)
    assert np.allclose(util.rescale(1, (-1, 1), (0, 96)), 96)
    assert np.allclose(util.rescale(48, (0, 96), (-1, 1)), 0)
    

def test_to_timevec():
    x = np.array([4, 5, 6])
    t = np.array([1, 2, 3])
    new_t = np.array([1.2, 2.2])

    ref = np.array([4.2, 5.2])
    out = util.to_timevec(new_t, x, t)
    assert np.allclose(out, ref)

    ref = np.array([4., 5.])
    out = util.to_timevec(new_t, x, t, kind='nearest')
    assert np.allclose(out, ref)
    
