import itertools
import threading
import functools
from copy import deepcopy

import numpy as np
import scipy.interpolate


class AttrDict(dict):
    """ Dict that allows access like attributes (d.key instead of d['key']) .

    From: http://stackoverflow.com/a/14620633/6079076
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class temp_nprandom_state:
    def __init__(self, seed=0):
        self.seed = seed

    def __enter__(self):
        self.prev_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.prev_state)


def threaded(fn):
    """ Decorator to run function on its own thread.

    From: https://stackoverflow.com/questions/19846332/python-threading-inside-a-class/19846691#19846691 """
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    return wrapper


def combine_params(params, add_runid=False):
    """ Dict of lists to list of dicts.

    Return list of all possible combinations of params.

    Args:
        params (dict): format 'key': [all possible values]
        add_runid
    """
    if not isinstance(params, list):
        params = [params]

    out = []
    for parlist in params:
        # make sure inputs are lists
        for k, v in parlist.items():
            if not isinstance(v, list):
                parlist[k] = [v]

        parlists = [dict(zip(parlist.keys(), x))
                    for x in itertools.product(*parlist.values())]

        if add_runid:
            for p in parlists:
                run_id = ''
                for k, v in sorted(p.items()):
                    if len(params[0][k]) > 1:
                        run_id += k + '_' + str(v) + '_'
                p['run_id'] = p.get('run_id', '') + run_id[:-1]

        out += parlists

    return out


def list_params(common, changing, add_runid=False):
    """ Add dict to list of several dicts.

        Return list of params; each element contains all keys from 'common'
    and its own keys from 'changing'

    Args:
        common (dict): common params (all outputs will have these)
        changing (list of dict): changing params; each output corresponds to one of these
        add_runid (str): add identifier to each params
    """
    assert isinstance(common, dict)
    assert isinstance(changing, list)

    # warning: python >= 3.6 only!
    parlist = [{**common, **x} for x in changing]

    if add_runid:
        # use only changing params to write run_id
        for p, c in zip(parlist, changing):
            run_id = ''
            for k in sorted(c.keys()):
                run_id += k + '_' + str(p[k]) + '_'
            p['run_id'] = run_id[:-1]

    return parlist


def paramdict2str(d, exclude=[]):
    """ Convert dict of params to string of form --name1=value1 --name2=value2 ... """
    out = ''
    for k, v, in sorted(d.items()):
        if k not in exclude:
            prefix = '' if k.startswith('-') or k.startswith('@')  else '--'
            out += ' {}{}{}{}'.format(prefix, k, '=' if v != '' else '', v)

    return out


def to_timevec(tout, x, tin, kind='linear'):
    """ Convert timeseries to given time vector.

    Args:
        tout (n x 1): output time
        x (m x l): input time series
        tin (m x 1): input time
    """
    if x.ndim == 1:
        x = x[..., np.newaxis]

    out = [scipy.interpolate.interpolate.interp1d(tin, xdim, kind=kind)(tout)
           for xdim in x.transpose()]

    return np.squeeze(np.array(out).T)


def grouper(iterable, n, rounding_mode='insert_none'):
    """ Iterate over chunks of iterable.

    Args:
        rounding_mode ['insert_none', 'ignore', or 'early_termination']:
    what to do when len(iterable) is not multiple of n

    Based on: http://stackoverflow.com/a/434411/6079076
    """
    def remove_none(X):
        for Y in X:
            yield tuple([y for y in Y if y is not None])

    def early_term(X):
        for Y in X:
            if all([y is not None for y in Y]):
                yield Y
    args = [iter(iterable)] * n
    if rounding_mode == 'ignore':
        return remove_none(itertools.zip_longest(*args, fillvalue=None))
    elif rounding_mode == 'early_termination':
        return early_term(itertools.zip_longest(*args, fillvalue=None))
    elif rounding_mode == 'insert_none':
        return itertools.zip_longest(*args, fillvalue=None)


def rescale(val, lim_orig=None, lim_out=(-1, 1)):
    """ Rescale val from limits lim0 to limits lim1. """
    if lim_orig is None:
        lim_orig = (val.min(), val.max())
    return (lim_out[0] +
            (lim_out[1] - lim_out[0]) *
            (val - lim_orig[0]) /
            (lim_orig[1] - lim_orig[0]))


def to_one_hot(v):
    """ Convert vector to one hot form. """
    n = len(v)
    m = max(v) + 1
    out = np.zeros((n, m))
    out[np.arange(n), v] = 1
    return out


def closest_sorted(x, v):
    """ Return id closest to v in vector x.

    Examples:
    >>> closest_sorted([1,2,3,4], 2.9)
    2
    >>> closest_sorted([1,2,3,4], 3.1)
    2
    >>> closest_sorted([1,2,3,4], 3.6)
    3

    """
    c1 = np.searchsorted(x, v)  # first candidate
    c2 = c1 - 1
    if abs(x[c1] - v) > abs(x[c2] - v):
        return c2
    else:
        return c1


def shuffle_all(*args):
    """ Do the same random permutation to all inputs. """
    idx = list(np.random.permutation(len(args[0])))
    try:
        # np arrays can be indexed like this
        return [a[idx] for a in args]
    except TypeError:
        return [[a[i] for i in idx] for a in args]


def circumscribed_square(rect):
    """ Return square circumscribing rectangle with integer coordinates.

    Useful for making bounding boxes square.

    Args:
        rect: (top, left, width, height)
    Returns:
        square (top, left, length, lenght)

    Examples:
    >>> circumscribed_square([10,10,10,20])
    [5, 10, 20, 20]
    >>> circumscribed_square([7,11,20,10])
    [7, 6, 20, 20]
    """
    l, t, w, h = [np.int32(x) for x in rect]
    if w < h:
        d = (h-w)/2
        l -= int(np.floor(d))
        w += int(2*d)
    else:
        d = (w-h)/2
        t -= int(np.floor(d))
        h += int(2*d)

    assert h == w

    return [l, t, w, h]


def inscribed_square(rect):
    """ Return central square inscribed in rectangle with integer coordinates.

    Useful for making rectangular image square.

    Args:
        rect: (top, left, width, height)
    Returns:
        square (top, left, length, lenght)

    Examples:
    >>> inscribed_square([10,10,10,20])
    [10, 15, 10, 10]
    >>> inscribed_square([7,11,20,10])
    [12, 11, 10, 10]
    """
    l, t, w, h = [np.int32(x) for x in rect]
    if w < h:
        d = (h-w)/2
        t += int(np.floor(d))
        h -= int(2*d)
    else:
        d = (w-h)/2
        l += int(np.floor(d))
        w -= int(2*d)

    assert h == w

    return [l, t, w, h]


def decode_maybe(s):
    """ Do nothing is s is string, else attempt to decode. """
    if isinstance(s, str):
        return s
    else:
        return s.decode()


def lru_cache_copy(maxsize=128, typed=False):
    """Same as functools.lru_cache, but returns copies.

    This way the cache cannot be modified.

    Source: https://stackoverflow.com/q/54909357
    """
    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator
