import itertools

import numpy as np


def combine_params(params, add_runid=False):
    """ Dict of lists to list of dicts.

    Return list of all possible combinations of params.

    Args:
        params (dict): format 'key': [all possible values]
        add_runid
    """
    # make sure inputs are lists
    for k, v in params.items():
        if not isinstance(v, list):
            params[k] = [v]

    parlist = [dict(zip(params.keys(), x)) for x in itertools.product(*params.values())]

    if add_runid:
        for p in parlist:
            run_id = ''
            for k, v in sorted(p.items()):
                if len(params[k]) > 1:
                    run_id += k + '_' + str(v) + '_'
            p['run_id'] = p.get('run_id', '') + run_id[:-1]

    return parlist


def list_params(common, changing, add_runid=False):
    """ Add dict to list of several dicts.

    Return list of params.

    Args:
        common (dict): common params (all outputs will have these)
        changing (list of dict): changing params; each output correspond to one of these
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


def to_timevec(tout, x, tin, kind='linear'):
    """ Convert timeseries to given time vector.

    Args:
        tout (n x 1): output time
        x (m x l): input time series
        tin (m x 1): input time
    """
    if x.ndim == 1:
        x = x[..., np.newaxis]

    return np.squeeze(np.array([np.interp(tout, tin, xdim)
                                for xdim in x.transpose()]).T)


def grouper(iterable, n, fillvalue=None):
    """ Iterate over chunks of iterable.

    Note: will fill last value with None if size is not a multiple of n.

    From: http://stackoverflow.com/a/434411/6079076
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def rescale(val, lim_orig=None, lim_out=(-1,1)):
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


