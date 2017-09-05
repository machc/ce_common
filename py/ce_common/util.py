import itertools
import threading

import numpy as np
import scipy.interpolate


class AttrDict(dict):
    """ Dict that allows access like attributes (d.key instead of d['key']) .

    From: http://stackoverflow.com/a/14620633/6079076
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
    for p in params:
        # make sure inputs are lists
        for k, v in p.items():
            if not isinstance(v, list):
                p[k] = [v]

        parlist = [dict(zip(p.keys(), x)) for x in itertools.product(*p.values())]

        if add_runid:
            for p in parlist:
                run_id = ''
                for k, v in sorted(p.items()):
                    if len(p[k]) > 1:
                        run_id += k + '_' + str(v) + '_'
                p['run_id'] = p.get('run_id', '') + run_id[:-1]

        out += parlist

    return out


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


def paramdict2str(d, exclude=[]):
    """ Convert dict of params to string of form --name1=value1 --name2=value2 ... """
    return ' '.join(['--{}{}{}'.format(k, '=' if v else '', v)
                     for k, v, in sorted(d.items()) if k not in exclude])


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


