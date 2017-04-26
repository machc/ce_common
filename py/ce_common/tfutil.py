""" tensorflow utilities """

import numpy as np
import tensorflow as tf


def count_weights(print_perlayer=True):
    """ Count number of weights on current tf graph. """
    acc_total = 0
    for v in tf.trainable_variables():
        dims = v.get_shape().as_list()
        total = np.prod(dims)
        acc_total += total
        if print_perlayer:
            print('{}: {}, {}'.format(v.name, dims, total))
    print('Accumulated total: {}'.format(acc_total))

    return acc_total
