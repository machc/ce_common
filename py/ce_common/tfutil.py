""" tensorflow utilities """
import os
import subprocess
import pickle
import queue
import threading
import re
import uuid
import time

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


def run_tensorboard(logdirs, ids, port=6123):
    """ Run tensorboard on given directories and run_ids. """
    # tensorboard command
    tbcmd = "source ~/.profile; pkill -9 tensorboard; tensorboard --reload_interval 60 --logdir={} --port={} &"

    tbflags = ''
    for run_id, logdir in zip(ids, logdirs):
        tbflags += run_id + ':' + logdir + ','
    tbflags = tbflags[:-1]
    # WARNING: we are ignoring this note from the manual:
    # Note: Do not use stdout=PIPE or stderr=PIPE with this function
    #       as that can deadlock based on the child process output volume.
    #       Use Popen with the communicate() method when you need pipes.
    subprocess.call(tbcmd.format(tbflags, port),
                    shell=True,
                    executable='bash',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)


def dispatch(params,
             outfile,
             regexp='',
             gpus=[0, 1, 2, 3],
             verbose=False):
    """ Dispatch list of training jobs.

    Args:
        params (list of dicts): containing 'cmd', 'id', 'logdir' keys
        outfile (str): file to save results to
        regexp (str): print stdout lines matching this
        gpus (list or list of lists): if list, entries are GPU ids; run locally
                                      if list of lists, entries are of format ['host', GPU_ID]

    """

    q = queue.Queue()
    for p in params:
        q.put(p)

    print("Starting queue of {} jobs on {} GPUs".format(len(params), len(gpus)))
    print('\n'.join(['{}: {}'.format(p['id'], p['cmd']) for p in params]))

    out = {}

    def process(gpu, q):
        """ Thread to initiate training processes. """
        while not q.empty():
            p = q.get()
            if isinstance(gpu, list):
                cmd = ('ssh {} CUDA_VISIBLE_DEVICES={} python3 {}'
                       .format(gpu[0], gpu[1], p['cmd']).split(' '))
                env = None
            else:
                cmd = ('python3 {}'.format(p['cmd']).split(' '))
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)

            # print(cmd)
            # note: if set env=CUDA_VISIBLE_DEVICES when running remotely, ssh-keys won't work
            logname = '{}/{}.log'.format(p['logdir'], p['id'])
            os.makedirs(p['logdir'], exist_ok=True)
            with open(logname, 'wt') as fout:
                out[p['id']] = subprocess.run(cmd,
                                              env=env,
                                              stdout=fout,
                                              stderr=fout)
            with open(logname, 'rt') as fin:
                lines = fin.readlines()

            rc = out[p['id']].lines = lines
            rc = out[p['id']].returncode
            if rc != 0:
                # TODO: add errors back to queue?
                #       doesn't soound like a great idea, some errors are harmless
                errmsg = ''
                for l in lines:
                    if ('Error' in l) or (verbose):
                        errmsg += l
                if errmsg:
                    print('{} returned {}. (possible error)\n{}'
                          .format(p['id'], out[p['id']].returncode, errmsg))

            for l in lines:
                m = re.match(regexp, l)
                if m:
                    print(m.string, end='')

    def manage_tb(entries, q, tsleep=30):
        """ Thread to manage tensorboard.

        Will compare queue and list, and run tb on the running/finished processes. """

        # cov_w0:$logdir/cov_w0'
        set_entries = set([tuple(e.items()) for e in entries])
        prev_qsize = len(entries) + 1
        while prev_qsize > 0:
            curr_qsize = q.qsize()
            curr_q = set([tuple(e.items()) for e in q.queue])
            # sleep to make sure new runs are properly initiated
            time.sleep(tsleep)
            if prev_qsize != curr_qsize:  # rerun tb?
                # tensorboard flags
                run_tensorboard([dict(e)['logdir'] for e in set_entries - curr_q],
                                [dict(e)['id'] for e in set_entries - curr_q])
                prev_qsize = curr_qsize

    tbt = threading.Thread(target=manage_tb, args=[params, q])
    tbt.daemon = True
    tbt.start()

    threads = []

    for gpu in gpus:
        t = threading.Thread(target=process, args=[gpu, q])
        t.daemon = True
        t.start()
        threads.append(t)

    # join all threads
    for t in threads:
        t.join()

    # save results
    # TODO: write function to load these results!
    fname = os.path.expanduser(outfile)
    os.makedirs(os.path.split(fname)[0], exist_ok=True)
    with open(fname, 'wb') as fout:
        print('Saving results to {}'.format(fname))
        pickle.dump({'params': params, 'out': out}, fout)

    return out
