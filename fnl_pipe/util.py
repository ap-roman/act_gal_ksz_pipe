import time

import mpi4py

import matplotlib.pyplot as plt
import numpy as np

import os
from os import path, listdir

import yaml

from pixell.curvedsky import alm2cl, map2alm
from pixell import enplot


# NOT thread safe
class Timer:
    def __init__(self):
        self.dt = 0.
        self.last_t = None
        self.active = False

    def start(self):
        if not self.active:
            self.last_t = time.time()
            self.active = True

    def stop(self):
        self.dt += time.time() - self.last_t
        self.active = False


def ensure_sep(path):
    pass


def get_dirs(base_path, stem=True):
    assert(path.isdir(base_path))
    dirs = [d for d in listdir(base_path) if path.isdir(path.join(base_path, d))]

    if stem:
        ret = dirs
    else:
        ret = [path.join(base_path, a) for a in dirs]

    return ret


def get_files(base_path, stem=True):
    assert(path.isdir(base_path))
    files = [f for f in listdir(base_path) if path.isfile(path.join(base_path, f))]

    if stem:
        ret = files
    else:
        ret = [path.join(base_path, a) for a in files]

    return ret


def get_sequential_label(base_path, title):
    dirs = get_dirs(base_path)
    dirs = [d for d in dirs if '_'.join(d.split('_')[:-1]) == title]

    max_label = -1
    for d in dirs:
        dig = d.split('_')[-1]
        if str.isdigit(dig):
            max_label = max(max_label, int(dig))

    return str(max_label + 1)


def remove_empty(lst):
    return [a for a in lst if a != '']


def ensure_dir(dirpath):
    dirsplt = remove_empty(dirpath.split(os.sep))

    if len(dirsplt) > 1:
        base = path.join(*dirsplt[:-1])
    else:
        base = ''

    if not path.exists(base):
        ensure_dir(base)
    elif not path.exists(dirpath):
        os.mkdir(dirpath)


class LogFile:
    def __init__(self, path):
        self.path = path

    def write(self, dat):
        with open(self.path, 'a') as f:
            f.write(dat)


# Manages output directory. Not parallel
# logs is a list of tuples (logname, logpath)
class OutputManager:
    def __init__(self, title, base_path='output', subdirs=['plots', 'logs'], logs=None,
                 mpi_rank=None, mpi_comm=None):
        assert title != ''
        assert 'logs' in subdirs

        self.base_path = base_path
        self.subdirs = subdirs

        self.mpi_rank = mpi_rank
        self.mpi_comm = mpi_comm
        self.is_mpi = (mpi_rank is not None)

        if self.is_mpi:
            label = None
            if self.mpi_rank ==0:
                label = title + '_' + get_sequential_label(base_path, title)
                ensure_dir(path.join(base_path, str(label)))

            label = mpi_comm.bcast(label, root=0)

            # if self.mpi_rank != 0:
            label += f'/rank_{self.mpi_rank}/'
            ensure_dir(path.join(base_path, str(label)))
        else:
            label = get_sequential_label(base_path, title)
            label = title + '_' + label

        self.working_dir = path.join(base_path, str(label))
        ensure_dir(path.join(base_path, str(label)))

        # generate subdirs
        for sd in subdirs:
            this_path = path.join(self.working_dir, sd)
            ensure_dir(this_path)

        if logs is None:
            logs = [title,]
        
        self.log_names = logs
        self.logs = {log:LogFile(os.path.join(self.working_dir, 'logs', log + '.log')) for log in logs}

    def log(self, line, log=None):
        assert log in self.log_names or log is None
        
        if line[-1] != '\n' or line[-1] != '\r':
            line += '\n'

        if log is not None:
            self.logs[log].write(line)
        else:
            for logname, log in self.logs.items():
                log.write(line)

    def printlog(self, line, log=None):
        line = str(line)
        print(line)
        self.log(line, log)

    def savefig(self, name, mode='matplotlib', fig=None):
        assert('plots' in self.subdirs)
        assert(mode == 'matplotlib' or mode == 'pixell' )

        plot_path = path.join(self.working_dir, 'plots', name)

        if mode == 'matplotlib':
            plt.savefig(plot_path)
        elif mode == 'pixell':
            assert fig is not None
            enplot.write(plot_path, fig)


def import_config(path):
    res = None

    with open(path, 'r') as f:
        res = yaml.safe_load(f)

    return res


def fequal(a,b,tol=1e-6):
    return np.all(np.abs(a - b) <= tol)


def fequal_either(a,b,tol=1e-3):
    return np.all(np.logical_or(np.abs(a - b) <= tol, 2 * np.abs(a - b) / np.abs(a + b) <= tol))


def map2cl(t_map, lmax):
    return alm2cl(map2alm(t_map, lmax=lmax))


def get_fname(path):
    return '.'.join(path.split(os.sep)[-1].split('.')[:-1])


def get_ext(path):
    return path.split('.')[-1]


def bounds_check(ipos, shape):
    return np.all(ipos[0] >= 0) and np.all(ipos[0] < shape[0]) and np.all(ipos[1] >= 0) \
           and np.all(ipos[1] < shape[1])


# gives a mask of in-bounds galaxies
def in_bounds(ipos, shape):
    return (ipos[0] >= 0) * (ipos[0] < shape[0]) * (ipos[1] >= 0) * (ipos[1] < shape[1])


def iround(f_ind):
    sgn = np.sign(f_ind).astype(int)
    ind_abs = (np.abs(f_ind) + 0.5).astype(int)
    return sgn * ind_abs


def masked_inv(ar):
    zeros = ar == 0.

    ret = ar.copy()
    ret[zeros] = 1.
    ret = 1./ret
    ret[zeros] = 0.

    return ret


def parse_act_beam(path):
    line = None
    with open(path, 'r') as f:
        lines = f.readlines()

    ret = []
    for line in lines:
        l = int(line[:6])
        amp = float(line[6:])
        ret.append((l,amp))

    ar = np.array(ret)
    return ar[:,0], ar[:,1]


# def run_from_config(path):
#     res = None
#     with f as open(path, 'r'):
#         res = yaml.safe_load(f)

#     assert res is not None

#     scripts = res.keys()

#     for script, value in res.items()
#         assert script in listdir('scripts')