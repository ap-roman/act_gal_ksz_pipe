import time

import matplotlib.pyplot as plt

import os
from os import path, listdir

import yaml


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


# Manages output directory. Not parallel
class OutputManager:
    def __init__(self, title, base_path='output', subdirs=['plots', 'logs']):
        assert title != ''

        self.base_path = base_path
        self.subdirs = subdirs

        label = get_sequential_label(base_path, title)
        label = title + '_' + label

        self.working_dir = path.join(base_path, str(label))
        ensure_dir(path.join(base_path, str(label)))

        # generate subdirs
        for sd in subdirs:
            this_path = path.join(self.working_dir, sd)
            ensure_dir(this_path)

    def savefig(self, name):
        assert('plots' in self.subdirs)

        plot_path = path.join(self.working_dir, 'plots', name)
        plt.savefig(plot_path)


def import_config(path):
    res = None

    with open(path, 'r') as f:
        res = yaml.safe_load(f)

    return res


# def run_from_config(path):
#     res = None
#     with f as open(path, 'r'):
#         res = yaml.safe_load(f)

#     assert res is not None

#     scripts = res.keys()

#     for script, value in res.items()
#         assert script in listdir('scripts')