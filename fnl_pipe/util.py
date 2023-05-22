import time

import mpi4py

import matplotlib.pyplot as plt
import numpy as np

import os
from os import path, listdir

import yaml

from pixell.curvedsky import alm2cl, map2alm
from pixell import enplot

import h5py


# buffers a transposed chunk; fixed size
class ChunkedTransposeWriter:
    def _init_file(self):
        with h5py.File(self.path, 'w') as f:
            f.attrs['complete'] = False
            f.attrs['iwrite'] = 0
            f.create_dataset(self.bufname, (self.nrow, self.ncol),
                             dtype=self.dtype)

    def __init__(self, path, * , chunk_size, nrow, ncol, bufname='array'):
        self.path = path
        self.bufname = bufname

        self.chunk_size = chunk_size
        self.nrow = nrow
        self.ncol = ncol
        self.dtype = 'f4' # fixed

        self.buf = np.empty((chunk_size, nrow), dtype=np.float64)

        self.iwrite = 0
        self.iwrite_chunk = 0
        self.ichunk = 0

        self.complete = False

        self._init_file()

    def _check_integrity(self, h5file):
        assert h5file.attrs['iwrite'] == self.ichunk * self.chunk_size
        assert h5file.attrs['complete'] == False

    def _flush_to_disk(self, h5file):
        h5file.flush()

    def add_row(self, row):
        assert not self.complete
        with h5py.File(self.path, 'a') as f:
            self._check_integrity(f)

            self.buf[self.iwrite_chunk] = row

            self.iwrite_chunk +=1
            self.iwrite += 1

            if self.iwrite_chunk == self.chunk_size:
                dset = f[self.bufname]
                i0_dset = self.ichunk * self.chunk_size

                dset[:, i0_dset:i0_dset + self.chunk_size] = self.buf.T
                f.attrs['iwrite'] += self.chunk_size

                self._flush_to_disk(f)
                self.iwrite_chunk = 0
                self.ichunk += 1

            if self.iwrite == self.ncol:
                self._finalize_open(f)

    def _finalize_open(self, h5file):
        h5file.attrs['complete'] = True
        self._flush_to_disk(h5file)
        self.complete = True

    def finalize(self):
        with h5py.File(self.path, 'a') as f:
            self._finalize_open(f)


# probably quite slow!
# could implement next-chunk async precaching
# need to avoid python indexing
class ChunkedMaskedReader:
    def _init_file(self):
        with h5py.File(self.path, 'r') as h5file:
            assert h5file.attrs['complete'] == True
            dset = h5file[self.bufname]
            self.ncol = dset.shape[1]
            self.nrow = dset.shape[0]

            assert len(self.row_mask) == dset.shape[0]
            assert dset.dtype == 'f'

            self.row_inds = np.arange(self.nrow)[self.row_mask]
            self.ngal_in = len(self.row_inds)
            assert self.ngal_in == self.row_mask.sum()

            # print(f'ChunkedMaskedReader: expected memory footprint {(self.dset_buffer_factor + 1) * self.chunk_size * self.ncol * 8 * 1e-9 :.3f} GB')

            # this buffer holds the chunk data to be read in get_row (inbounds, in-cut)
            self.buf = np.empty((self.chunk_size, self.ncol),dtype=np.float64)

    def _get_chunk(self, ichunk):
        assert ichunk < self.nchunk
        # if ichunk == self.nchunk - 1: self.has_next_chunk = False; print('last chunk')

        with h5py.File(self.path, 'r') as h5file:
            dset = h5file[self.bufname]
            
            i0 = ichunk * self.chunk_size

            if ichunk < self.nchunk - 1:
                chunk_size = self.chunk_size
            else:
                chunk_size = self.last_chunk_size
                # print(f'last chunk! size {chunk_size}')

            imax = i0 + chunk_size

            n_this_chunk = chunk_size
            # print(f'n_this_chunk {n_this_chunk} chunk_size {self.chunk_size}')
            # print(f'i0 {i0}, imax {imax}, len(row_inds) {len(self.row_inds)}')
            chunk_inds = self.row_inds[i0:imax]
            # chunk_mask = np.zeros((self.nrow, self.ncol), dtype=bool)
            # chunk_mask[chunk_inds] = True

            # fill temporary buffer
            nfill_buf = 0
            idset = chunk_inds[0]
            ibuf = 0

            while(nfill_buf < n_this_chunk):
                this_nfill_dset = min(self.nbuf_dset, self.nrow - idset)
                this_mask = self.row_mask[idset:idset + this_nfill_dset]

                this_nfill_buf = this_mask.sum()

                if ibuf + this_nfill_buf > chunk_size:
                    # print('edge case')
                    this_nfill_buf = chunk_size - ibuf
                    ind_max = chunk_inds[-1]

                    this_nfill_dset = ind_max - idset + 1
                    this_mask = self.row_mask[idset:idset + this_nfill_dset]
                    # print(this_nfill_buf, this_mask.sum())
                    assert this_nfill_buf == this_mask.sum()

                # print('ichunk ibuf nfill_buff, n_this_chunk, this_nfill_buf, this_nfill_dset')
                # print(self.ichunk, ibuf, nfill_buf, n_this_chunk, this_nfill_buf, this_nfill_dset)
                # print(this_nfill_buf, this_nfill_dset, self.nbuf_dset, self.nrow - idset, ibuf + this_nfill_buf, self.chunk_size)
                nfill_buf += this_nfill_buf # the number of galaxies in this subregion
                # size_gb = this_nfill_dset * self.ncol * 8 * 1e-9
                # print(f'attempting to read {this_nfill_dset} rows from dset')

                # t0 = time.time()
                # tmp = np.array(dset[idset:idset + this_nfill_dset])
                # self.buf[ibuf:ibuf + this_nfill_buf] = tmp[this_mask]
                self.buf[ibuf:ibuf + this_nfill_buf] = dset[idset:idset + this_nfill_dset][this_mask]
                # del tmp
                # dt = time.time() - t0
                # print(f'finished; elapsed time: {dt:.2e} s, rate: {size_gb / dt:.3e} GB/s')

                idset += this_nfill_dset
                ibuf += this_nfill_buf

            assert nfill_buf == n_this_chunk

            # self.buf[:n_this_chunk] = dset[chunk_mask].reshape((n_this_chunk, self.ncol))
            self.chunk_inds = [i0, imax]
            self.n_this_chunk = n_this_chunk
            self.nread += nfill_buf
            self.ichunk = ichunk

    def __init__(self, path, chunk_size, row_mask, bufname='array', dset_buffer_factor=50):
        self.path = path
        self.bufname = bufname
        self.chunk_size = chunk_size
        self.row_mask = row_mask
        assert dset_buffer_factor >= 1 and type(dset_buffer_factor) is int
        self.dset_buffer_factor = dset_buffer_factor
        self.nbuf_dset = self.chunk_size * self.dset_buffer_factor

        self._init_file()

        self.nread = 0
        self.ichunk = 0
        self.nchunk = self.ngal_in // self.chunk_size + (self.ngal_in % self.chunk_size != 0)
        # print(f'ChunkedMaskedReader: nchunk {self.nchunk}')
        if self.ngal_in % self.chunk_size == 0:
            self.last_chunk_size = self.chunk_size
        else:
            self.last_chunk_size = self.ngal_in % self.chunk_size

        self.has_next_chunk = True
        self._get_chunk(self.ichunk)

    def get_next_chunk(self):
        ret = self.buf.copy()
        ret_inds = self.chunk_inds.copy()
        n_this_chunk = self.n_this_chunk

        if self.ichunk + 1 < self.nchunk:
            # print('ChunkedMaskedReader: getting next chunk')
            self._get_chunk(self.ichunk + 1)
        else:
            assert self.nread == self.ngal_in
            self.has_next_chunk = False

        return ret_inds, ret[:n_this_chunk]

    # def get_row(self, ind0, ind1):
    #     assert ind0 < ind1
    #     assert ind0 < self.ngal_in
    #     assert ind1 <= self.ngal_in
    #     ichunk0 = ind0 // self.chunk_size
    #     chunk_ind0 = ind0 % self.chunk_size
    #     ichunk1 = ind1 // self.chunk_size
    #     chunk_ind1 = ind1 % self.chunk_size

    #     if ichunk0 != self.ichunk:
    #         # print(f'getting new chunk {ichunk}, current chunk {self.ichunk}')
    #         self._get_chunk(ichunk0, ichunk1)

    #     return self.buf[chunk_ind]


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


def get_label(base_path, title, replace=False):
    suffix=''
    if not replace: suffix = '_' + get_sequential_label(base_path, title)
    label = title + suffix

    return str(label)


# Manages output directory. Not parallel
# logs is a list of tuples (logname, logpath)
class OutputManager:
    def __init__(self, title, base_path='output', subdirs=['plots', 'logs'], logs=None, replace=False,
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
            if self.mpi_rank == 0:
                label = get_label(base_path, title, replace)
                ensure_dir(path.join(base_path, label))

            label = mpi_comm.bcast(label, root=0)
            label += f'/rank_{self.mpi_rank}/'
            ensure_dir(path.join(base_path, str(label)))
        else:
            label = get_label(base_path, title, replace)

        self.working_dir = path.join(base_path, label)
        ensure_dir(path.join(base_path, label))

        # generate subdirs
        for sd in subdirs:
            this_path = path.join(self.working_dir, sd)
            ensure_dir(this_path)

        if logs is None:
            logs = [title,]
        
        self.log_names = logs
        self.logs = {log:LogFile(os.path.join(self.working_dir, 'logs', log + '.log')) for log in logs}
        self.default_log = 'log'

    def set_default_log(self, log=None):
        if log is None: log = 'log'
        
        assert log in self.log_names
        self.default_log = log

    def log(self, line, log=None):
        assert log in self.log_names or log is None

        if line[-1] != '\n' or line[-1] != '\r':
            line += '\n'

        if log is not None:
            self.logs[log].write(line)
        else:
            self.logs[self.default_log].write(line)
            # for logname, log in self.logs.items():
            #     log.write(line)

    def printlog(self, line, rank=None, *, log=None):
        line = str(line)
        if rank is not None:
            line = f'rank {rank}: ' + line

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


########## YAML config file stuff ###########

def update_locals(local_dict, param_dict):
    for key, value in param_dict.items():
        local_dict[key] = value


def validate_pass(yaml_dict):
    pass


def validate_config(yaml_dict, supported_scripts):
    assert 'base_path' in yaml_dict.keys()
    assert 'tmp_dir' in yaml_dict.keys()
    assert 'scripts' in yaml_dict.keys()

    scripts = yaml_dict['scripts']
    for script in scripts.keys():
        assert script in supported_scripts


def validate_script_config(yaml_dict):
    assert 'base_path' in yaml_dict.keys()


def get_yaml_dict(path, vad_fun=validate_pass):
    ret = None
    with open(path, 'r') as file:
        ret = yaml.safe_load(file)
        vad_fun(ret)

    return ret

########## End YAML config file stuff ###########
