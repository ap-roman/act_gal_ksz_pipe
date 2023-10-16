# This module defines classes and functions relevant to gal x CMB operations.
# For example, classes that mix galaxy and CMB data are defined here.

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

from pixell.curvedsky import map2alm, almxfl, alm2map, alm2cl, rand_map
from pixell import enmap, enplot

from fnl_pipe.util import map2cl, ChunkedTransposeWriter, average_fl, matmul, get_size_alm, masked_inv, fequal, get_dirs

from mpi4py import MPI
import hashlib
import pickle
import os
from os import path, listdir

# # an enmap hash class
# class MapHash:
#     def __init__(self, map_t):
#         m = hashlib.sha256()
#         m.update(map_t)
#         self._val = m.hexdigest() 

#     def __eq__(self, b):
#         return isinstance(b, MapHash) and self._val == b._val


def get_hash(obj):
    m = hashlib.sha256()
    m.update(obj)
    return m.hexdigest()


# A CMB x Gal hash class
# This class can be used to ensure that two CMB x Gal pipelines will produce
# statistically equvialent results i.e. that the transfer functions and noise realizations
# should agree to sampling error. Note that this is not a guarantee; the iterative transfer
# function can still fail to converge.
#
# For multiple frequencies, there are edge-cases in which two pipes may be statistically equivalent
# without having equal hashes; this may occur if one frequency's weights are near-zero.
#
# Right now this only works with a single-frequency cmb pipe
class CMBxGalHash:
    def __init__(self, *, cmb_pipe, gal_pipe, gal_mask):
        self._hashes = {}
        # self._hashor = {} # this is used for the lweight, which has looser agreement criteria

        self._cmb_pipe = cmb_pipe
        self._gal_pipe = gal_pipe
        self._gal_mask = gal_mask

        self.update()

    def __eq__(self, b):
        return isinstance(b, CMBxGalHash) and self._hashes == b._hashes and fequal(self.lweight, b.lweight, tol=1e-6)

    def _self_check(self):
        assert self._cmb_pipe.init_data
        assert self._cmb_pipe.init_fkp_f
        assert self._cmb_pipe.init_lweight_f
        assert self._gal_pipe.init_lists

    def update(self):
        self._self_check()
        cmb_pipe = self._cmb_pipe
        gal_pipe = self._gal_pipe
        gal_mask = self._gal_mask
        hashes = self._hashes

        hashes['map_t'] = get_hash(cmb_pipe.map_t)
        hashes['ivar_t'] = get_hash(cmb_pipe.ivar_t)
        hashes['fkp'] = get_hash(cmb_pipe.fkp)
        # hashes['lweight'] = get_hash(cmb_pipe.l_weight)

        self.lweight = np.array(cmb_pipe.l_weight).copy()

        hashes['gal_inds'] = get_hash(np.concatenate(gal_pipe.gal_inds))
        hashes['vrs'] = get_hash(gal_pipe.vrs)

        hashes['gal_mask'] = get_hash(gal_mask)

    def save(self, outpath):
        with open(outpath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, inpath):
        with open(inpath, 'rb') as file:
            ret = pickle.load(file)
            assert isinstance(ret, cls)
            return ret


# Data products state/directory interface
class DataProducts:
    def __init__(self, dp_path):
        self.has_nl = False
        self.has_fl = False
        self.has_mc_list = False

        self.update_parent(dp_path)

    def _check_nl(self, dp_path):
        nl_path = dp_path + os.sep + 'nl'
        ret = path.exists(nl_path)
        nl_files = get_files(nl_path)

        assert len(nl_files) == 1 and get_ext(nl_files[0]) == '.'

        return ret

    def update_parent(self, dp_path):
        self.has_nl = self.has_fl = self.has_mc_list = False

        assert path.exists(dp_path) and path.isdir(dp_path)

        if path.exists(dp_path + os.sep + 'nl'): self.has_nl = True
        if path.exists(dp_path + os.sep + 'fl'): self.has_fl = True
        
# This class handles updates to the noise cl function and the transfer function
# It ensures that only compatible maps and nl/fl functions can be used to analyze data
#
# This class has three use-cases:
#  - write new nl/fl/mc data into a clean file
#  - update onr or more products in an existing file
#  - read nl/fl/mc data and conduct a compatibile analysis
class DataProductsManager:
    def __init__(self):
        self.pipe_hash = None
        self.has_pipes = False

        # data products state
        self.dp_state = None

    # def register_pipes(self, cmb_pipe, gal_pipe, gal_mask):
    #     self.pipe_hash = CMBxGalHash(cmb_pipe=cmb_pipe, gal_pipe=gal_pipe, gal_mask=gal_mask)
    
    #     self.has_pipes = true

    # takes the path to a valid products folder as an input
    def register_products(self, prod_path):
        dp_state = DPState()
        dp_state.update_parent(prod_path)

        self.dp_state = dp_state

    # verifies that a data consumer pipeline is compatible with the data products
    def register_pipes(self, *, cmb_pipe, gal_pipe, gal_mask):
        assert self.dp_state

        pipe_hash = CMBxGalHash(cmb_pipe=cmb_pipe, gal_pipe=gal_pipe, gal_mask=gal_mask)
        assert pipe_hash == self.dp_state.hash


def _write_nl(nl_path, nl_tilde, var_scale, **kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs['nl_tilde'] = nl_tilde
    kwargs['var_scale'] = var_scale
    np.save(nl_path, kwargs)


def _write_xl(xl_path, xl, **kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs['xl'] = xl
    np.save(xl_path, kwargs)


def _write_mc_list(mc_path, mc_list, **kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs['mc_list'] = mc_list
    np.save(mc_path, kwargs)


class MFMetadata:
    # A metadata class for the multifrequency analysis
    # sigma0: the zero-variance offset in the denominator of the shared FKP weighting
    # r_lwidth: 5500. * r_lwidth is the l-width of the FKP "theory" function i.e.
    # cl_th = exp(-(l / (5500. * r_lwidth))**2)
    def __init__(self, sigma0, r_lwidth):
        self.sigma0 = sigma0
        self.r_lwidth = r_lwidth

        self.lwidth = 5500. * self.r_lwidth

    def __eq__(self, b):
        return self.sigma0 == b.sigma0 and self.r_lwidth == b.r_lwidth

    def get_lwidth(self):
        return self.lwidth



# fl maps in -> output power spectra
# xl maps out -> input 
# xl = 1./fl, where division is defined
class TransferFunction:
    def __init__(self, nave_fl=1):
        assert nave_fl > 0

        self.init = False
        self.nave_fl = nave_fl 

    # init via fl
    def init_xl(self, xl):
        self.xl = average_fl(xl, self.nave_fl)
        self.xl[0] = 0.

        self.init + True

    # init via cl
    def init_cl(self, cl_in, cl_out):
        assert len(cl_in) == len(cl_out)
        in_mask = cl_in == 0.
        out_mask = cl_out == 0.

        # xl = cl_in / cl_out
        cl_out_inv = masked_inv(cl_out)
        self.xl = cl_in * cl_out_inv
        self.xl[in_mask] = 0.

        self.init = True

    def get_cl_in(cl_out):
        assert self.init
        return self.xl * cl_out


# # This class transforms a map to a pseudo-map via 
# class MapXform:
#     def __init__(self, pre_masks=None, l_weight=None, post_masks=None):

#     def process_map(self, map_t):
#         return 


# class CltzXform:

# # the l-weighted
# class EstimatorXform:
#     def __init__(self,):


# A class that estimates an input power spectrum
# that yields a certain target output power specturm
# class TransferEstimator:
#     def __init__(self):

#     def get_cl_in(self):
#         pass # subclasses implement

# class CltzXlEstimator(TransferEstimator):
    


# # a class to handle the pseudo transforms,
# # transfer functions, and noise model
# class PseudoXform:
#     def __init__(self, process_map)
#         self.process_map = process_map

#     def


# A class to manage the functions used to transform the data
# from full sky to the desired representaiton
# def NlClFun:
#     def __init__(self,)

#     def map2nl(self, rand_map):



# def 


# This class contains the details of a particular set of
# process xforms
class TildeXform:
    def __init__(self, cmb_spec):
        self.cspec = cmb_spec
        self.lmax = cmb_spec.lmax

    # subclasses implement
    def process_map(self, map_t):
        pass

    def map2cl(self, map_t):
        return map2cl(self.process_map(map_t), lmax=self.lmax)

class LWeightTildeXform(TildeXform):
    def __init__(self, map_weight, lweight, cmb_spec, post_mask=None, freq=None, om=None, plots=True):
        self.map_weight = map_weight
        self.lweight = lweight
        self.post_mask = post_mask
        self.om = om
        self.freq = freq
        self.plots = plots

        super().__init__(cmb_spec)

    def process_map(self, map_t):
        printlog = self.om.printlog
        plots = self.plots and self.om is not None
        lweight = self.lweight

        map_fkp = self.map_weight * map_t

        if self.lweight is not None:
            # printlog('process_map: using nontrivial lweight')
            t_hp_alm = almxfl(map2alm(map_fkp, lmax=self.lmax), lfilter=lweight)
            t_hp = alm2map(t_hp_alm, self.cspec.make_zero_map())
        else:
            # printlog('process_map: skipping lweight')
            t_hp = map_fkp

        if self.post_mask is not None:
            # printlog('process_map: applying post-mask')
            ret = t_hp * self.post_mask
        else:
            # printlog('process_map: not applying post-mask')
            ret = t_hp

        if plots:
            fig = enplot.plot(enmap.downgrade(t_hp, 16), ticks=15, colorbar=True)
            self.om.savefig(f't_hp_{self.freq}', mode='pixell', fig=fig)
            
            fig = enplot.plot(enmap.downgrade(ret, 16), ticks=15, colorbar=True)
            self.om.savefig(f't_hp_gal_masked_{self.freq}', mode='pixell', fig=fig)

        return ret


# a tilde xform without l-weighting
class FlatTildeXform(LWeightTildeXform):
    def __init__(self, map_weight, cmb_spec, post_mask=None, freq=None, om=None, plots=True):
        super().__init__(map_weight, None, cmb_spec, post_mask, freq, om, plots)


# The intended workflow for a SimModel is
# to 
# 1) make the sim map via make_sim_map,
# 2) process the map via a tilde xform's process_map function
# 3) back
class SimModel:
    def __init__(self, om=None):
        if om is not None: self.printlog = om.printlog
        else: self.printlog = lambda x: None

    def add_noise_map(self, map_t):
        pass

    def make_sim_map(self):
        pass


# TODO: generalize these cl_mpi functions!
# def do_mpi_cl_loop(cl_fun, ntrial, comm):
#     assert ntrial % comm.size == 0


class KSimModel(SimModel):
    # This is a three-parameter noise model; 
    # var = B * var_act (B rescales map variance)
    # A foreground term is given by 
    # cl_fg = A (ells_trunc / 3000)^alpha
    # where ells_trunc = np.maximum(ells, 1500)
    def __init__(self, A, alpha, B, cl_cmb_lensed, std_map, cmb_spec, om=None):
        super().__init__(om=om)
        assert len(cl_cmb_lensed) == cmb_spec.lmax + 1
        self.printlog(f'received noise model with params A: {A:.3e} B: {B:.3e} alpha: {alpha:.3e}')
        self.A = A
        self.B = B
        std_scale = np.sqrt(B)
        self.printlog(f'std_scale: {std_scale}')
        self.std_map_rescale = std_scale * std_map 
        self.alpha = alpha
        self.cl_cmb_beamed = cl_cmb_lensed * cmb_spec.bl2

        self.cspec = cmb_spec

        ells = np.arange(cmb_spec.lmax + 1)
        ells_trunc = np.maximum(ells, 1500).astype(float)
        self.cl_fg_beamed = A * np.power(ells_trunc / 3000, alpha) * cmb_spec.bl2
        self.cl_combined_beamed = self.cl_cmb_beamed + self.cl_fg_beamed

    def add_noise_map(self, map_t):
        map_t += enmap.ndmap(np.random.normal(size=self.cspec.shape) * self.std_map_rescale, 
                               self.cspec.wcs)

    def make_noise_map(self):
        ret_map = self.cspec.make_zero_map()
        self.add_noise_map(ret_map)
        return ret_map

    def add_fg_map(self, map_t):
        map_t += self.cspec.rand_map(self.cl_fg_beamed)

    def make_fg_map(self):
        ret_map = self.cspec.make_zero_map()
        self.add_fg_map(ret_map)
        return ret_map

    def add_cmb_map(self, map_t):
        map_t += self.cspec.rand_map(self.cl_cmb_beamed)

    def make_cmb_map(self):
        ret_map = self.cspec.make_zero_map()
        self.add_cmb_map(ret_map)
        return ret_map

    def add_cls_map(self, map_t):
        map_t += self.cspec.rand_map(self.cl_combined_beamed)

    def make_sim_map(self, add_noise=True):
        ret_map = self.cspec.make_zero_map()
        # self.add_fg_map(ret_map)
        # self.add_cmb_map(ret_map)
        self.add_cls_map(ret_map)
        if add_noise: self.add_noise_map(ret_map)
        return ret_map


# A tiny class that combines the features of a tilde xform model and a sim model
# The point is to provide sim maps on demand and perhaps a few more features down the road
class TildeSimModel:
    def __init__(self, tilde_xform, sim_model, om=None):
        self.tilde_xform = tilde_xform
        self.cspec = tilde_xform.cspec
        self.lmax = self.cspec.lmax
        self.sim_model = sim_model

        if om is not None: self.printlog = om.printlog
        else: self.printlog = lambda x: None

    def make_sim_tilde_map(self,):
        return self.tilde_xform.process_map(self.sim_model.make_sim_map())

    def make_sim_cl_single(self, add_noise):
        return self.tilde_xform.map2cl(self.sim_model.make_sim_map(add_noise=add_noise))

    def _make_cl_sim_loop(self, ntrial, add_noise):
        sim_cl = np.zeros(self.lmax + 1)
        t0 = time.time()
        for itrial in range(ntrial):
            sim_cl += self.make_sim_cl_single(add_noise=add_noise)
            self.printlog(f'make_cl_sim: completed trial {itrial + 1} of {ntrial}, time per iter: {(time.time() - t0) / (itrial + 1)}')
        return sim_cl / ntrial

    def _make_cl_sim_mpi(self, ntrial, comm, add_noise):
        assert ntrial % comm.size == 0
        my_ntrial = ntrial // comm.size

        my_sim_cl = self._make_cl_sim_loop(my_ntrial, add_noise)

        lmax = self.lmax

        recvbuf = None
        if comm.rank == 0:
            recvbuf = np.empty((comm.size, lmax + 1))

        comm.Gather(my_sim_cl, recvbuf, root=0)
        if comm.rank == 0:
            sim_cl = recvbuf.mean(axis=0)
        else:
            sim_cl = np.empty(lmax + 1)

        comm.Bcast(sim_cl)
        return sim_cl

    # verify cl agreement
    def make_cl_sim(self, ntrial=1, comm=None, *, add_noise=True):
        if comm is None:
            return self._make_cl_sim_loop(ntrial, add_noise)
        else:
            return self._make_cl_sim_mpi(ntrial, comm, add_noise)


def fit_flat_nl(nl):
    lmax = len(nl) - 1
    ells = np.arange(lmax + 1)
    n_ell = 2 * ells + 1

    ivar_ell_weight = n_ell / n_ell.sum()
    nl_best = (ivar_ell_weight * nl).sum()

    nl_use = np.repeat(nl_best, lmax + 1)

    return nl_use


class CMBxGalPipe:
    def __init__(self, cmb_pipe, galaxy_pipe, tilde_sim, *, rand_pipe=None, output_manager=None):
        self.cmb_pipe = cmb_pipe
        self.galaxy_pipe = galaxy_pipe
        self.rand_pipe = rand_pipe

        self.tilde_sim = tilde_sim

        if output_manager is None:
            output_manager = cmb_pipe.output_manager
        self.output_manager = output_manager

        self.init_nl = False
        self.init_xl = False
        self.init_t_hp = False
        self.init_z_map_f = False

        self.alpha_mc = None

    def make_cl_tg(self, ntrial, comm):
        pass

    def _make_cl_tz(self, map_t):
        assert self.init_z_map_f
        lmax = self.cmb_pipe.lmax
        t_alm = map2alm(map_t, lmax=lmax)
        return alm2cl(t_alm, self.z_alm)

    def make_cl_tz_sig_new(self, map_t=None):
        if map_t is None:
            map_t = self.cmb_pipe.map_t

        lmax = self.cmb_pipe.lmax
        pmap = self.tilde_sim.tilde_xform.process_map
        return self._make_cl_tz(pmap(map_t))

    def make_cl_tz_mc_inner(self, ntrial):
        lmax = self.cmb_pipe.lmax
        make_sim_map = self.tilde_sim.make_sim_tilde_map
        pmap = self.tilde_sim.tilde_xform.process_map

        cl_tz = np.empty((ntrial, lmax + 1))

        t0 = time.time()
        for itrial in range(ntrial):
            cl_tz[itrial] = self._make_cl_tz(pmap(make_sim_map()))

        return cl_tz

    def make_cl_tz_mc_mpi_new(self, ntrial_mc, comm):
        assert ntrial_mc % comm.size == 0

        my_ntrial = ntrial_mc // comm.size

        printlog = self.output_manager.printlog
        rank = comm.rank
        size = comm.size

        lmax = self.cmb_pipe.lmax
        make_sim_map = self.tilde_sim.sim_model.make_sim_map
        pmap = self.tilde_sim.tilde_xform.process_map

        my_cl_tz = np.empty((my_ntrial, lmax + 1))

        t0 = time.time()
        for itrial in range(my_ntrial):
            my_cl_tz[itrial] = self._make_cl_tz(pmap(make_sim_map()))
            printlog(f'node {MPI.Get_processor_name()}: completed mc trial iteration {itrial} of {my_ntrial}, time per iter: {(time.time() - t0)/(itrial + 1):.3e}', rank=rank)

        cl_tz = None
        if rank == 0:
            cl_tz = np.empty((ntrial_mc, lmax + 1))
        
        comm.Gather(my_cl_tz, cl_tz, root=0)
        return cl_tz


    # def make_cl_tz_mc_new(self, ntrial_mc, comm=None):
    #     if comm is None:


    # Completely process a map as per the new prescription
    # Here we take an input ACT map, sdss completeness map, fkp + foreground weight/mask,
    # and optimal l-weighting
    #
    # We compute the processed mask as follows:
    # (sky) T_act --*fkp_map--> T_fkp --map2alm--> T_fkp_lm --*l_weight-->
    # T_hp_lm --alm2map--> T_hp --*sdss_mask--> T_preprocessed
    # 
    # in general the final step can be streamlined by extracting only the 
    # indices of T_hp and sdss_mask that correspond to galaxy locations present
    # in the galaxy survey (galaxy_pipe)
    def process_map(self, map_t, plots=False):
        cmb_pipe = self.cmb_pipe
        gal_pipe = self.galaxy_pipe

        assert cmb_pipe.init_data and cmb_pipe.init_fkp_f and cmb_pipe.init_lweight_f
        assert gal_pipe.init_lists

        fkp = cmb_pipe.fkp
        l_weight = cmb_pipe.l_weight

        map_fkp = fkp * map_t

        t_hp_alm = almxfl(map2alm(map_fkp, lmax=cmb_pipe.lmax), lfilter=l_weight)
        t_hp = alm2map(t_hp_alm, cmb_pipe.make_zero_map())
        ret = t_hp * self.gal_mask

        if plots:
            fig = enplot.plot(enmap.downgrade(t_hp, 16), ticks=15, colorbar=True)
            self.output_manager.savefig(f't_hp_{cmb_pipe.freq}', mode='pixell', fig=fig)
            
            fig = enplot.plot(enmap.downgrade(ret, 16), ticks=15, colorbar=True)
            self.output_manager.savefig(f't_hp_gal_masked{cmb_pipe.freq}', mode='pixell', fig=fig)

        return ret

    # process_map, but apply only the fkp function and return the alms
    def process_map_fkp_alm(self, map_t):
        cmb_pipe = self.cmb_pipe
        gal_pipe = self.galaxy_pipe

        assert cmb_pipe.init_data and cmb_pipe.init_fkp_f
        assert gal_pipe.init_lists

        fkp = cmb_pipe.fkp

        return map2alm(map_t * fkp, lmax=cmb_pipe.lmax)

    # Make this a special case of init_t_hp?
    def make_t_hp_nomask(self, real_space=True):
        cmb_pipe = self.cmb_pipe
        assert cmb_pipe.init_data
        assert cmb_pipe.init_metadata
        assert cmb_pipe.init_fkp_f
        assert cmb_pipe.init_lweight_f

        t_alm = map2alm(cmb_pipe.fkp * cmb_pipe.map_t, lmax=cmb_pipe.lmax)
        t_hp_alm = almxfl(t_alm, lfilter=cmb_pipe.l_weight)

        if real_space: return alm2map(t_hp_alm, cmb_pipe.make_empty_map())
        else: return t_hp_alm

    # process the map the old way and look for ringing
    def process_map_wrong(self, map_t, plots=False):
        cmb_pipe = self.cmb_pipe
        gal_pipe = self.galaxy_pipe

        assert cmb_pipe.init_data and cmb_pipe.init_fkp_f and cmb_pipe.init_lweight_f
        assert gal_pipe.init_lists
        fkp = cmb_pipe.fkp * self.gal_mask # apply the galaxy survey completeness mask
        l_weight = cmb_pipe.l_weight

        map_fkp = fkp * map_t

        t_hp_alm = almxfl(map2alm(map_fkp, lmax=cmb_pipe.lmax), lfilter=l_weight)
        t_hp = alm2map(t_hp_alm, cmb_pipe.make_zero_map())

        if plots:
            fig = enplot.plot(enmap.downgrade(t_hp, 16), ticks=15, colorbar=True)
            self.output_manager.savefig(f't_hp_old_{cmb_pipe.freq}', mode='pixell', fig=fig)

        return t_hp

    def make_sim_map(self, cl_sim=None):
        cmb_pipe = self.cmb_pipe
        map_ref = cmb_pipe.map_t

        if cl_sim is None:
            assert self.init_xl
            cl_sim = self.cl_act_sim

        return rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_sim) + cmb_pipe.make_noise_map()

    # remove an lweight^2 factor
    def make_sim_map_cltz(self, cl_sim=None):
        cmb_pipe = self.cmb_pipe
        map_ref = cmb_pipe.map_t

        if cl_sim is None:
            assert self.init_xl
            cl_sim = self.cl_act_sim

        return rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_sim) + cmb_pipe.make_noise_map()

    def update_xl(self, tf, *, xl_path=None, **kwargs):
        assert self.init_nl
        assert tf.init
        xl = tf.xl
        assert len(xl) == self.cmb_pipe.lmax + 1

        if xl_path is not None:
            _write_xl(xl_path, xl, **kwargs)

        self.tf = tf

        # The noise pseudo spectrum should never exceed the data pseudo spectrum
        assert np.all(self.cl_act_tilde >= self.nl_tilde)

        self.cl_act_sim = (self.cl_act_tilde - self.nl_tilde) * xl
        self.init_xl = True

    # map -> process_map -> map2cl
    def map2pm2cl(self, map_t):
        return map2cl(self.process_map(map_t), lmax=self.cmb_pipe.lmax)

    # map -> process_map_fkp_alm -> alm2cl
    def map2pmcltz2cl(self, map_t):
        return alm2cl(self.process_map_fkp_alm(map_t))

    def full_cl_fun(self, cl_in):
        lmax = self.cmb_pipe.lmax
        map_ref = self.cmb_pipe.map_t
        return map2cl(self.process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_in)), lmax=lmax)

    def cltz_cl_fun(self, cl_in):
        lmax = self.cmb_pipe.lmax
        map_ref = self.cmb_pipe.map_t
        return alm2cl(self.process_map_fkp_alm(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_in)))
        
    # sim_cl_fun is the function used to generate simulated
    # sim_cl_fun takes cl_in as its sole input and returns the psuedo cl
    def _xl_loop(self, cl_in, ntrial, sim_cl_fun=None, rank=None):
        rankstr = ''
        if rank is not None:
            assert isinstance(rank, int)
            rankstr = f'rank {rank}: '

        lmax = self.cmb_pipe.lmax
        printlog = self.output_manager.printlog
        process_map = self.process_map
        map_ref = self.cmb_pipe.map_t

        if sim_cl_fun is None:
            sim_cl_fun = self.full_cl_fun
        else:
            printlog('_fl_loop: using custom sim_cl_fun')

        cl_out = np.zeros(lmax + 1)
        t0 = time.time()
        for itrial_fl in range(ntrial):
            printlog(rankstr + f'doing trial {itrial_fl} of {ntrial}')
            cl_out += sim_cl_fun(cl_in)
            # cl_out += map2cl(process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_in)), lmax=lmax)
            time_per_fl = (time.time() - t0) / (itrial_fl + 1)
            printlog(rankstr + f'time per fl eval: {time_per_fl:.3e}')
        cl_out /= ntrial

        return cl_out

    def _xl_loop_mpi(self, cl_in, ntrial, comm, sim_cl_fun=None):
        lmax = self.cmb_pipe.lmax

        rank = comm.Get_rank()
        size = comm.Get_size()

        assert ntrial % size == 0
        my_ntrial = ntrial // size

        my_cl_out = self._xl_loop(cl_in, my_ntrial, sim_cl_fun)

        print('got my_cl_out')
        if rank == 0:
            cl_out = np.zeros((size, lmax + 1))
        else: 
            cl_out = np.empty(lmax + 1)

        comm.Gather(my_cl_out, cl_out, root=0)
        if rank == 0:
            cl_out = cl_out.sum(axis=0) / size

        comm.Bcast(cl_out, root=0)

        return cl_out

    def make_fl_iter(self, *, sim_cl_fun=None, nave_fl=32, ntrial_fl=60, niter=4, fl_path=None, plots=False, comm=None):
        assert self.init_nl

        cmb_pipe = self.cmb_pipe
        ells = self.cmb_pipe.ltool.ells
        norm = cmb_pipe.ltool.norm
        lmax = self.cmb_pipe.lmax
        map_ref = self.cmb_pipe.map_t
        process_map = self.process_map
        nl_tilde = self.nl_tilde
        cl_act_tilde = self.cl_act_tilde
        lmin_cut = self.cmb_pipe.lmin_cut
        printlog = self.output_manager.printlog

        rank = None
        size = None
        do_mpi_fl = comm is not None

        if do_mpi_fl:   
            rank = comm.Get_rank()
            size = comm.Get_size()
            assert ntrial_fl % size == 0

        # assert (lmax % ntrial_fl) == 0 # safe?
        if sim_cl_fun is None:
            sim_cl_fun = self.full_cl_fun
        else: printlog('make_fl_iter: using custom sim fun')

        xl = np.ones(lmax + 1)
        tf = TransferFunction()
        tf.init_xl(xl)
        cl_target = cl_act_tilde - nl_tilde
        rms_err = np.zeros(niter)

        for iiter in range(niter):
            print(f'doing iter {iiter}')
            cl_in = cl_target * tf.xl

            if do_mpi_fl:
                print(f'doing mpi fl, rank: {rank}')
                cl_out = self._xl_loop_mpi(cl_in, ntrial_fl, comm, sim_cl_fun)
                print(f'finished _fl_loop_mpi, rank = {rank}')

                tf.init_cl(cl_in, cl_out)
            else:
                cl_out = self._xl_loop(cl_in, ntrial_fl, sim_cl_fun)

                tf.init_cl(cl_in, cl_out)

            xl = average_fl(tf.xl, nave_fl)
            xl[0] = 0.

            tf.init_xl(xl)

            if plots:
                plt.figure(dpi=300)
                plt.plot(ells[100:], xl[100:], label='xl')
                plt.xlabel(r'$l$')
                plt.ylabel(r'$f_l$')
                plt.yscale('log')
                self.output_manager.savefig(f'xl_iter_{iiter}.png')
                plt.close()

                plt.figure(dpi=300)
                plt.title(r'intermediate input $C_l$')
                plt.plot(ells[100:], (norm * (cl_target * xl))[100:])
                plt.xlabel(r'$l$')
                plt.ylabel(r'$C_ll(l+1)$ input')
                plt.yscale('log')
                self.output_manager.savefig(f'cl_sim_iter_{iiter}.png')
                plt.close()

            fit_goodness = np.sqrt(np.sum((cl_target[lmin_cut+1:] - cl_out[lmin_cut+1:])**2)/(lmax - lmin_cut))
            rms_err[iiter] = fit_goodness
            printlog(f'make_xfer_iter {iiter} of {niter}: rms cl error= {fit_goodness:.3e}')

        if plots:
            cl_in = cl_target * xl
            cl_out = np.zeros(lmax + 1)
            for itrial_fl in range(ntrial_fl):
                cl_out += sim_cl_fun(cl_in)
            cl_out /= ntrial_fl

            plt.figure(dpi=300)
            plt.title('rms error vs iteration')
            plt.yscale('log')
            plt.ylabel('rms error')
            plt.xlabel('iteration')
            plt.plot(rms_err)
            self.output_manager.savefig('xfer_error_iter.png', mode='matplotlib')
            plt.close()

            plt.figure(dpi=300)
            plt.plot(ells, norm * cl_target, label='cl_target')
            plt.plot(ells, norm * cl_out, label='cl_out')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\Delta_l$')
            plt.yscale('log')
            plt.legend()
            self.output_manager.savefig('cl_xfer_compare.png', mode='matplotlib')
            plt.close()

            cl_ave = 0.5 * (cl_target + cl_out)
            plt.figure(dpi=300)
            plt.plot(ells, (cl_out - cl_target) / cl_ave)
            plt.xlabel(r'$l$')
            plt.ylabel(r'relative error')
            # plt.ylim([-1,1])
            self.output_manager.savefig('cl_relative_error.png', mode='matplotlib')
            plt.close()

        self.update_xl(tf, xl_path=xl_path, ntrial_xl=ntrial_xl, niter_xl=niter,
                       nave_xl=nave_fl)

    def import_xl(self, xl_path):
        xl_dict = np.load(xl_path, allow_pickle=True)
        this_xl = xl_dict.item()['xl']
        var_scale = xl_dict.item()['var_scale']
        tf = TransferFunction()
        tf.init_xl(this_xl)
        self.update_xl(tf)

    # def make_fl_standard(self, ntrial_fl=1, nave_l=32, plots=False):
    #     cmb_pipe = self.cmb_pipe
    #     ells = self.cmb_pipe.ltool.ells
    #     norm = cmb_pipe.ltool.norm
    #     lmax = self.cmb_pipe.lmax
    #     map_ref = self.cmb_pipe.map_t
    #     process_map = self.process_map

    #     cl_ref = np.ones(lmax + 1)
        
    #     fl = np.zeros(lmax + 1) # base xfer function
    #     nl_tilde = np.zeros(lmax + 1)
    #     for itrial in range(ntrial_fl):
    #         t_proc = process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_ref))
    #         t_noise = process_map(cmb_pipe.make_noise_map())
    #         this_nl = map2cl(t_noise, lmax=lmax)
    #         cl_out = map2cl(t_proc, lmax=lmax)

    #         nl_tilde += this_nl
    #         fl += cl_out/cl_ref

    #     fl /= ntrial_fl
    #     nl_tilde /= ntrial_fl

    #     self._update_nl(nl_tilde)

    #     assert lmax % nave_l == 0
    #     fl_ave = np.repeat(fl[1:].reshape(lmax // nave_l, nave_l).sum(axis=-1)/nave_l, nave_l)

    #     self.fl_standard = np.ones(lmax + 1)
    #     self.fl_standard[1:] = fl_ave

    #     cl_cmb_test = (self.cl_tilde_act - self.nl_tilde) / self.fl_standard
    #     cl_out_test = map2cl(process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_cmb_test)), lmax=lmax)

    #     # this plot is a mandatory diagnostic
    #     plt.figure(dpi=300)
    #     plt.title('standard (diagonal) xfer function diagnostic')
    #     plt.plot(ells, norm * (self.cl_tilde_act - self.nl_tilde), label='act pseudo cl (input)')
    #     plt.plot(ells, norm * cl_out_test, label='random pseudo cl (output)')
    #     plt.ylabel(r'$\Delta_l$')
    #     plt.xlabel(r'$l$')
    #     plt.legend()
    #     self.output_manager.savefig(f'cl_fl_diagnostic_{cmb_pipe.freq}.png')
    #     plt.close()

    #     if plots:
    #         plt.figure(dpi=300)
    #         plt.plot(ells[1:], fl[1:], label='fl_fine')
    #         plt.plot(ells[1:], self.fl_standard[1:], label='fl_averaged')
    #         plt.legend()
    #         plt.ylabel('fl')
    #         plt.xlabel(r'$l$')
    #         self.output_manager.savefig(f'fl_nave_{nave_l}_ntrial_{ntrial_fl}.png')
    #         plt.close()

    #     self.init_fl = True

    def make_rand_nl(self):
        return map2cl(self.process_map(self.cmb_pipe.make_noise_map()), lmax=self.cmb_pipe.lmax)

    def make_rand_nl_cltz(self):
        return alm2cl(self.process_map_fkp_alm(self.cmb_pipe.make_noise_map()))

    def update_nl(self, nl_tilde, nl_fun=None, *, var_scale=1., nl_path=None, plots=True, **kwargs):
        assert len(nl_tilde) == self.cmb_pipe.lmax + 1

        if nl_fun is None:
            nl_fun = self.map2pm2cl

        if nl_path is not None:
            _write_nl(nl_path, nl_tilde, var_scale, **kwargs)

        self.nl_tilde = nl_tilde
        self.cl_act_tilde = nl_fun(self.cmb_pipe.map_t)
        self.cmb_pipe.update_var_scale(var_scale)
        self.init_nl = True

        if plots:
            om = self.output_manager
            ells = self.cmb_pipe.ltool.ells
            plt.figure(dpi=300, facecolor='w')
            plt.title(r'$\tilde{N}_l$')
            plt.plot(ells, nl_tilde, label='sim noise')
            plt.plot(ells[3001:], self.cl_act_tilde[3001:], label='act map (l>=3000)')
            plt.legend()
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\tilde{N}_l$')
            om.savefig(f'nl_tilde_f{self.cmb_pipe.freq}.png')
            plt.close()

    def _nl_loop_single(self, nl_fun, ntrial_nl):
        printlog = self.output_manager.printlog
        lmax = self.cmb_pipe.lmax

        t0 = time.time()
        nl_tilde = np.zeros(lmax + 1)
        for itrial in range(ntrial_nl):
            printlog(f'_nl_loop: doing nl trial {itrial + 1}')
            nl_tilde += nl_fun()
            tnow = time.time()
            printlog(f'time per iteration: {(tnow - t0)/(itrial + 1):.3e} s')
        nl_tilde /= ntrial_nl

        return nl_tilde

    def _nl_loop_mpi(self, nl_fun, ntrial_nl, comm):
        printlog = self.output_manager.printlog
        lmax = self.cmb_pipe.lmax

        rank = comm.rank
        size = comm.size
        assert ntrial_nl % size == 0
        my_ntrial_nl = ntrial_nl // size

        my_nl_tilde = self._nl_loop_single(nl_fun, my_ntrial_nl)

        recvbuf = None
        if rank == 0:
            recvbuf = np.empty((size, lmax + 1), dtype=np.float64)

        comm.Gather(my_nl_tilde, recvbuf, root=0)
        if rank == 0:
            nl_tilde = recvbuf.mean(axis=0)
        else:
            nl_tilde = np.empty(lmax + 1)

        comm.Bcast(nl_tilde)
        return nl_tilde

    # a seamless interface to the single process / mpi methods
    def _nl_loop(self, nl_fun, ntrial_nl, comm=None):
        if comm is None:
            return self._nl_loop_single(nl_fun, ntrial_nl)
        else:
            return self._nl_loop_mpi(nl_fun, ntrial_nl, comm)

    # make_nl inner loop that matches nl to the pseudo-(data) spectrum beyond a certain reference l
    # TODO: add var_scale tolerance factor?
    def make_nl_lmatch(self, nl_path=None, ntrial_nl=32, match_lowest=False, l_match=9000, n_ave=100,
                       n_iter=2, comm=None, plots=False, l_plot_cut=2000):
        om = self.output_manager
        printlog = om.printlog

        # TODO: compute and store cl_act_pseudo once before nl is set to avoid recomputing each iteration....
        # TODO: tie this in with the nl/cl/pseudo-chain object to manage various choices of
        # full sky -> windowed + ivar'd
        cl_act_pseudo = alm2cl(self.process_map_fkp_alm(self.cmb_pipe.map_t))

        cl_smoothed = average_fl(cl_act_pseudo, n_ave)
        l_min = np.argmin(cl_smoothed)
        printlog(f'cl_smoothed minimum: {l_min}')
        
        if match_lowest:
            l_match = l_min

        cl_match = cl_smoothed[l_match + 1]

        # compute nl_tilde
        nl_tilde = self._nl_loop(self.make_rand_nl_cltz, ntrial_nl, comm)

        # fit a flat spectrum
        nl_flat = fit_flat_nl(nl_tilde)

        var_scale = (cl_match / nl_flat[l_match + 1]) * self.cmb_pipe.var_scale # this is an array of constant value
        printlog(f'make_nl_lmatch: scaling map variance by {var_scale:.3e}')

        if plots:
            ells = self.cmb_pipe.ltool.ells
            norm = self.cmb_pipe.ltool.norm
            plt.figure(dpi=300, facecolor='w')
            plt.plot(ells[l_plot_cut:], (norm * cl_act_pseudo)[l_plot_cut:], label='cl_pseudo')
            plt.plot(ells[l_plot_cut:], (norm * nl_tilde)[l_plot_cut:], label='flat', alpha=0.35)
            plt.plot(ells[l_plot_cut:], (norm * nl_flat)[l_plot_cut:], label='flat')
            plt.xlabel('l')
            plt.ylabel('Delta_l')
            om.savefig(f'nl_niter_{n_iter}.png')

        if n_iter > 1:
            self.cmb_pipe.update_var_scale(var_scale)
            return self.make_nl_lmatch(nl_path, ntrial_nl, match_lowest, l_match, n_ave, n_iter - 1,
                                       comm, plots)
        else:
            # self.update_nl(nl_tilde, var_scale=var_scale, nl_path=nl_path, ntrial_nl=ntrial_nl)
            return nl_flat


    def make_nl(self, *, nl_fun=None, nl_flat=False, ntrial_nl=32, nl_path=None, plots=False):
        cmb_pipe = self.cmb_pipe
        lmax = self.cmb_pipe.lmax
        ells = self.cmb_pipe.ltool.ells
        norm = cmb_pipe.ltool.norm
        process_map = self.process_map
        printlog = self.output_manager.printlog

        if nl_fun is None:
            nl_fun = self.make_rand_nl

        nl_tilde = self._nl_loop(nl_fun, ntrial_nl)

        if nl_flat:
            n_ell = 2 * ells + 1

            ivar_ell_weight = n_ell / n_ell.sum()
            nl_best = (ivar_ell_weight * nl_tilde).sum()

            nl_tilde_use = np.repeat(nl_best, lmax + 1)
        else:
            nl_tilde_use = nl_tilde

        if plots:
            plt.figure(dpi=300)
            if nl_flat:
                plt.plot(ells, nl_tilde, label='nl_tilde raw')
            plt.plot(ells, nl_tilde_use, label='nl_tilde filtered')
            plt.legend()
            plt.xlabel(r'$l$')
            plt.ylabel(r'$N_l$')
            self.output_manager.savefig(f'nl_{self.cmb_pipe.freq}_ghz_{ntrial_nl}_ntrial.png', 
                                        mode='matplotlib')
            plt.close()

            if nl_flat:
                plt.figure(dpi=300)
                plt.plot(ells, ivar_ell_weight)
                plt.xlabel(r'$l$')
                plt.ylabel('ivar l-weight')
                self.output_manager.savefig(f'nl_ivar_weight{self.cmb_pipe.freq}_ghz_{ntrial_nl}_ntrial.png', 
                                            mode='matplotlib')
                plt.close()

        self.update_nl(nl_tilde_use, nl_path=nl_path, ntrial_nl=ntrial_nl)

    def import_nl(self, nl_path, nl_fun=None):
        nl_dict = np.load(nl_path, allow_pickle=True)
        nl_tilde = nl_dict.item()['nl_tilde']
        var_scale = nl_dict.item()['var_scale']
        self.update_nl(nl_tilde, var_scale=var_scale, nl_fun=nl_fun)

    # produces a (ngal, ntrail_mc) array of masked t_hp evaluated at galaxy locs 
    # useful to "freeze" a particular MC realization or to just save time
    # in the context of parameter optimazation.
    def write_mc_list(self, ntrial_mc=32, outpath=None):
        assert ntrial_mc > 0
        cmb_pipe = self.cmb_pipe
        lmax = cmb_pipe.lmax
        gal_pipe = self.galaxy_pipe
        gal_decs, gal_ras = gal_pipe.gal_inds
        ngal = gal_pipe.ngal_in
        vrs = gal_pipe.vr_list
        map_ref = self.cmb_pipe.map_t
        process_map = self.process_map
        cl_act_sim = self.cl_act_sim
        printlog = self.output_manager.printlog

        # t_mc_list = np.empty((ntrial_mc, gal_pipe.ngal_in), dtype=np.float64)

        cw = ChunkedTransposeWriter(outpath, chunk_size=32, nrow=ngal, ncol=ntrial_mc, bufname='t_mc')

        printlog(f'GalxCMBPipe: ngal_in: {gal_pipe.ngal_in}')
        printlog(f'GalxCMBPipe: t_mc_list size: {ntrial_mc * gal_pipe.ngal_in * 8 * 1e-9} GB')

        t0 = time.time()
        for itrial in range(ntrial_mc):
            t_hp_mc = process_map(self.make_sim_map())
            t_hp_list = t_hp_mc[gal_decs, gal_ras]
            
            cw.add_row(t_hp_list)
            
            dt_total = time.time() - t0
            dt_per_iter = dt_total / (itrial + 1)
            printlog(f'completed MC iteration {itrial}, time per iter: {dt_per_iter:.2e} s')
        cw.finalize()


        self.ntrial_mc_list = ntrial_mc
        # self.t_mc_list = t_mc_list.T
        assert cw.complete


        # if outpath is not None:
        #     _write_mc_list(outpath, self.t_mc_list)

    # def make_mc_list_mpi(self, ntrial_mc, rank, size, max_ram=None):
    #     assert ntrial_mc % size == 0

    #     if max_ram is not None:
    #         assert ntrial_mc 

    def _init_t_hp(self):
        self.t_hp = self.process_map(self.cmb_pipe.map_t) # the processed act map
        self.init_t_hp = True
    
    def init_z_map(self, map_weight=None):
        gal_pipe = self.galaxy_pipe
        printlog = self.output_manager.printlog

        if map_weight is None:
            v_offset = 0.
        else: 
            # compute the fkp weighted mean velocity
            fkp_list = gal_pipe.get_map_list(map_weight / map_weight.pixsizemap())
            v_offset = (fkp_list * gal_pipe.vrs).sum() / (fkp_list.sum())

            printlog(f'init_z_map: fkp weighted mean velocity is {v_offset:.3e}')


        self.z_map = self.galaxy_pipe.get_vr_map(v_offset=v_offset)

        # if self.rand_pipe is not None:
        #     scale = self.galaxy_pipe.ngal_in / self.rand_pipe.ngal_in
        #     self.z_map -= scale * self.rand_pipe.get_vr_map()

        self.z_alm = map2alm(self.z_map, lmax=self.cmb_pipe.lmax)
        self.init_z_map_f = True

    #@profile
    def compute_estimator(self, ntrial_mc=0, buffered_reader=None, v_shuffle=False, verbose=True):

        cmb_pipe = self.cmb_pipe
        lmax = cmb_pipe.lmax
        gal_pipe = self.galaxy_pipe
        gal_decs, gal_ras = gal_pipe.gal_inds
        vrs = gal_pipe.vr_list
        map_ref = self.cmb_pipe.map_t
        process_map = self.process_map
        printlog = self.output_manager.printlog

        if not self.init_t_hp:
            self._init_t_hp()
        t_hp = self.t_hp

        t_hp_list = t_hp[gal_decs, gal_ras]
        ngal_list = len(t_hp_list)

        vr_list = gal_pipe.vrs
        if v_shuffle: vr_list = vr_list[np.random.permutation(ngal_list)]

        # a_ksz_unnorm = gal_pipe.get_xz_list(t_hp).sum()
        a_ksz_unnorm = (vr_list * t_hp_list).sum()
        self.a_ksz_unnorm = a_ksz_unnorm # used by mpi version
        a_std_bootstrap = np.sqrt(gal_pipe.ngal_in * np.var(t_hp_list) * np.var(vrs))
        a_bootstrap = a_ksz_unnorm/a_std_bootstrap
        a_std_bootstrap_2 = np.sqrt(((vrs * t_hp_list)**2).sum())
        a_bootstrap_2 = a_ksz_unnorm/a_std_bootstrap_2

        if verbose: printlog(f'analytic estimator: {a_bootstrap:.3e}')
        if verbose: printlog(f'analytic estimator 2: {a_bootstrap_2:.3e}')

        ret = {'a_ksz_bootstrap': a_bootstrap, 'a_ksz_bootstrap_2': a_bootstrap_2}
        
        if ntrial_mc > 0:
            cl_act_sim = self.cl_act_sim
            assert self.init_nl
            assert self.init_xl
            alphas = np.zeros(ntrial_mc, dtype=np.float64)
            t0 = time.time()

            if buffered_reader is None:
                for itrial in range(ntrial_mc):
                    t_hp_mc = process_map(self.make_sim_map())
                    t_mc_list = t_hp_mc[gal_decs, gal_ras]

                    alphas[itrial] = (vrs * t_mc_list).sum()

                    dt_total = time.time() - t0
                    dt_per_iter = dt_total / (itrial + 1)
                    printlog(f'completed MC trial iteration, time per iter: {dt_per_iter:.2e} s')

                    if itrial > 1:
                        a_std_sofar = np.std(alphas[:itrial + 1])
                        printlog(f'a_ksz so far: {a_ksz_unnorm/a_std_sofar:.3e}')
            else:
                cut_inbounds_mask = gal_pipe.cut_inbounds_mask

                ngal = 0
                while ngal < gal_pipe.ngal_in:
                    printlog(f'getting next chunk')
                    gal_inds, t_hp_br = buffered_reader.get_next_chunk()
                    printlog(f'done')
                    ngal_this_chunk = t_hp.shape[0]
                    assert t_hp_br.shape[0] == gal_inds[1] - gal_inds[0]

                    alphas[:] += (vrs[gal_inds[0]:gal_inds[1], None] * t_hp_br).sum(axis=0)

                    dt_total = time.time() - t0
                    dt_per_iter = dt_total / (ngal + ngal_this_chunk)

                    dt_total_expected = gal_pipe.ngal_in * dt_per_iter
                    ngal += ngal_this_chunk
                    # printlog(f'completed {ngal} MC gals, time per iter: {dt_per_iter:.2e} s, expected total time: {dt_total_expected:.2e} s')

                assert not buffered_reader.has_next_chunk
                # TODO: This assert is failing for some reason!!!
                # assert ngal == gal_pipe.ngal_in, f'ngal: {ngal}, ngal_in: {gal_pipe.ngal_in}'
                dt_total = time.time() - t0
                printlog(f'completed entire galaxy mc loop in {dt_total:.2e} s')

            a_std_mc = np.std(alphas)
            a_ksz_mc = a_ksz_unnorm / a_std_mc
            ret['a_ksz_mc'] = a_ksz_mc
            self.alphas = alphas

        return ret

    # def make_mc_t_fkp_alm(self):
    #     return self.process_map_fkp_alm(self.make_sim_map())

    def make_cl_tz(self, map_t):
        assert self.init_xl
        assert self.init_z_map_f
        lmax = self.cmb_pipe.lmax

        t_alm = self.process_map_fkp_alm(map_t)
        return alm2cl(t_alm, self.z_alm)

    def make_cl_tz_mc(self):
        return self.make_cl_tz(self.make_sim_map_cltz())

    def make_cl_tz_mpi(self, my_ntrial_mc, ntrial_mc, comm):
        printlog = self.output_manager.printlog
        rank = comm.rank
        size = comm.size

        lmax = self.cmb_pipe.lmax

        my_cl_tz = np.empty((my_ntrial_mc, lmax + 1))

        t0 = time.time()
        for itrial in range(my_ntrial_mc):
            my_cl_tz[itrial] = self.make_cl_tz_mc()
            printlog(f'node {MPI.Get_processor_name()}: completed mc trial iteration {itrial}, time per iter: {(time.time() - t0)/(itrial + 1):.3e}', rank=rank)

        cl_tz = None
        if rank == 0:
            cl_tz = np.empty((ntrial_mc, lmax + 1))
        
        comm.Gather(my_cl_tz, cl_tz, root=0)
        return cl_tz

    # make data realization cl_tz
    def make_cl_tz_sig(self):
        cl_tz = self.make_cl_tz(self.cmb_pipe.map_t)
        self.cl_tz = cl_tz
        return cl_tz


# A class very similar to CMBxGalPipe but with one, two, or three CMB frequencies
# It may be useful to expand this to N frequencies in the future
#
# We add the planck_mask argument expliclty here since it's shared across all cmb pipes

# TODO: implement nl_coarse factor in all calcualtions
class CMBMFxGalPipe:
    def __init__(self, cmb_pipes, galaxy_pipe, gal_mask, planck_mask, nl_coarse, mf_meta,
                 *, output_manager=None, l_ksz_sum=2000, plots=False):
        assert len(cmb_pipes) < 4 # is this even necessary now?
        
        lmax_ref = cmb_pipes[0].lmax
        for pipe in cmb_pipes:
            assert pipe.lmax == lmax_ref
        
        assert pipe.lmax % nl_coarse == 0 # this may be unnecessary with new 

        assert l_ksz_sum > 1500

        self.plots = plots

        self.nl_coarse = nl_coarse
        self.lmax = lmax_ref
        self.nfreq = len(cmb_pipes)
        self.freqs = [pipe.freq for pipe in cmb_pipes]

        self.cmb_pipes = cmb_pipes
        self.galaxy_pipe = galaxy_pipe
        self.gal_mask = gal_mask # a completeness mask of the galaxy survey
        self.planck_mask = planck_mask


        self.mf_meta = mf_meta

        # lweight step-filter for pre-processing
        self.step_lweight = np.zeros(self.lmax + 1) # a default l-weight applied to each frequency
        self.lzero = 1500
        self.step_lweight[self.lzero:] = 1.
        assert l_ksz_sum > self.lzero
        self.l_ksz_sum = l_ksz_sum

        if output_manager is None:
            output_manager = cmb_pipe.output_manager

        self.output_manager = output_manager

        if plots: self._save_skyplot(self.planck_mask, 'planck_mask.png')

        self.init_nl = False
        self.init_xl = False
        self.init_t_hp = False

        # some multifrequency definitions
        # this also happens to be the correct init order
        self.init_data = False
        self.init_fkp_f = False
        self.init_t_tilde_f = False
        self.init_cl_tt_f = False
        self.init_lweights_f = False

        self.std_ksz = None

    def _save_skyplot(self, map_t, name, ticks=15, downgrade=16, colorbar=True):
        fig = enplot.plot(enmap.downgrade(map_t, downgrade), ticks=ticks, colorbar=colorbar)
        self.output_manager.savefig(name, mode='pixell', fig=fig)

    def import_data(self, plots=False):
        beams = []
        for pipe in self.cmb_pipes:
            if not pipe.init_data:
                pipe.import_data()
            beams.append(pipe.beam)    
        self.beams = np.array(beams)
        
        self.bl = np.zeros((self.nfreq, self.nfreq, self.lmax + 1))
        self.bl_inv = np.zeros((self.nfreq, self.nfreq, self.lmax + 1))
        for ifreq in range(self.nfreq):
            self.bl[ifreq, ifreq] = self.beams[ifreq]
            self.bl_inv[ifreq, ifreq] = 1./self.beams[ifreq]
        
        self.init_data = True

    # A generalized FKP function shared across multiple frequencies
    # Note: need to re-optimize to translate r_fkp -> sigma0
    def _make_fkp(self, sigma0, ivars=None):
        prod1 = self.cmb_pipes[0].make_zero_map() + 1.
        prod2 = self.cmb_pipes[0].make_zero_map()

        if ivars is None:
            nfreq = self.nfreq
            ivars = [pipe.ivar_t for pipe in self.cmb_pipes]
        else:
            assert len(ivars) > 0
            nfreq = len(ivars)

        for ivar in ivars:
            prod1 *= ivar

        for ifreq in range(nfreq):
            summand = self.cmb_pipes[0].make_zero_map() + 1.
            for ifreq2 in range(nfreq):
                if ifreq2 != ifreq: summand *= ivars[ifreq2]
            prod2 += summand

        fkp = prod1 * masked_inv(sigma0**2 * prod1 + prod2)

        ref_map = self.cmb_pipes[0].map_t

        # zero pixels where at least one ivar map is zero 
        # logically this is a sound practice
        # this also addresses the numerical problem of div-by-zero
        # where all nfreq ivar maps are zero
        zero_mask = np.zeros(ref_map.shape, dtype=np.uint8)
        for ivar_t in ivars:
            zero_mask = np.logical_or(zero_mask, ivar_t == 0.)
        fkp[zero_mask] = 0.

        return fkp

    # make a combined FKP weighting to apply to each map
    def init_fkp(self, ivars=None, plots=False):
        om = self.output_manager
        cmb_pipes = self.cmb_pipes

        # we only require that the data be initialized
        for pipe in cmb_pipes:
            if not pipe.init_data: pipe.import_data()

        # r_fkp = cmb_pipes[0].metadata.r_fkp # NOTE: should replace this with a MF metadata equivalent

        # NOTE: tune the sigma0 parameter!
        fkp = self._make_fkp(sigma0=self.mf_meta.sigma0, ivars=ivars)

        self.fkp = fkp * self.planck_mask

        if plots:
            self._save_skyplot(self.fkp, 'shared_fkp_planckmask.png')

        self.init_fkp_f = True

    def init_t_tilde(self, plots=False):
        assert self.init_data
        assert self.init_fkp_f
        
        wcs = self.cmb_pipes[0].map_t.wcs

        t_tilde = enmap.ndmap(np.empty((self.nfreq, get_size_alm(self.lmax)), dtype=np.complex64), wcs)

        for pipe, ifreq in zip(self.cmb_pipes, range(self.nfreq)):
            t_fkp = pipe.map_t * self.fkp
            t_tilde_lm = almxfl(map2alm(t_fkp, lmax=self.lmax), lfilter=self.step_lweight)
            t_tilde[ifreq,:] = t_tilde_lm
             
        self.t_tilde = t_tilde
        self.init_t_tilde_f = True

    def init_cl_tt(self, plots=False):
        om = self.output_manager
        assert self.init_t_tilde_f

        lzero = self.lzero

        ells = np.arange(self.lmax + 1)[lzero:]
        norm = ells * (ells + 1)**2 / 2 / np.pi

        t_tilde = self.t_tilde
        cl_tt = alm2cl(t_tilde[:,None,:], t_tilde[None,:,:])
        self.cl_tt = cl_tt

        if plots:
            plt.figure(dpi=300, facecolor='w')
            plt.title(r'$C_l^{TT}$')
            for ifreq1, freq1 in zip(range(self.nfreq), self.freqs):
                for ifreq2, freq2 in zip(range(ifreq1, self.nfreq, 1), self.freqs[ifreq1:]):
                    plt.plot(ells, norm * cl_tt[ifreq1, ifreq2, lzero:], label=f'{freq1} x {freq2}')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\Delta_l^{TT}$ ($\mu K^2$)')
            plt.legend()
            om.savefig('cl_tt_crosspower.png')

            plt.figure(dpi=300, facecolor='w')
            plt.title(r'$r_l^{TT}$ (Correlation Matrix)')
            for ifreq1, freq1 in zip(range(self.nfreq), self.freqs):
                for ifreq2, freq2 in zip(range(ifreq1, self.nfreq, 1), self.freqs[ifreq1:]):
                    plt.plot(ells, cl_tt[ifreq1, ifreq2, lzero:] / np.sqrt(cl_tt[ifreq1, ifreq1, lzero:] * cl_tt[ifreq2, ifreq2, lzero:]), label=f'{freq1} x {freq2}')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$r_l^{TT}$ (-1-1)')
            plt.legend()
            om.savefig('rl_tt.png')

        # print(f'cl_tt shape {cl_tt.shape}')
        cl_tt_inv = np.linalg.inv(cl_tt[...,self.lzero:].T).T
        # print(f'cl_tt_inv shape {cl_tt_inv.shape}')

        self.cl_tt_inv = cl_tt_inv
        
        self.init_cl_tt_f = True

    def _get_cl_th(self):
        lwidth = self.mf_meta.get_lwidth()
        ells = np.arange(self.lmax + 1).astype(float)

        return np.exp(-(ells/lwidth)**2)

    def init_lweights(self, plots=False):
        # TODO: implement
        # TODO: streamline expression?

        om = self.output_manager

        lcut = self.l_ksz_sum
        lzero = self.lzero


        cl_th_sf = self._get_cl_th()
        cl_th = np.array([cl_th_sf] * self.nfreq)
        self.cl_th = cl_th

        # G_l B^-1 is our l weight
        # = (lambda / C_l^{ZZ}) C_l^{th}^T C_l^{TT}^{-1}

        self.lweights = np.empty((self.nfreq, self.lmax + 1))
        self.lweights[:,lzero:] = np.einsum('ij,ikj->kj', cl_th[:,lzero:], np.einsum('ijk,jlk->ilk', self.bl[...,lzero:], self.cl_tt_inv))
        self.lweights[:,:lzero] = 0.

        if plots:
            ells = np.arange(self.lmax + 1)
            plt.figure(dpi=300, facecolor='w')
            for lweight, freq in zip(self.lweights, self.freqs):
                plt.plot(ells, lweight, label=f'{freq} GHz')   
            plt.legend()
            om.savefig('lweights_mf.png')

        self.init_lweights_f = True

    def compute_var(self):
        assert self.init_cl_tt
        
        lcut = self.lzero

        ells = np.arange(self.lmax + 1)[lcut:]
        cl_th = self.cl_th[:, lcut:]
        bl = self.bl[...,lcut:]
        bl_inv = self.bl_inv[...,lcut:]
        
        cl_inv = self.cl_tt_inv
        cl_tt = self.cl_tt[...,lcut:]
        
        bcb = np.einsum('ijk,jlk->ilk', bl, np.einsum('ijk,jlk->ilk', cl_inv, bl))
        ivar = ((2 * ells + 1) * np.einsum('ij,ikj,kj->j', cl_th, bcb, cl_th)).sum()

        var = 1./ivar
        self.std_ksz = np.sqrt(var)

        return var

    def init_mf(self, ivars=None, plots=False):
        self.import_data(plots=plots)
        self.init_fkp(plots=plots, ivars=ivars)
        self.init_t_tilde(plots=plots)
        self.init_cl_tt(plots=plots)
        self.init_lweights(plots=plots)
        self.compute_var()

    # Note: it may be faster to just do this all in harmonic space
    def process_maps(self, plots=False):
        cmb_pipes = self.cmb_pipes

        ref_pipe = cmb_pipes[0]
        maps = [pipe.map_t for pipe in cmb_pipes]

        # redo asserts?
        # assert cmb_pipe.init_data and cmb_pipe.init_fkp_f and cmb_pipe.init_lweight_f

        fkp = self.fkp
        lweights = self.lweights

        ret = ref_pipe.make_zero_map()

        for map_t, l_weight, freqs in zip(maps, lweights, self.freqs):
            map_fkp = fkp * map_t

            t_hp_alm = almxfl(map2alm(map_fkp, lmax=self.lmax), lfilter=l_weight)
            t_hp = alm2map(t_hp_alm, ref_pipe.make_zero_map())
            ret += t_hp

            if plots:
                fig = enplot.plot(enmap.downgrade(t_hp, 16), ticks=15, colorbar=True)
                self.output_manager.savefig(f't_hp_{freq}', mode='pixell', fig=fig)
            
        if plots:
            fig = enplot.plot(enmap.downgrade(ret, 16), ticks=15, colorbar=True)
            self.output_manager.savefig(f't_hp_gal_masked{freq}', mode='pixell', fig=fig)

        ret *= self.gal_mask
        return ret

    # provides a list of maps by frequency rather than summing
    def process_maps_list(self, lweights=None, plots=False):
        cmb_pipes = self.cmb_pipes

        ref_pipe = cmb_pipes[0]
        maps = [pipe.map_t for pipe in cmb_pipes]

        # redo asserts?
        # assert cmb_pipe.init_data and cmb_pipe.init_fkp_f and cmb_pipe.init_lweight_f

        fkp = self.fkp
        if lweights is None:
            lweights = self.lweights
        else:
            assert len(lweights) == self.nfreq

        ret_maps = []

        for map_t, l_weight, freqs in zip(maps, lweights, self.freqs):
            map_fkp = fkp * map_t

            t_hp_alm = almxfl(map2alm(map_fkp, lmax=self.lmax), lfilter=l_weight)
            t_hp = alm2map(t_hp_alm, ref_pipe.make_zero_map())
            ret_maps.append(t_hp * self.gal_mask)

        return ret_maps

    def _get_bs_samples(self, map_t, n_bs):
        printlog = self.output_manager.printlog

        gal_pipe = self.galaxy_pipe
        bs_samples = np.empty(n_bs)
        t_hp_list = gal_pipe.get_map_list(map_t)

        t0 = time.time()
        for itrial_bs in range(n_bs):
            bs_samples[itrial_bs] = gal_pipe.get_bs_estimator_list(t_hp_list)

            if itrial_bs % 1000 == 0:
                dt_iter = (time.time() - t0) / (itrial_bs + 1)
                printlog(f'bootstrap normalize time per iter: {dt_iter:.3e}, itrial: {itrial_bs} of {n_bs}')

        return bs_samples

    def _get_bs_mf(self, maps_list, n_bs):
        assert len(maps_list) == self.nfreq
        printlog = self.output_manager.printlog

        gal_pipe = self.galaxy_pipe
        bs_samples = np.empty((self.nfreq, n_bs))

        t0 = time.time()
        for itrial_bs in range(n_bs):
            bs_inds = gal_pipe.get_bs_inds()
            vr_bs = gal_pipe.vrs[bs_inds]

            for ifreq in range(self.nfreq):
                bs_samples[ifreq, itrial_bs] = (maps_list[ifreq][bs_inds] * vr_bs).sum()

            if itrial_bs % 1000 == 0:
                dt_iter = (time.time() - t0) / (itrial_bs + 1)
                printlog(f'bootstrap normalize time per iter: {dt_iter:.3e}, itrial: {itrial_bs} of {n_bs}')

        return bs_samples

    def investigate_mf_estimator(self, n_bs):
        printlog = self.output_manager.printlog
        gal_pipe = self.galaxy_pipe
        vrs = gal_pipe.vrs
        assert self.nfreq > 1

        maps = self.process_maps_list()
        maps_list = [gal_pipe.get_map_list(map_t) for map_t in maps]

        alphas_sig = [(map_t * vrs).sum() for map_t in maps_list]

        alphas_bs = self._get_bs_mf(maps_list, n_bs)

        # the covariance matrix of the individual frequency alpha estimators
        covar_freqs = np.cov(alphas_bs)

        alphas_norm = [ar / np.sqrt(covar_freqs[ifreq, ifreq]) for ar, ifreq in zip(alphas_bs, range(self.nfreq))]
        covar_norm = np.cov(alphas_norm)

        alphas_bs_sf = np.mean(alphas_norm, axis=1) / np.std(alphas_norm, axis=1)
        printlog(f'single-frequency ksz estimators: {alphas_bs_sf}')

        alphas_mean = np.mean(alphas_norm, axis=1)
        assert alphas_mean.shape[0] == self.nfreq

        optweight = np.einsum('ij,j->i', np.linalg.inv(covar_norm), alphas_mean)
        alphas_mf_bf = np.einsum('i,ij->j', optweight, alphas_norm)

        alpha_mf = np.mean(alphas_mf_bf) / np.std(alphas_mf_bf)

        printlog(f'mf ksz estimator: {alpha_mf:.3e}')

    def compute_estimator(self, n_bs=0, n_mc=0):
        printlog = self.output_manager.printlog

        mf_map = self.process_maps()

        a_ksz = self.galaxy_pipe.get_xz_list(mf_map).sum()

        printlog(f'alpha_ksz_mf {a_ksz / self.std_ksz:.3e}')

        std_bs = None
        if n_bs > 0:
            bs_samples = self._get_bs_samples(mf_map, n_bs)
            mean_bs = np.mean(bs_samples)
            std_bs = np.std(bs_samples)
            printlog(f'mean_bs, {mean_bs:.3e} std_bs {std_bs:.3e}')
            printlog(f'alpha_ksz_bs {a_ksz/std_bs:.3e}')

        if n_mc > 0:
            # TODO: implement multifrequency MC
            printlog('multifrequency mc not implemented!')
            pass
