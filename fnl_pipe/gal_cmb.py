# This module defines classes and functions relevant to gal x CMB operations.
# For example, classes that mix galaxy and CMB data are defined here.

import numpy as np
import matplotlib.pyplot as plt
import time

from pixell.curvedsky import map2alm, almxfl, alm2map, alm2cl, rand_map
from pixell import enmap, enplot

from fnl_pipe.util import map2cl, ChunkedTransposeWriter, average_fl, matmul, get_size_alm, masked_inv

# import line_profiler

# TODO: implement
# class CMBxGalHash:
#     def __init__(self, *, map_t, fkp_t, lweight):
#         self.val = hash()

#     def __eq__(self, b):
#         assert isinstance(b, CMBxGalHash)
#         assert self.val == b.val


def _write_nl(nl_path, nl_tilde, **kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs['nl_tilde'] = nl_tilde
    np.save(nl_path, kwargs)


def _write_fl(fl_path, fl, **kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs['fl'] = fl
    np.save(fl_path, kwargs)


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
        self.init_fl = False
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

        ret_map = ref_pipe.make_zero_map()

        for map_t, l_weight, freqs in zip(maps, lweights, self.freqs):
            map_fkp = fkp * map_t

            t_hp_alm = almxfl(map2alm(map_fkp, lmax=self.lmax), lfilter=l_weight)
            t_hp = alm2map(t_hp_alm, ref_pipe.make_zero_map())
            ret_map += t_hp

            if plots:
                fig = enplot.plot(enmap.downgrade(t_hp, 16), ticks=15, colorbar=True)
                self.output_manager.savefig(f't_hp_{freq}', mode='pixell', fig=fig)
                
                fig = enplot.plot(enmap.downgrade(ret, 16), ticks=15, colorbar=True)
                self.output_manager.savefig(f't_hp_gal_masked{freq}', mode='pixell', fig=fig)

        ret_map *= self.gal_mask
        return ret_map

    def _get_bootstrap_norm(self, map_t, n_bs):
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

        return np.mean(bs_samples), np.std(bs_samples)

    def compute_estimator(self, n_bs=0, n_mc=0):
        printlog = self.output_manager.printlog

        mf_map = self.process_maps()

        a_ksz = self.galaxy_pipe.get_xz_list(mf_map).sum()

        printlog(f'alpha_ksz_mf {a_ksz / self.std_ksz:.3e}')

        std_bs = None
        if n_bs > 0:
            mean_bs, std_bs = self._get_bootstrap_norm(mf_map, n_bs)
            printlog(f'mean_bs, {mean_bs:.3e} std_bs {std_bs:.3e}')
            printlog(f'alpha_ksz_bs {a_ksz/std_bs:.3e}')

        if n_mc > 0:
            # TODO: implement multifrequency MC
            printlog('multifrequency mc not implemented!')
            pass


class CMBxGalPipe:
    def __init__(self, cmb_pipe, galaxy_pipe, gal_mask, *, output_manager=None):
        self.cmb_pipe = cmb_pipe
        self.galaxy_pipe = galaxy_pipe
        self.gal_mask = gal_mask # a completeness mask of the galaxy survey

        if output_manager is None:
            output_manager = cmb_pipe.output_manager

        self.output_manager = output_manager

        self.init_nl = False
        self.init_fl = False
        self.init_t_hp = False

        self.alpha_mc = None

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

    def update_nl(self, nl_tilde, *, nl_path=None, **kwargs):
        assert len(nl_tilde) == self.cmb_pipe.lmax + 1

        if nl_path is not None:
            _write_nl(nl_path, nl_tilde, **kwargs)

        self.nl_tilde = nl_tilde
        self.cl_act_tilde = map2cl(self.process_map(self.cmb_pipe.map_t), lmax=self.cmb_pipe.lmax)
        self.init_nl = True

    def make_nl(self, *, ntrial_nl=32, nl_path=None, plots=False):
        cmb_pipe = self.cmb_pipe
        lmax = self.cmb_pipe.lmax
        ells = self.cmb_pipe.ltool.ells
        norm = cmb_pipe.ltool.norm
        process_map = self.process_map
        printlog = self.output_manager.printlog

        t0 = time.time()
        nl_tilde = np.zeros(lmax + 1)
        for itrial in range(ntrial_nl):
            printlog(f'doing nl trial {itrial + 1}')
            t_noise = process_map(cmb_pipe.make_noise_map())
            this_nl = map2cl(t_noise, lmax=lmax)
            nl_tilde += this_nl
            tnow = time.time()
            printlog(f'time per iteration: {(tnow - t0)/(itrial + 1):.3e} s')
        nl_tilde /= ntrial_nl

        if plots:
            plt.figure(dpi=300)
            plt.plot(ells, nl_tilde)
            plt.xlabel(r'$l$')
            plt.ylabel(r'$N_l$')
            self.output_manager.savefig(f'nl_{self.cmb_pipe.freq}_ghz_{ntrial_nl}_ntrial.png', 
                                        mode='matplotlib')
            plt.close()

        self.update_nl(nl_tilde, nl_path=nl_path, ntrial_nl=ntrial_nl)

    def import_nl(self, nl_path):
        nl_dict = np.load(nl_path, allow_pickle=True)
        self.update_nl(nl_dict.item()['nl_tilde'])

    def update_fl(self, fl, *, fl_path=None, **kwargs):
        assert self.init_nl
        assert len(fl) == self.cmb_pipe.lmax + 1

        if fl_path is not None:
            _write_fl(fl_path, fl, **kwargs)

        self.fl = fl
        self.cl_act_sim = (self.cl_act_tilde - self.nl_tilde) / self.fl
        self.init_fl = True

    def _fl_loop_mpi(self, cl_in, ntrial, comm):
        lmax = self.cmb_pipe.lmax

        rank = comm.Get_rank()
        size = comm.Get_size()

        assert ntrial % size == 0
        my_ntrial = ntrial // size

        my_cl_out, my_fl = self._fl_loop(cl_in, my_ntrial)

        print('got my_cl_out')
        cl_out = None
        if rank == 0: cl_out = np.zeros((size, lmax + 1))

        comm.Gather(my_cl_out, cl_out, root=0)
        if rank == 0:
            cl_out = cl_out.sum(axis=0) / size
            this_fl = cl_out / cl_in
            return cl_out, this_fl
        else:
            # return empty arrays to await a broadcast
            # possibly unsafe
            return np.empty(lmax + 1), np.empty(lmax + 1)

    def _fl_loop(self, cl_in, ntrial, rank=None):
        rankstr = ''
        if rank is not None:
            assert isinstance(rank, int)
            rankstr = f'rank: {rank} '

        lmax = self.cmb_pipe.lmax
        printlog = self.output_manager.printlog
        process_map = self.process_map
        map_ref = self.cmb_pipe.map_t

        cl_out = np.zeros(lmax + 1)
        t0 = time.time()
        for itrial_fl in range(ntrial):
            printlog(rankstr + f'doing trial {itrial_fl} of {ntrial}')
            cl_out += map2cl(process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_in)),lmax=lmax)
            time_per_fl = (time.time() - t0) / (itrial_fl + 1)
            printlog(rankstr + f'time per fl eval: {time_per_fl:.3e}')
        cl_out /= ntrial
        this_fl = cl_out / cl_in

        return cl_out, this_fl

    def make_fl_iter(self, *, nave_fl=32, ntrial_fl=60, niter=4, fl_path=None, plots=False, comm=None):
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

        fl = np.ones(lmax + 1)
        cl_target = cl_act_tilde - nl_tilde
        rms_err = np.zeros(niter)
        for iiter in range(niter):
            print(f'doing iter {iiter}')
            cl_in = cl_target/fl

            if do_mpi_fl:
                print(f'doing mpi fl, rank: {rank}')
                cl_out, this_fl = self._fl_loop_mpi(cl_in, ntrial_fl, comm)
                print(f'finished _fl_loop_mpi, rank = {rank}')

                comm.Bcast(cl_out, root=0)
                comm.Bcast(this_fl, root=0)
            else:
                cl_out, this_fl = self._fl_loop(cl_in, ntrial_fl)

            fl = average_fl(this_fl, nave_fl)
            fl[0] = 1.

            fit_goodness = np.sqrt(np.sum((cl_target[lmin_cut+1:] - cl_out[lmin_cut+1:])**2)/(lmax - lmin_cut))
            rms_err[iiter] = fit_goodness
            printlog(f'make_xfer_iter {iiter} of {niter}: rms cl error= {fit_goodness:.3e}')

        if plots:
            cl_in = cl_target/fl
            cl_out = np.zeros(lmax + 1)
            for itrial_fl in range(ntrial_fl):
                cl_out += map2cl(process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_in)),lmax=lmax)
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
            plt.legend()
            self.output_manager.savefig('cl_xfer_compare.png', mode='matplotlib')
            plt.close()

            cl_ave = 0.5 * (cl_target + cl_out)
            plt.figure(dpi=300)
            plt.plot(ells, (cl_out - cl_target) / cl_ave)
            plt.xlabel(r'$l$')
            plt.ylabel(r'relative error')
            plt.ylim([-1,1])
            self.output_manager.savefig('cl_relative_error.png', mode='matplotlib')
            plt.close()

        self.update_fl(fl, fl_path=fl_path, ntrial_fl=ntrial_fl, niter_fl=niter,
                       nave_fl=nave_fl)

    def import_fl(self, fl_path):
        fl_dict = np.load(fl_path, allow_pickle=True)
        self.update_fl(fl_dict.item()['fl'])

    def make_fl_standard(self, ntrial_fl=1, nave_l=32, plots=False):
        cmb_pipe = self.cmb_pipe
        ells = self.cmb_pipe.ltool.ells
        norm = cmb_pipe.ltool.norm
        lmax = self.cmb_pipe.lmax
        map_ref = self.cmb_pipe.map_t
        process_map = self.process_map

        cl_ref = np.ones(lmax + 1)
        
        fl = np.zeros(lmax + 1) # base xfer function
        nl_tilde = np.zeros(lmax + 1)
        for itrial in range(ntrial_fl):
            t_proc = process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_ref))
            t_noise = process_map(cmb_pipe.make_noise_map())
            this_nl = map2cl(t_noise, lmax=lmax)
            cl_out = map2cl(t_proc, lmax=lmax)

            nl_tilde += this_nl
            fl += cl_out/cl_ref

        fl /= ntrial_fl
        nl_tilde /= ntrial_fl

        self._update_nl(nl_tilde)

        assert lmax % nave_l == 0
        fl_ave = np.repeat(fl[1:].reshape(lmax // nave_l, nave_l).sum(axis=-1)/nave_l, nave_l)

        self.fl_standard = np.ones(lmax + 1)
        self.fl_standard[1:] = fl_ave

        cl_cmb_test = (self.cl_tilde_act - self.nl_tilde) / self.fl_standard
        cl_out_test = map2cl(process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, ps=cl_cmb_test)), lmax=lmax)

        # this plot is a mandatory diagnostic
        plt.figure(dpi=300)
        plt.title('standard (diagonal) xfer function diagnostic')
        plt.plot(ells, norm * (self.cl_tilde_act - self.nl_tilde), label='act pseudo cl (input)')
        plt.plot(ells, norm * cl_out_test, label='random pseudo cl (output)')
        plt.ylabel(r'$\Delta_l$')
        plt.xlabel(r'$l$')
        plt.legend()
        self.output_manager.savefig(f'cl_fl_diagnostic_{cmb_pipe.freq}.png')
        plt.close()

        if plots:
            plt.figure(dpi=300)
            plt.plot(ells[1:], fl[1:], label='fl_fine')
            plt.plot(ells[1:], self.fl_standard[1:], label='fl_averaged')
            plt.legend()
            plt.ylabel('fl')
            plt.xlabel(r'$l$')
            self.output_manager.savefig(f'fl_nave_{nave_l}_ntrial_{ntrial_fl}.png')
            plt.close()

        self.init_fl = True

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
            t_hp_mc = process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, 
                                           ps=cl_act_sim) + cmb_pipe.make_noise_map())
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
            assert self.init_fl
            alphas = np.zeros(ntrial_mc, dtype=np.float64)
            t0 = time.time()

            if buffered_reader is None:
                for itrial in range(ntrial_mc):
                    t_hp_mc = process_map(rand_map(shape=map_ref.shape, wcs=map_ref.wcs, 
                                                   ps=cl_act_sim) + cmb_pipe.make_noise_map())
                    t_mc_list = t_hp_mc[gal_decs, gal_ras]

                    alphas[itrial] = (vrs * t_mc_list).sum()

                    dt_total = time.time() - t0
                    dt_per_iter = dt_total / (itrial + 1)
                    printlog(f'completed MC mctrial iteration, time per iter: {dt_per_iter:.2e} s')

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