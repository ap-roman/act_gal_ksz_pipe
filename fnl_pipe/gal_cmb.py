# This module defines classes and functions relevant to gal x CMB operations.
# For example, classes that mix galaxy and CMB data are defined here.

import numpy as np
import matplotlib.pyplot as plt
import time

from pixell.curvedsky import map2alm, almxfl, alm2map, rand_map
from pixell import enmap, enplot

from fnl_pipe.util import map2cl, ChunkedTransposeWriter

# import line_profiler

# smoothes the l=1 to lmax entries of fl
# specifically ignores l=0
def average_fl(fl, nave_l=32):
    lmax = len(fl) - 1
    return np.repeat(fl[1:].reshape(lmax // nave_l, nave_l).sum(axis=-1)/nave_l, nave_l)


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

            fl[0] = 1.
            fl[1:] = average_fl(this_fl, nave_l=nave_fl)

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
        assert self.init_nl
        assert self.init_fl

        self.t_hp = self.process_map(self.cmb_pipe.map_t) # the processed act map
        self.init_t_hp = True
    
    #@profile
    def compute_estimator(self, ntrial_mc=32, buffered_reader=None):
        assert self.init_nl
        assert self.init_fl

        cmb_pipe = self.cmb_pipe
        lmax = cmb_pipe.lmax
        gal_pipe = self.galaxy_pipe
        gal_decs, gal_ras = gal_pipe.gal_inds
        vrs = gal_pipe.vr_list
        map_ref = self.cmb_pipe.map_t
        process_map = self.process_map
        cl_act_sim = self.cl_act_sim
        printlog = self.output_manager.printlog

        if not self.init_t_hp:
            self._init_t_hp()
        t_hp = self.t_hp

        t_hp_list = t_hp[gal_decs, gal_ras]

        a_ksz_unnorm = gal_pipe.get_xz_list(t_hp).sum()
        self.a_ksz_unnorm = a_ksz_unnorm # used by mpi version
        a_std_bootstrap = np.sqrt(gal_pipe.ngal_in * np.var(t_hp_list) * np.var(vrs))
        a_bootstrap = a_ksz_unnorm/a_std_bootstrap
        a_std_bootstrap_2 = np.sqrt(((vrs * t_hp_list)**2).sum())
        a_bootstrap_2 = a_ksz_unnorm/a_std_bootstrap_2

        printlog(f'analytic estimator: {a_bootstrap:.3e}')
        printlog(f'analytic estimator 2: {a_bootstrap_2:.3e}')

        ret = {'a_ksz_bootstrap': a_bootstrap, 'a_ksz_bootstrap_2': a_bootstrap_2}
        
        if ntrial_mc > 0:
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
                    gal_inds, t_hp = buffered_reader.get_next_chunk()
                    printlog(f'done')
                    ngal_this_chunk = t_hp.shape[0]
                    assert t_hp.shape[0] == gal_inds[1] - gal_inds[0]

                    alphas[:] += (vrs[gal_inds[0]:gal_inds[1], None] * t_hp).sum(axis=0)

                    dt_total = time.time() - t0
                    dt_per_iter = dt_total / (ngal + ngal_this_chunk)

                    dt_total_expected = gal_pipe.ngal_in * dt_per_iter
                    ngal += ngal_this_chunk
                    # printlog(f'completed {ngal} MC gals, time per iter: {dt_per_iter:.2e} s, expected total time: {dt_total_expected:.2e} s')

                assert not buffered_reader.has_next_chunk
                assert ngal == gal_pipe.ngal_in
                dt_total = time.time() - t0
                printlog(f'completed entire galaxy mc loop in {dt_total:.2e} s')

            a_std_mc = np.std(alphas)
            a_ksz_mc = a_ksz_unnorm / a_std_mc
            ret['a_ksz_mc'] = a_ksz_mc
            self.alphas = alphas

        return ret