from fnl_pipe.util import masked_inv, get_fname, map2cl, parse_act_beam, fequal

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils, curvedsky, enplot
from pixell.curvedsky import map2alm, alm2map, rand_alm, almxfl, rand_map, alm2cl


cmb_freqs = [90, 150, 220] # GHz
cmb_freq_labels = ["90", "150", "220"]


class ltool:
    def __init__(self, lmax):
        self.lmax = lmax
        self.ells = np.arange(1 + lmax)
        self.iells = masked_inv(self.ells)
        self.iells2 = self.iells**2
        self.norm = self.ells * (self.ells + 1) / 2 / np.pi


class ACTMetadata:
    def __init__(self, r_fkp, r_lwidth):
        self.r_fkp = r_fkp
        self.r_lwidth = r_lwidth

    def copy(self):
        return deepcopy(self)


# A class to represent an individual act map at a single frequency
# this class does not explicitly reference a galaxy mask since this is defined
# at the "cross analysis" level, i.e. higher up.
class ACTPipe:
    def __init__(self, map_path, ivar_path, beam_path, planck_mask_path,
        output_manager, freq=None, lmax=12000, lmax_sim=12000, plots=False, metadata=None,
        l_ksz = 3000, lmin_cut=1500):
        assert freq is not None
        assert freq == 90 or freq == 150 or freq == 220
        self.freq = freq

        self.l_ksz = l_ksz

        self.map_path = map_path
        self.ivar_path = ivar_path
        self.beam_path = beam_path
        self.planck_mask_path = planck_mask_path
        # self.gal_mask_path = gal_mask_path

        self.output_manager = output_manager

        self.lmax = lmax
        self.ltool = ltool(lmax)
        self.lmax_sim = lmax_sim
        self.ltool_sim = ltool(lmax_sim)
        self.lmin_cut = lmin_cut

        self.plots = plots

        self.init_data = False
        # self.init_metadata = False # parameters for pixel weight and lweight
        if metadata is None:
            self.init_metadata = False
        else:
            self.metadata = metadata.copy()
            self.init_metadata = True

        self.init_lweight_f = False
        self.init_xfer = False
        self.init_fkp_f = False

    # def init(self):
    #     assert self.
        
    #     if not self.init_data:
    #         self.import_data()

    #     self.init_lweight()

    def get_planck_mask(self):
        printlog = self.output_manager.printlog

        printlog(f'importing planck foreground mask {self.planck_mask_path}')
        planck_mask_ar = np.load(self.planck_mask_path)
        mmin = planck_mask_ar.min()
        mmax = planck_mask_ar.max()
        printlog(f'planck mask range: {mmin} {mmax}')
        assert mmin >= 0. and mmax <= 1 * (1 + 1e-6) # Check that one_time_setup was run

        planck_mask = enmap.ndmap(planck_mask_ar, self.map_t.wcs)
        planck_mask = np.minimum(np.maximum(planck_mask, 0.), 1.)

        return planck_mask

    def import_data(self, plots=False, ntrial_nl=1):
        printlog = self.output_manager.printlog
        plots = plots or self.plots

        ells = self.ltool.ells
        norm = self.ltool.norm

        printlog(f'importing act map {self.map_path}')
        self.map_t = enmap.read_map(self.map_path)[0]

        self.cl_tt_act = map2cl(self.map_t, lmax=self.lmax)

        # WARN: untracked unit conversion
        self.adelt = self.map_t.wcs.wcs.cdelt * np.pi / 180.
        self.cdelt = self.map_t.wcs.wcs.cdelt

        assert fequal(np.abs(self.adelt[0]), np.abs(self.adelt[1]))

        self.angular_res = self.adelt[0] # radians
        self.pixel_area = self.angular_res**2 # steridians

        printlog(f'importing act ivar map {self.ivar_path}')
        self.eta_n2 = enmap.read_map(self.ivar_path)[0] / self.pixel_area # trick to save ram
        self.map_std = np.sqrt(masked_inv(self.eta_n2 * self.pixel_area))

        # quality check on map_std?
        if plots:
            plt.figure(dpi=300)
            plt.title('noise_cl')
            # plt.plot(ells, norm * self.nl, label='nl')
            plt.plot(ells, norm * self.cl_tt_act, label='cl_act')
            # plt.plot(ells, norm * self.cl_psuedo_act, label='cl_act (no noise)')
            plt.legend()
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\Delta_l$')
            self.output_manager.savefig('cl_map.png')
            plt.close()

        # TODO: can consolidate into

        # TODO: mutual overlap check between galaxy mask and survey?
        printlog(f'importing beam {self.beam_path}')
        beam_ells, beam_amps = parse_act_beam(self.beam_path)
        self.lmax_beam = max(beam_ells)
        self.beam_full = beam_amps
        assert self.lmax_beam >= max(self.lmax, self.lmax_sim)
        self.beam_sim = self.beam_full[:self.lmax_sim + 1]
        self.beam = self.beam_full[:self.lmax + 1]

        if plots:
            plt.figure(dpi=300)
            plt.plot(beam_amps)
            plt.xlabel(r'$l$')
            plt.ylabel(r'beam amplitude')
            self.output_manager.savefig(f'act_beam_{self.freq}_ghz.png')
            plt.close()

        self.init_data = True

    def update_metadata(self, act_metadata, do_init=True):
        if self.metadata.r_lwidth != act_metadata.r_lwidth:
            self.init_lweight_f = False
            self.init_xfer = False

        if self.metadata.r_fkp != act_metadata.r_fkp:
            self.init_fkp_f = False
            self.init_xfer = False

        self.metadata = act_metadata.copy()
        self.init_metadata = True

        if do_init: self.init()

    def get_sim_map():
        assert self.init_xfer and self.init_fkp_f
        pass
        #TODO

    def make_empty_map(self):
        return enmap.empty(self.map_t.shape, self.map_t.wcs)

    def make_zero_map(self):
        return enmap.ndmap(np.zeros(self.map_t.shape), self.map_t.wcs)

    def make_noise_map(self):
        return enmap.ndmap(np.random.normal(size=self.map_t.shape) * self.map_std, 
                           self.map_t.wcs)

    # single frequency lweight
    def init_lweight(self, plots=False):
        assert self.init_metadata
        assert self.init_fkp_f

        plots = plots or self.plots

        r_lwidth = self.metadata.r_lwidth
        ells = self.ltool.ells

        cl_ksz_th = np.exp(-(ells / r_lwidth / 5500.)**2)
        cl_tt_act = self.cl_tt_act # defined on the FKP weighted act data?

        # WARN: no normalization included
        self.l_weight = self.beam * cl_ksz_th / cl_tt_act
        self.l_weight[:self.lmin_cut + 1] = 0.

        # self.l_weight = self.l_weight / self.l_weight[self.l_ksz]
        self.l_weight = self.l_weight / max(self.l_weight)

        if plots:
            plt.figure(dpi=300)
            plt.title('l-weight (one power of the beam)')
            plt.plot(self.ltool.ells, self.l_weight)
            plt.xlabel(r'$l$')
            plt.ylabel('l-weight')

            self.output_manager.savefig(f'l_weight_{self.freq}.png')
            plt.close()

        self.init_lweight_f = True

    def init_fkp(self, plots=False):
        assert self.init_data
        assert self.init_metadata

        printlog = self.output_manager.printlog
        
        plots = plots or self.plots

        ctt_3k_act = 24.8 * 2 * np.pi / 3000 / (3000 + 1)

        b2_l0 = self.beam[1]**2

        r_fkp = self.metadata.r_fkp

        planck_mask = self.get_planck_mask()
        self.fkp = planck_mask * self.eta_n2 / (r_fkp/(b2_l0 * ctt_3k_act) + self.eta_n2)
        del self.eta_n2

        printlog(f'FKP min/max: {np.min(self.fkp):.3e} {np.max(self.fkp):.3e}')

        # plot the fkp weight function, the cl
        if plots:
            fig1 = enplot.plot(enmap.downgrade(self.fkp, 16), ticks=15, colorbar=True)
            self.output_manager.savefig(f'fkp_map_{self.freq}', mode='pixell',
                                        fig=fig1)

            fig2 = enplot.plot(enmap.downgrade(planck_mask, 16), ticks=15, colorbar=True)
            self.output_manager.savefig(f'planck_mask_{self.freq}', mode='pixell',
                                        fig=fig2)

            fig3 = enplot.plot(enmap.downgrade(planck_mask * self.map_t, 16),
                               ticks=15, range=300,  colorbar=True)
            self.output_manager.savefig(f'cmb_planck_masked_{self.freq}', mode='pixell',
                                        fig=fig3)

        self.init_fkp_f = True

    def make_xfer(self, plots=False):
        pass


class MultiFreqPipe:
    def __init__(self, act_pipes):
        self.freqs = []
        for pipe in act_pipes:
            freq = pipe.freq
            assert freq not in self.freqs # check for duplicates

            self.freqs.append(freq)

        self.act_pipes = act_pipes

    def import_data():
        pass # TODO

    # TODO: finish - the main component is a relative weighting applied to the respective maps
    # This weighting