from fnl_pipe.util import OutputManager, get_fname, get_yaml_dict, get_planck_mask
from fnl_pipe.cmb import ACTPipe
from fnl_pipe.gal_cmb import CMBMFxGalPipe, MFMetadata
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import itertools

import numpy as np

import matplotlib.pyplot as plt

import sys


def get_inds(ar, inds):
    return [ar[i] for i in inds]

def snr_theory(a, b, r):
    return np.sqrt(a**2 + b**2 - 2 * r * a * b) / np.sqrt(1 - r**2)

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='mf-bootstrap-mapweight',)
    printlog = om.printlog

    # YAML config file import
    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    local_dict = locals()
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(key, value)
        local_dict[key] = value
    printlog('################## DONE ##################')
    # end YAML snippet

    # must define maps, ivars, beams, and freqs arrays in the config file
    # f_inds controls which indices are used in the other lists;
    # it's an easy way to choose subsets of frequencies
    # freqs = get_inds(freqs, f_inds)
    # maps = get_inds(maps, f_inds)
    # ivars = get_inds(ivars, f_inds)
    # beams = get_inds(beams, f_inds)

    cmb_pipes = []

    for freq, map_path, ivar_path, beam_path in zip(freqs, maps, ivars, beams):
        act_pipe = ACTPipe(map_path, ivar_path, beam_path, planck_mask_path,
                       om, lmax=LMAX, freq=freq, plots=True)
        act_pipe.import_data()
        cmb_pipes.append(act_pipe)

    ref_map = cmb_pipes[0].map_t

    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)

    zerr_cut = ZerrCut(zerr_max)
    if do_cut:
        north_cut = AndCut([LRGNorthCut(), zerr_cut])
        south_cut = AndCut([LRGSouthCut(), zerr_cut])
    else:
        north_cut = zerr_cut
        south_cut = zerr_cut

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)

    cutstring = f'zerr_max {zerr_max:.3f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
    printlog(cutstring)

    # printlog('getting gal pipe')
    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    # printlog('done')

    # printlog('importing data')
    gal_pipe.import_data()
    # printlog('done')

    # make and plot cl_zz

    cl_zz = gal_pipe.get_cl(lmax=LMAX)
    ells = np.arange(LMAX + 1)

    plt.figure(dpi=300, facecolor='w')
    plt.title('Galaxy Field Power Spectrum')
    plt.plot(ells[2000:], cl_zz[2000:])
    plt.ylabel(r'$C_l^{ZZ}$')
    plt.xlabel(r'$l$')
    om.savefig('cl_zz.png')
    plt.close()

    # do multifrequency bootstrap estimate

    planck_mask = get_planck_mask(planck_mask_path, cmb_pipes[0].map_t.wcs)

    # use sigma0 = 0. for now, need to optimize
    mf_meta = MFMetadata(0., 1.56)

    # only apply the index cut to the estimator pipes
    estimator_pipes = get_inds(cmb_pipes, f_inds)

    # pull just the ivars we indend to use
    ivars_ksz = get_inds([pipe.ivar_t for pipe in cmb_pipes], f_inds_ivar)

    nfreq = len(f_inds)

    assert nfreq > 1

    # single frequency cross-pipes
    sf_cross_pipes = []
    maps = []
    for cmb_pipe in estimator_pipes:
        cross_pipe = CMBMFxGalPipe([cmb_pipe,], gal_pipe, gal_mask, planck_mask, NL_COARSE, mf_meta, 
                                   output_manager=om, l_ksz_sum=L_KSZ_SUM, plots=True)
        cross_pipe.init_mf(plots=True, ivars = ivars_ksz)
        sf_cross_pipes.append(cross_pipe)
        these_maps = cross_pipe.process_maps_list()
        assert len(these_maps) == 1
        this_map = these_maps[0]

        maps.append(gal_pipe.get_map_list(this_map))
    maps = np.array(maps)

    vrs = gal_pipe.vrs
    alphas_mf = np.empty((NTRIAL_BS, nfreq))
    for itrial_bs in range(NTRIAL_BS):
        bs_inds = gal_pipe.get_bs_inds()
        alphas_mf[itrial_bs, :] = (vrs[None, bs_inds] * maps[:, bs_inds]).sum(axis=1)

    alphas_mf = alphas_mf.T

    alphas_cross = np.cov(alphas_mf)
    printlog(alphas_cross)

    alphas_norm = alphas_mf.copy()
    for ifreq in range(nfreq):
        alphas_norm[ifreq, :] /= np.sqrt(alphas_cross[ifreq, ifreq])
    an_cov = np.cov(alphas_norm)
    printlog(an_cov)

    assert nfreq == 2

    r = an_cov[0,1]

    an_mean = np.mean(alphas_norm, axis=1)
    snr_sf = an_mean / np.std(alphas_norm, axis=1)
    printlog(f'single frequency SNR {snr_sf}')

    icov = np.linalg.inv(an_cov)
    printlog(f'alpha_mean array {an_mean}')

    # map space weights
    ms_weights = np.einsum('ij,j->i', icov, an_mean)
    printlog(f'alpha weights {ms_weights}')

    snr_mf_th = snr_theory(*snr_sf, r=r)
    printlog(f'theoretical combined SNR: {snr_mf_th:.3e}')

    alphas_mf = np.einsum('i,ij->j', ms_weights, alphas_norm)
    snr_mf = np.mean(alphas_mf) / np.std(alphas_mf)
    printlog(f'combined multifrequency SNR {snr_mf:.3e}')