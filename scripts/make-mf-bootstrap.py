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


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='make-mf-bootstrap',)
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

    cross_pipe = CMBMFxGalPipe(estimator_pipes, gal_pipe, gal_mask, planck_mask, NL_COARSE, mf_meta, 
                               output_manager=om, l_ksz_sum=L_KSZ_SUM, plots=True)

    ivars_ksz = get_inds([pipe.ivar_t for pipe in cmb_pipes], f_inds_ivar)
    cross_pipe.init_mf(plots=True, ivars=ivars_ksz)

    cross_pipe.compute_estimator(n_bs=NTRIAL_BS)