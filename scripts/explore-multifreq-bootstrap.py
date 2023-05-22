from fnl_pipe.util import OutputManager, get_fname, get_yaml_dict
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import itertools

import numpy as np

import sys

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='explore-mf-bootstrap', logs=['log',])
    printlog = om.printlog

    assert len(sys.argv) == 2, 'usage: python <scriptname> <config_file>'

    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    local_dict = locals()
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(key, value)
        local_dict[key] = value
    printlog('################## DONE ##################')

    # maps = [map_90, map_150]
    # ivars = [ivar_90, ivar_150]
    # beams = [beam_90, beam_150]
    # freqs = [90, 150]

    maps = [map_90, map_150, map_220]
    ivars = [ivar_90, ivar_150, ivar_220]
    beams = [beam_90, beam_150, beam_220]
    freqs = [90, 150, 220]

    nfreq = len(freqs)

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62) # not necessarily applicable to all frequencies

    r_fkp = act_md.r_fkp


    # Messy FKP code!
    eta_n2s = []
    act_pipes = []
    eta2_tot = None

    ref_map = None
    for map_path, ivar_path, beam_path, freq in zip(maps, ivars, beams, freqs):
        act_pipe = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                               om, freq=freq, metadata=act_md, plots=True)
        act_pipe.import_data()
        act_pipes.append(act_pipe)
        eta_n2s.append(act_pipe.eta_n2)

        if eta2_tot is None:
            eta2_tot = 1. / act_pipe.eta_n2
        else:
            eta2_tot += 1. / act_pipe.eta_n2

        ref_map = act_pipe.map_t


    # a hackey attempt at a "low" inverse variance
    eta2_0 = np.quantile(eta2_tot.flatten(), 0.025)

    printlog(f'eta2_0 {eta2_0:.3e} quantile {0.025:.3e}')

    eta_n2s = enmap.ndmap(eta_n2s, ref_map.wcs)
    eta_prod = eta_n2s.prod(axis=0)

    fig1 = enplot.plot(enmap.downgrade(eta_prod, 16), ticks=15, colorbar=True)
    om.savefig(f'eta_prod', mode='pixell', fig=fig1)

    eta_prod_2 = act_pipes[0].make_zero_map() # products involving nfreq - 1 copies of eta_n2

    if nfreq == 2:
        eta_prod_2 = eta_n2s.sum(axis=0)

    if nfreq == 3:
        for ifreq in range(nfreq):
            for jfreq in range(nfreq):
                if ifreq != jfreq: eta_prod_2 += eta_n2s[ifreq] * eta_n2s[jfreq]

    fig2 = enplot.plot(enmap.downgrade(eta_prod_2, 16), ticks=15, colorbar=True)
    om.savefig(f'eta_prod_2', mode='pixell', fig=fig2)


    # # define the FKP weight function for all frequences as 
    # # \frac{1}{\sigma_0^2 + \sigma_T^2}, where \sigma_T^2 is the quadrature sum of 
    # # individual channel variances
    fkp = (1./5.72e4) * eta_prod / (1e-5 * eta_prod + eta_prod_2)

    # n_nan = (fkp == np.nan).sum()

    # printlog(f'fkp fraction NaN: {n_nan / np.prod(fkp.shape)}')
    # fkp = np.nan_to_num(fkp)

    ctt_3k_act = 24.8 * 2 * np.pi / 3000 / (3000 + 1)

    printlog(f'old fkp norm {r_fkp/ctt_3k_act:.3e}')
    # fkp = eta_n2s[0] * eta_n2s[1] / (len(eta_n2s) * r_fkp / ctt_3k_act  +  eta_n2s[0] + eta_n2s[1])

    # apply the "isnan" mask to 

    zero_mask = np.zeros(ref_map.shape, dtype=np.uint8)
    for eta_n2 in eta_n2s:
        zero_mask = np.logical_or(zero_mask, eta_n2 == 0)

    fkp[zero_mask] = 0

    for act_pipe in act_pipes:
        act_pipe.init_fkp(fkp=fkp, plots=True)
        act_pipe.init_lweight()

    ref_map = act_pipe.map_t

    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)

    zerr_cut = ZerrCut(zerr_max)
    if do_cut:
        north_cut = AndCut([LRGNorthCut(), zerr_cut])
        south_cut = AndCut([LRGNorthCut(), zerr_cut])
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

    # printlog('making vr lists')
    gal_pipe.make_vr_list()
    # printlog('done')
    for act_pipe in act_pipes:
        cross_pipe = CMBxGalPipe(act_pipe, gal_pipe, gal_mask, output_manager=om)

        # printlog('computing estimator')
        alpha_dict = cross_pipe.compute_estimator(ntrial_mc=0) # only do bootstrap
        # printlog('done')
        
        printlog(f"frequency: {act_pipe.freq}, alpha_ksz_bs: {alpha_dict['a_ksz_bootstrap']:.3e}, alpha_ksz_bs_2: {alpha_dict['a_ksz_bootstrap_2']:.3e}")