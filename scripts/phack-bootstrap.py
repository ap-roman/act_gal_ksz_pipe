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
    om = OutputManager(base_path='output', title='phack-bootstrap', logs=['log',])
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
    
    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t
    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    zerr_grid = np.linspace(0.025, 0.1, 8)
    vr_widths = ['0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']
    do_cuts = [True, False]

    cut_labels = []
    alphas_bs = []

    param_set = list(itertools.product(zerr_grid, do_cuts, vr_widths))

    for zerr_max, do_cut, vr_width in param_set:
        cutstring = f'zerr_max {zerr_max:.3f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
        printlog(cutstring)
        
        zerr_cut = ZerrCut(zerr_max)
        if do_cut:
            north_cut = AndCut([LRGNorthCut(), zerr_cut])
            south_cut = AndCut([LRGNorthCut(), zerr_cut])
        else:
            north_cut = zerr_cut
            south_cut = zerr_cut

        # printlog('getting gal pipe')
        gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
        # printlog('done')

        # printlog('importing data')
        gal_pipe.import_data()
        # printlog('done')

        # printlog('making vr lists')
        gal_pipe.make_vr_list()
        # printlog('done')

        cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)
        cross_pipe.import_nl(nl_path)
        cross_pipe.import_fl(fl_path)

        # printlog('computing estimator')
        alpha_dict = cross_pipe.compute_estimator(ntrial_mc=0) # only do bootstrap
        # printlog('done')

        printlog(alpha_dict)
        alphas_bs.append(alpha_dict['a_ksz_bootstrap_2'])
        cut_labels.append(cutstring)

    printlog('zerr_max, do_lrg_cut, vr_width, alpha_bs_2')
    for alpha, params in zip(alphas_bs, param_set):
        zerr_max, do_cut, vr_width = params
        printlog(f'{zerr_max:.3f}, {do_cut:b}, {vr_width}, {alpha:.3e}')
