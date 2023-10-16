from fnl_pipe.util import OutputManager, get_fname, get_yaml_dict, fequal, ChunkedMaskedReader
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
import sys

import mpi4py


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='make-paper', logs=['log',], replace=True)
    printlog = om.printlog

    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    local_dict = locals()
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(f'{key}: {value}')
        local_dict[key] = value
    printlog('################## DONE ##################')

    printlog = om.printlog

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t
    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)
    desi_mask = enmap.read_map(desi_mask_path)
    sdss_mask = enmap.read_map(sdss_mask_path)
    mask_dict = {'interesect_mask': gal_mask, 'desi_mask': desi_mask, 'sdss_mask': sdss_mask}

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)

    zerr_cut = ZerrCut(zerr_max)
    if do_cut:
        north_cut = AndCut([LRGNorthCut(), zerr_cut])
        south_cut = AndCut([LRGSouthCut(), zerr_cut])
    else:
        north_cut = zerr_cut
        south_cut = zerr_cut

    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()
    row_mask = gal_pipe.cut_inbounds_mask

    cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)

    printlog('importing nl')
    cross_pipe.import_nl(nl_path)
    printlog('importing fl')
    cross_pipe.import_fl(fl_path)

    np.save(beam_out_path, act_pipe_150.beam)
    np.save(lweight_path, act_pipe_150.l_weight)
    np.save(cl_sim_path, cross_pipe.cl_act_sim)
