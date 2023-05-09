from fnl_pipe.util import OutputManager, get_fname
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import itertools

import numpy as np

NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 1024

plots = True

data_path = '/home/aroman/data/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

# act_path = data_path + 'act/'
# map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits' # private
# ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits' # private
# beam_path = act_path + 'beam_f150_daynight.txt' # proprietary beam file
# # nl_path = f'data/nl_desils_{NTRIAL_NL}.npy'
# # fl_path = f'data/fl_desils_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'
# nl_path = f'data/nl_{NTRIAL_NL}.npy'
# fl_path = f'data/fl_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'


act_path = data_path + 'act_pub/'
map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits' # public
ivar_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_ivar.fits' # public
beam_path = act_path + 'act_planck_dr5.01_s08s18_f150_daynight_beam.txt' # public beam
nl_path = f'data/nl_desils_pub_{NTRIAL_NL}.npy'
fl_path = f'data/fl_desils_pub_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'


# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
gal_mask_path = data_path + 'vr_source/desi_ls/intersect_sdss_desi_mask.fits'


# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
# gal_mask_path = data_path + 'vr_source/desi_ls/intersect_sdss_desi_mask.h5'
desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'

mc_list_out = data_path + f'vr_source/desils/desils_cmass_nmc_{NTRIAL_MC}.h5'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'

# catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'
catalog_base = data_path + 'vr_summaries/desils/'
catalog_files = get_files(catalog_base)


zerr_max = 0.01
vr_width = '1.0'


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='make-desils-mc-list', logs=['log'])
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

    printlog(f'importing catalog files {catalog_files}')

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    
    north_cut = NullCut()
    south_cut = NullCut()

    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)
    cross_pipe.import_nl(nl_path)
    cross_pipe.import_fl(fl_path)

    cross_pipe.write_mc_list(ntrial_mc=NTRIAL_MC, outpath=mc_list_out)
