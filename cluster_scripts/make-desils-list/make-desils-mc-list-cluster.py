from fnl_pipe.util import OutputManager, get_fname
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import itertools

import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 1024

freq_str = '150'

plots = True

data_path = '/gpfs/aroman/data/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

act_path = data_path + 'act_pub/'
map_path = act_path + f'act_dr5.01_s08s18_AA_f{freq_str}_daynight_map_srcfree.fits' # public
ivar_path = act_path + f'act_dr5.01_s08s18_AA_f{freq_str}_daynight_ivar.fits' # public
beam_path = act_path + f'act_planck_dr5.01_s08s18_f{freq_str}_daynight_beam.txt' # public beam
nl_path = data_path + f'desils/nl_desils_pub_actonly_f{freq_str}_{NTRIAL_NL}.npy'
fl_path = data_path + f'desils/fl_desils_pub_actonly_f{freq_str}_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'

gal_mask_path = data_path + 'vr_source/desils/intersect_sdss_desi_mask.fits'

desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'

mc_list_base = data_path + f'desils/desils_cmass_actonly_f{freq_str}_nmc_{NTRIAL_MC}'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'

zerr_max = 0.1
vr_width = '1.0'


if __name__ == "__main__":
    assert NTRIAL_MC % size == 0
    my_ntrial_mc = NTRIAL_MC // size

    om = OutputManager(base_path='output', title='make-desils-mc-list-cluster', logs=['log'],
                       mpi_rank=rank, mpi_comm=comm)
    printlog = om.printlog
 
    printlog(f'my rank is {rank} of {size} on node {MPI.Get_processor_name()}, comm {comm.Get_name()}')

    mc_list_out = mc_list_base + f'_rank_{rank}.h5'

    # WARN: magic numbers from old optimization
    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t
    printlog(f'rank {rank}: importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    
    north_cut = NullCut()
    south_cut = NullCut()

    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)
    cross_pipe.import_nl(nl_path)
    cross_pipe.import_fl(fl_path)

    cross_pipe.write_mc_list(ntrial_mc=my_ntrial_mc, outpath=mc_list_out)
